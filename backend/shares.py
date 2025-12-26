"""SMB/CIFS share mounting utilities."""

import subprocess
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import hashlib


MOUNT_BASE = Path("/mnt/paperlinse")


def is_network_path(path: str) -> bool:
    """Check if path looks like a network share."""
    normalized = path.replace("\\", "/")
    # Match //server/share, \\server\share, or /IP/share patterns
    if normalized.startswith("//"):
        return True
    if path.startswith("\\\\"):
        return True
    # Also detect /10.x.x.x/... or /192.168.x.x/... style paths (missing leading slash)
    if re.match(r'^/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/', normalized):
        return True
    # Also detect hostname patterns like /server/share
    if re.match(r'^/[a-zA-Z0-9][-a-zA-Z0-9]*/', normalized) and not Path(normalized).exists():
        # Only treat as network if local path doesn't exist
        return True
    return False


def normalize_share_path(path: str) -> str:
    """Normalize share path to //server/share format."""
    # Convert backslashes to forward slashes
    path = path.replace("\\", "/")
    # Ensure it starts with //
    if not path.startswith("//"):
        if path.startswith("/"):
            path = "/" + path
        else:
            path = "//" + path
    return path


def get_mount_point(share_path: str) -> Path:
    """Generate a consistent mount point for a share path."""
    # Create a hash-based mount point name to handle special characters
    normalized = normalize_share_path(share_path)
    # Extract server and share name for readable mount point
    parts = normalized.strip("/").split("/")
    if len(parts) >= 2:
        mount_name = f"{parts[0]}_{parts[1]}"
    else:
        mount_name = parts[0] if parts else "share"
    # Add hash suffix for uniqueness
    hash_suffix = hashlib.md5(normalized.encode()).hexdigest()[:8]
    return MOUNT_BASE / f"{mount_name}_{hash_suffix}"


def is_mounted(mount_point: Path) -> bool:
    """Check if a path is currently a mount point."""
    if not mount_point.exists():
        return False
    try:
        result = subprocess.run(
            ["mountpoint", "-q", str(mount_point)],
            capture_output=True
        )
        return result.returncode == 0
    except Exception:
        return False


def check_mount_capability() -> Tuple[bool, str]:
    """Check if we can mount CIFS shares."""
    # Check if mount.cifs exists
    result = subprocess.run(["which", "mount.cifs"], capture_output=True)
    if result.returncode != 0:
        return False, "mount.cifs not found. Install cifs-utils: apt install cifs-utils"
    
    # Check if we're in a restricted container
    try:
        # Try to check /proc/mounts access
        with open("/proc/mounts", "r") as f:
            f.read()
    except:
        return False, "Cannot access /proc/mounts. May be running in restricted container."
    
    return True, "OK"


def mount_share(
    share_path: str,
    username: str = "",
    password: str = "",
    domain: str = ""
) -> Tuple[bool, str, Optional[Path]]:
    """
    Mount an SMB/CIFS share.
    
    Returns:
        Tuple of (success, message, mount_point)
    """
    # Check mount capability first
    can_mount, mount_msg = check_mount_capability()
    if not can_mount:
        return False, mount_msg, None
    
    normalized = normalize_share_path(share_path)
    mount_point = get_mount_point(share_path)
    
    # Check if already mounted
    if is_mounted(mount_point):
        return True, f"Share already mounted at {mount_point}", mount_point
    
    # Create mount point directory
    try:
        MOUNT_BASE.mkdir(parents=True, exist_ok=True)
        mount_point.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return False, "Permission denied creating mount point. Run as root or with sudo.", None
    except Exception as e:
        return False, f"Failed to create mount point: {e}", None
    
    # Build mount options - try different SMB versions
    base_opts = ["soft", "rsize=65536", "wsize=65536"]
    
    if username:
        base_opts.append(f"username={username}")
        if password:
            base_opts.append(f"password={password}")
        if domain:
            base_opts.append(f"domain={domain}")
    else:
        base_opts.append("guest")
    
    # Try different SMB versions (3.0, 2.1, 2.0, 1.0)
    smb_versions = ["3.0", "2.1", "2.0", "1.0"]
    last_error = ""
    
    for vers in smb_versions:
        mount_opts = base_opts + [f"vers={vers}"]
        opts_str = ",".join(mount_opts)
        
        try:
            result = subprocess.run(
                ["mount", "-t", "cifs", normalized, str(mount_point), "-o", opts_str],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, f"Successfully mounted {normalized} at {mount_point} (SMB {vers})", mount_point
            else:
                last_error = result.stderr.strip() or result.stdout.strip() or "Unknown error"
                # Check for specific errors that won't be fixed by trying different versions
                if "Operation not permitted" in last_error:
                    break  # Container/capability issue, no point trying other versions
                if "Permission denied" in last_error or "Access denied" in last_error:
                    break  # Auth issue
                    
        except subprocess.TimeoutExpired:
            last_error = "Mount timed out. Check network connectivity."
            break
        except Exception as e:
            last_error = str(e)
    
    # Clean up empty mount point
    try:
        mount_point.rmdir()
    except:
        pass
    
    # Provide helpful error message
    if "Operation not permitted" in last_error:
        return False, (
            f"Mount failed: {last_error}\n"
            "This usually means:\n"
            "- Running in a container without --privileged or --cap-add SYS_ADMIN\n"
            "- AppArmor/SELinux blocking the mount\n"
            "Try: Run with --privileged flag or mount the share on the host"
        ), None
    
    return False, f"Mount failed: {last_error}", None


def unmount_share(mount_point: Path) -> Tuple[bool, str]:
    """Unmount a share."""
    if not is_mounted(mount_point):
        return True, "Not mounted"
    
    try:
        result = subprocess.run(
            ["umount", str(mount_point)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            try:
                mount_point.rmdir()
            except:
                pass
            return True, "Successfully unmounted"
        else:
            return False, f"Unmount failed: {result.stderr.strip()}"
    except Exception as e:
        return False, f"Unmount error: {str(e)}"


def get_effective_path(
    path: str,
    username: str = "",
    password: str = "",
    domain: str = ""
) -> Tuple[bool, str, Optional[Path]]:
    """
    Get the effective local path for a given path.
    For network shares, mounts them first.
    For local paths, returns them directly.
    
    Returns:
        Tuple of (success, message, effective_path)
    """
    if is_network_path(path):
        return mount_share(path, username, password, domain)
    else:
        # Local path
        local_path = Path(path)
        if local_path.exists():
            return True, f"Local path: {path}", local_path
        else:
            return False, f"Path does not exist: {path}", None
