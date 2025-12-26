"""Configuration management for Paperlinse."""

import os
import json
import base64
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet

CONFIG_DIR = Path(__file__).parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "settings.json"
KEY_FILE = CONFIG_DIR / ".key"


def get_or_create_key() -> bytes:
    """Get or create encryption key for sensitive data."""
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_bytes(key)
    KEY_FILE.chmod(0o600)
    return key


def encrypt_value(value: str) -> str:
    """Encrypt a sensitive value."""
    if not value:
        return ""
    f = Fernet(get_or_create_key())
    return base64.urlsafe_b64encode(f.encrypt(value.encode())).decode()


def decrypt_value(encrypted: str) -> str:
    """Decrypt a sensitive value."""
    if not encrypted:
        return ""
    try:
        f = Fernet(get_or_create_key())
        return f.decrypt(base64.urlsafe_b64decode(encrypted.encode())).decode()
    except Exception:
        return ""


class ShareCredentials(BaseModel):
    """Credentials for network share access."""
    username: str = ""
    password: str = ""  # Stored encrypted
    domain: str = ""


class IncomingShareConfig(BaseModel):
    """Configuration for the incoming document share."""
    path: str = ""
    credentials: ShareCredentials = Field(default_factory=ShareCredentials)
    poll_interval_seconds: int = 30


class StorageConfig(BaseModel):
    """Configuration for document storage."""
    path: str = ""
    credentials: ShareCredentials = Field(default_factory=ShareCredentials)


class AppConfig(BaseModel):
    """Main application configuration."""
    incoming_share: IncomingShareConfig = Field(default_factory=IncomingShareConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    def save(self) -> None:
        """Save configuration to disk with encrypted passwords."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create a copy for saving with encrypted passwords
        data = self.model_dump()
        
        # Encrypt passwords before saving
        if data["incoming_share"]["credentials"]["password"]:
            data["incoming_share"]["credentials"]["password"] = encrypt_value(
                data["incoming_share"]["credentials"]["password"]
            )
        if data["storage"]["credentials"]["password"]:
            data["storage"]["credentials"]["password"] = encrypt_value(
                data["storage"]["credentials"]["password"]
            )
        
        CONFIG_FILE.write_text(json.dumps(data, indent=2))
        CONFIG_FILE.chmod(0o600)
    
    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from disk and decrypt passwords."""
        if not CONFIG_FILE.exists():
            return cls()
        
        try:
            data = json.loads(CONFIG_FILE.read_text())
            
            # Decrypt passwords after loading
            if data.get("incoming_share", {}).get("credentials", {}).get("password"):
                data["incoming_share"]["credentials"]["password"] = decrypt_value(
                    data["incoming_share"]["credentials"]["password"]
                )
            if data.get("storage", {}).get("credentials", {}).get("password"):
                data["storage"]["credentials"]["password"] = decrypt_value(
                    data["storage"]["credentials"]["password"]
                )
            
            return cls(**data)
        except Exception as e:
            print(f"Error loading config: {e}")
            return cls()
