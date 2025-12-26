"""Paperlinse - Personal Document OCR and Management System."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import os

from config import AppConfig, ShareCredentials, IncomingShareConfig, StorageConfig
from shares import get_effective_path, is_network_path, get_mount_point, is_mounted

app = FastAPI(
    title="Paperlinse",
    description="Personal Document OCR and Management System",
    version="0.1.0"
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class CredentialsInput(BaseModel):
    username: str = ""
    password: str = ""
    domain: str = ""


class IncomingShareInput(BaseModel):
    path: str
    credentials: CredentialsInput
    poll_interval_seconds: int = 30


class StorageInput(BaseModel):
    path: str
    credentials: CredentialsInput


class ConfigInput(BaseModel):
    incoming_share: IncomingShareInput
    storage: StorageInput


class ConfigOutput(BaseModel):
    incoming_share: IncomingShareInput
    storage: StorageInput
    
    class Config:
        # Mask passwords in output
        pass


class TestConnectionRequest(BaseModel):
    path: str
    credentials: CredentialsInput


class TestConnectionResponse(BaseModel):
    success: bool
    message: str
    writable: bool = False
    mount_point: Optional[str] = None


# API Routes
@app.get("/api/config")
async def get_config() -> dict:
    """Get current configuration (passwords masked)."""
    config = AppConfig.load()
    return {
        "incoming_share": {
            "path": config.incoming_share.path,
            "credentials": {
                "username": config.incoming_share.credentials.username,
                "password": "***" if config.incoming_share.credentials.password else "",
                "domain": config.incoming_share.credentials.domain,
            },
            "poll_interval_seconds": config.incoming_share.poll_interval_seconds,
        },
        "storage": {
            "path": config.storage.path,
            "credentials": {
                "username": config.storage.credentials.username,
                "password": "***" if config.storage.credentials.password else "",
                "domain": config.storage.credentials.domain,
            },
        },
    }


@app.post("/api/config")
async def save_config(config_input: ConfigInput) -> dict:
    """Save configuration."""
    # Load existing config to preserve encrypted passwords if not changed
    existing = AppConfig.load()
    
    incoming_password = config_input.incoming_share.credentials.password
    if incoming_password == "***":
        incoming_password = existing.incoming_share.credentials.password
    
    storage_password = config_input.storage.credentials.password
    if storage_password == "***":
        storage_password = existing.storage.credentials.password
    
    config = AppConfig(
        incoming_share=IncomingShareConfig(
            path=config_input.incoming_share.path,
            credentials=ShareCredentials(
                username=config_input.incoming_share.credentials.username,
                password=incoming_password,
                domain=config_input.incoming_share.credentials.domain,
            ),
            poll_interval_seconds=config_input.incoming_share.poll_interval_seconds,
        ),
        storage=StorageConfig(
            path=config_input.storage.path,
            credentials=ShareCredentials(
                username=config_input.storage.credentials.username,
                password=storage_password,
                domain=config_input.storage.credentials.domain,
            ),
        ),
    )
    
    config.save()
    return {"status": "ok", "message": "Configuration saved successfully"}


@app.post("/api/test-connection")
async def test_connection(request: TestConnectionRequest) -> TestConnectionResponse:
    """Test connection to a path (local or network share)."""
    path = request.path
    creds = request.credentials
    
    try:
        # Use the shares module to handle both local and network paths
        success, message, effective_path = get_effective_path(
            path,
            username=creds.username,
            password=creds.password,
            domain=creds.domain
        )
        
        if not success:
            return TestConnectionResponse(
                success=False,
                message=message,
                writable=False
            )
        
        # Check if the effective path is a directory
        if not effective_path.is_dir():
            return TestConnectionResponse(
                success=False,
                message=f"Path is not a directory: {effective_path}",
                writable=False
            )
        
        # Check if writable
        writable = os.access(str(effective_path), os.W_OK)
        
        # Include mount point info for network shares
        mount_info = ""
        if is_network_path(path):
            mount_info = f" (mounted at {effective_path})"
        
        return TestConnectionResponse(
            success=True,
            message=f"Successfully connected to {path}{mount_info}",
            writable=writable,
            mount_point=str(effective_path) if is_network_path(path) else None
        )
        
    except PermissionError:
        return TestConnectionResponse(
            success=False,
            message=f"Permission denied accessing: {path}",
            writable=False
        )
    except Exception as e:
        return TestConnectionResponse(
            success=False,
            message=f"Error accessing path: {str(e)}",
            writable=False
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "paperlinse"}


# Serve static frontend files
FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend SPA."""
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
else:
    @app.get("/")
    async def root():
        return {"message": "Paperlinse API - Frontend not built yet"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
