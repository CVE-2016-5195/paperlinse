"""Paperlinse - Personal Document OCR and Management System."""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from config import AppConfig, ShareCredentials, IncomingShareConfig, StorageConfig, LLMConfig, ProcessingConfig, VisionLLMConfig, DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT_DE, DEFAULT_USER_PROMPT_EN, VISION_LLM_MODELS
from shares import get_effective_path, is_network_path, get_mount_point, is_mounted
from database import Database, DocumentModel, DocumentStatus, PageModel
from processor import get_processor, DocumentProcessor, get_processing_tracker
from events import get_broadcaster, emit_queue_updated, emit_stats_updated, emit_document_approved, emit_document_rejected

logger = logging.getLogger(__name__)

# Background processor task
_processor_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _processor_task
    
    # Startup
    logger.info("Initializing Paperlinse...")
    
    # Start event broadcaster
    broadcaster = get_broadcaster()
    await broadcaster.start()
    
    # Initialize database
    try:
        await Database.init_schema()
        logger.info("Database schema initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logger.warning("Continuing without database - some features will be unavailable")
    
    # Clean up queue items and documents for files that no longer exist
    try:
        processor = get_processor()
        cleanup_result = await processor.cleanup_missing_files()
        if cleanup_result['queue_removed'] > 0 or cleanup_result['documents_removed'] > 0:
            logger.info(f"Cleanup: {cleanup_result['queue_removed']} queue items, {cleanup_result['documents_removed']} documents removed")
    except Exception as e:
        logger.warning(f"Queue cleanup failed: {e}")
    
    # Start background processor
    config = AppConfig.load()
    if config.incoming_share.path:
        processor = get_processor()
        interval = config.incoming_share.poll_interval_seconds or 30
        _processor_task = asyncio.create_task(
            processor.start_background_processing(interval)
        )
        logger.info(f"Background processor started (interval: {interval}s)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Paperlinse...")
    
    # Stop event broadcaster
    await broadcaster.stop()
    
    if _processor_task:
        processor = get_processor()
        processor.stop_background_processing()
        _processor_task.cancel()
        try:
            await _processor_task
        except asyncio.CancelledError:
            pass
    
    await Database.close()
    logger.info("Paperlinse shutdown complete")


app = FastAPI(
    title="Paperlinse",
    description="Personal Document OCR and Management System",
    version="0.2.0",
    lifespan=lifespan
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response models
# ============================================================================

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


class TestConnectionRequest(BaseModel):
    path: str
    credentials: CredentialsInput


class TestConnectionResponse(BaseModel):
    success: bool
    message: str
    writable: bool = False
    mount_point: Optional[str] = None


class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    modified: float


class IncomingFilesResponse(BaseModel):
    success: bool
    message: str
    files: list[FileInfo] = []
    count: int = 0


class StatsResponse(BaseModel):
    pending: int = 0
    processing: int = 0
    awaiting_approval: int = 0
    approved: int = 0
    rejected: int = 0
    error: int = 0
    total: int = 0


class PageResponse(BaseModel):
    id: int
    page_number: int
    ocr_text: str
    confidence: float
    image_path: str
    original_filename: str


class IdentifierResponse(BaseModel):
    """Response for a document identifier."""
    id: Optional[int] = None
    identifier_type: str
    identifier_value: str


class DocumentResponse(BaseModel):
    id: int
    topic: str
    summary: str
    document_date: Optional[str] = None
    sender: str
    receiver: str
    document_type: str
    language: str
    folder_path: str
    status: str
    processing_error: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    processed_at: Optional[str] = None
    # Enhanced metadata fields
    file_hash: Optional[str] = None
    mime_type: Optional[str] = None
    page_count: int = 0
    identifiers: list[IdentifierResponse] = []
    iban: Optional[str] = None
    bic: Optional[str] = None
    due_date: Optional[str] = None
    pages: list[PageResponse] = []


class RelatedDocumentResponse(BaseModel):
    """Response for a related document."""
    id: int
    topic: str
    sender: str
    document_date: Optional[str] = None
    document_type: str
    status: str
    matching_identifiers: list[dict] = []


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class DocumentUpdateRequest(BaseModel):
    topic: Optional[str] = None
    summary: Optional[str] = None
    sender: Optional[str] = None
    receiver: Optional[str] = None
    document_type: Optional[str] = None
    document_date: Optional[str] = None


class ProcessingModeResponse(BaseModel):
    mode: str  # 'manual' or 'automatic'


class ProcessingModeRequest(BaseModel):
    mode: str  # 'manual' or 'automatic'


class ProcessFileRequest(BaseModel):
    file_path: str


class IncomingFileStatus(BaseModel):
    """Status of a file in the incoming folder."""
    name: str
    path: str
    size: int
    modified: float
    queue_status: Optional[str] = None  # queued, processing, completed, error
    document_id: Optional[int] = None
    document_status: Optional[str] = None  # processing, awaiting_approval, approved, rejected, error
    topic: Optional[str] = None
    sender: Optional[str] = None
    error_message: Optional[str] = None
    thumbnail_url: Optional[str] = None


class IncomingFilesStatusResponse(BaseModel):
    """Response with all incoming files and their statuses."""
    success: bool
    message: str
    files: list[IncomingFileStatus] = []
    count: int = 0
    auto_approve: bool = False


class LLMConfigResponse(BaseModel):
    """Response with LLM configuration."""
    model: str
    available_models: list[str] = []
    system_prompt: str
    user_prompt_de: str
    user_prompt_en: str


class LLMConfigRequest(BaseModel):
    """Request to update LLM configuration."""
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt_de: Optional[str] = None
    user_prompt_en: Optional[str] = None


class ProcessingConfigResponse(BaseModel):
    """Response with processing configuration."""
    concurrent_workers: int
    batch_size: int


class ProcessingConfigRequest(BaseModel):
    """Request to update processing configuration."""
    concurrent_workers: int
    batch_size: int


class VisionLLMModelInfo(BaseModel):
    """Information about a Vision LLM model."""
    id: str
    name: str
    size_gb: float
    description: str
    installed: bool = False
    current: bool = False


class VisionLLMConfigResponse(BaseModel):
    """Response with Vision LLM configuration."""
    enabled: bool
    model_path: str
    device: str
    max_pixels: int = 512 * 28 * 28  # Default ~401k pixels
    min_pixels: int = 4 * 28 * 28    # Default ~3.1k pixels
    prompt_de: str = ""  # Vision LLM German prompt
    prompt_en: str = ""  # Vision LLM English prompt
    available: bool = False
    available_models: list[VisionLLMModelInfo] = []
    status_message: str = ""


class VisionLLMConfigRequest(BaseModel):
    """Request to update Vision LLM configuration."""
    enabled: Optional[bool] = None
    model_path: Optional[str] = None
    device: Optional[str] = None
    max_pixels: Optional[int] = None
    min_pixels: Optional[int] = None
    prompt_de: Optional[str] = None  # Vision LLM German prompt
    prompt_en: Optional[str] = None  # Vision LLM English prompt


class ProcessingProgressResponse(BaseModel):
    """Response with current processing progress."""
    is_active: bool
    queued: int  # Files waiting to be processed
    processing: int  # Files currently being processed
    completed: int  # Files completed (awaiting approval or approved)
    error: int  # Files with errors
    total: int  # Total files in incoming folder
    current_files: list[str] = []  # Names of files currently being processed
    # Detailed step info for each file being processed
    processing_details: list[dict] = []  # [{file_name, step, step_detail, progress_percent}]
    llm_model: str = ""  # Currently configured LLM model


class CancelProcessingRequest(BaseModel):
    """Request to cancel processing."""
    file_path: Optional[str] = None  # Specific file to cancel, or None for all


class CancelProcessingResponse(BaseModel):
    """Response for cancel processing request."""
    success: bool
    message: str
    cancelled_count: int = 0
    cancelled_files: list[str] = []


class ModelInfo(BaseModel):
    """Information about an LLM model."""
    id: str
    name: str
    size_mb: int
    description: str
    model_type: str = "text"  # "text" for Ollama models, "vision" for OpenVINO vision models
    installed: bool = False
    current: bool = False
    pulling: bool = False
    pull_progress: int = 0


class ModelsListResponse(BaseModel):
    """Response with list of available models."""
    models: list[ModelInfo]
    current_model: str
    ollama_available: bool


class ModelSwitchRequest(BaseModel):
    """Request to switch to a different model."""
    download_if_missing: bool = True


# ============================================================================
# Helper functions
# ============================================================================

def document_to_response(doc: DocumentModel) -> DocumentResponse:
    """Convert DocumentModel to API response."""
    return DocumentResponse(
        id=doc.id or 0,
        topic=doc.topic,
        summary=doc.summary,
        document_date=doc.document_date.isoformat() if doc.document_date else None,
        sender=doc.sender,
        receiver=doc.receiver,
        document_type=doc.document_type,
        language=doc.language,
        folder_path=doc.folder_path,
        status=doc.status.value,
        processing_error=doc.processing_error,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
        updated_at=doc.updated_at.isoformat() if doc.updated_at else None,
        processed_at=doc.processed_at.isoformat() if doc.processed_at else None,
        file_hash=doc.file_hash or None,
        mime_type=doc.mime_type or None,
        page_count=doc.page_count or 0,
        identifiers=[
            IdentifierResponse(
                id=ident.id,
                identifier_type=ident.identifier_type,
                identifier_value=ident.identifier_value
            )
            for ident in doc.identifiers
        ],
        iban=doc.iban or None,
        bic=doc.bic or None,
        due_date=doc.due_date.isoformat() if doc.due_date else None,
        pages=[
            PageResponse(
                id=p.id or 0,
                page_number=p.page_number,
                ocr_text=p.ocr_text,
                confidence=p.confidence,
                image_path=p.image_path,
                original_filename=p.original_filename
            )
            for p in doc.pages
        ]
    )


# Supported document file extensions
DOCUMENT_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}


def get_document_files(directory: Path) -> list[FileInfo]:
    """Get all document files from a directory."""
    files = []
    try:
        for item in directory.iterdir():
            if item.is_file() and item.suffix.lower() in DOCUMENT_EXTENSIONS:
                stat = item.stat()
                files.append(FileInfo(
                    name=item.name,
                    path=str(item),
                    size=stat.st_size,
                    modified=stat.st_mtime
                ))
    except Exception as e:
        logger.error(f"Error reading directory: {e}")
    return sorted(files, key=lambda f: f.modified, reverse=True)


# ============================================================================
# Configuration API Routes
# ============================================================================

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


@app.get("/api/config/llm")
async def get_llm_config() -> LLMConfigResponse:
    """Get LLM prompt configuration."""
    from llm import get_ollama_client, OLLAMA_MODEL
    
    config = AppConfig.load()
    
    # Get the effective model (from config or environment)
    effective_model = config.llm.model if config.llm.model else OLLAMA_MODEL
    
    # Get available models from Ollama
    available_models = []
    try:
        client = await get_ollama_client()
        available_models = await client.list_models()
    except Exception as e:
        logger.warning(f"Could not fetch available models: {e}")
    
    return LLMConfigResponse(
        model=effective_model,
        available_models=available_models,
        system_prompt=config.llm.system_prompt,
        user_prompt_de=config.llm.user_prompt_de,
        user_prompt_en=config.llm.user_prompt_en
    )


@app.post("/api/config/llm")
async def save_llm_config(request: LLMConfigRequest) -> dict:
    """Save LLM prompt configuration."""
    config = AppConfig.load()
    
    # Update only provided fields
    new_model = request.model if request.model is not None else config.llm.model
    new_system_prompt = request.system_prompt if request.system_prompt is not None else config.llm.system_prompt
    new_user_prompt_de = request.user_prompt_de if request.user_prompt_de is not None else config.llm.user_prompt_de
    new_user_prompt_en = request.user_prompt_en if request.user_prompt_en is not None else config.llm.user_prompt_en
    
    config.llm = LLMConfig(
        model=new_model,
        system_prompt=new_system_prompt,
        user_prompt_de=new_user_prompt_de,
        user_prompt_en=new_user_prompt_en
    )
    config.save()
    
    # Update the global Ollama client with the new model if it changed
    if request.model is not None:
        from llm import get_ollama_client
        client = await get_ollama_client()
        client.model = new_model if new_model else client.model
    
    return {"status": "ok", "message": "LLM configuration saved successfully"}


@app.post("/api/config/llm/reset")
async def reset_llm_config() -> dict:
    """Reset LLM prompts to defaults (keeps the model setting)."""
    config = AppConfig.load()
    config.llm = LLMConfig(
        model=config.llm.model,  # Keep the model setting
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt_de=DEFAULT_USER_PROMPT_DE,
        user_prompt_en=DEFAULT_USER_PROMPT_EN
    )
    config.save()
    return {"status": "ok", "message": "LLM configuration reset to defaults"}


# ============================================================================
# Vision LLM Configuration API Routes
# ============================================================================

@app.get("/api/config/vision-llm")
async def get_vision_llm_config() -> VisionLLMConfigResponse:
    """Get Vision LLM configuration and status."""
    from vision_llm import is_vision_llm_available, get_vision_llm_status
    
    config = AppConfig.load()
    
    # Check if vision LLM is available
    available = False
    status_message = ""
    try:
        available = is_vision_llm_available()
        status = get_vision_llm_status()
        # Get error message from status (key is "error", not "message")
        status_message = status.get("error", "") or ""
    except Exception as e:
        status_message = f"Error checking Vision LLM status: {e}"
    
    # Build list of available models with installation status
    available_models = []
    for model_id, display_name, size_gb, description in VISION_LLM_MODELS:
        # Check if this model is installed by looking for model files
        model_dir_name = model_id.split("/")[-1]  # e.g., "Qwen3-VL-2B-Instruct-int4"
        model_path = Path("models") / model_dir_name
        is_installed = model_path.exists() and (
            (model_path / "openvino_model.xml").exists() or 
            (model_path / "openvino_language_model.xml").exists()
        )
        
        # Check if this is the currently active model
        is_current = (
            config.vision_llm.enabled and 
            is_installed and 
            model_dir_name.lower() in config.vision_llm.model_path.lower()
        )
        
        available_models.append(VisionLLMModelInfo(
            id=model_id,
            name=display_name,
            size_gb=size_gb,
            description=description,
            installed=is_installed,
            current=is_current
        ))
    
    # Import default prompts for fallback
    from config import DEFAULT_VISION_PROMPT_DE, DEFAULT_VISION_PROMPT_EN
    
    return VisionLLMConfigResponse(
        enabled=config.vision_llm.enabled,
        model_path=config.vision_llm.model_path,
        device=config.vision_llm.device,
        max_pixels=getattr(config.vision_llm, 'max_pixels', 512 * 28 * 28),
        min_pixels=getattr(config.vision_llm, 'min_pixels', 4 * 28 * 28),
        prompt_de=getattr(config.vision_llm, 'prompt_de', DEFAULT_VISION_PROMPT_DE),
        prompt_en=getattr(config.vision_llm, 'prompt_en', DEFAULT_VISION_PROMPT_EN),
        available=available,
        available_models=available_models,
        status_message=status_message
    )


@app.post("/api/config/vision-llm")
async def save_vision_llm_config(request: VisionLLMConfigRequest) -> dict:
    """Save Vision LLM configuration."""
    from vision_llm import VisionLLMClient, get_vision_llm_client
    from config import DEFAULT_VISION_PROMPT_DE, DEFAULT_VISION_PROMPT_EN
    
    config = AppConfig.load()
    
    # Update only provided fields
    new_enabled = request.enabled if request.enabled is not None else config.vision_llm.enabled
    new_model_path = request.model_path if request.model_path is not None else config.vision_llm.model_path
    new_device = request.device if request.device is not None else config.vision_llm.device
    new_max_pixels = request.max_pixels if request.max_pixels is not None else getattr(config.vision_llm, 'max_pixels', 512 * 28 * 28)
    new_min_pixels = request.min_pixels if request.min_pixels is not None else getattr(config.vision_llm, 'min_pixels', 4 * 28 * 28)
    new_prompt_de = request.prompt_de if request.prompt_de is not None else getattr(config.vision_llm, 'prompt_de', DEFAULT_VISION_PROMPT_DE)
    new_prompt_en = request.prompt_en if request.prompt_en is not None else getattr(config.vision_llm, 'prompt_en', DEFAULT_VISION_PROMPT_EN)
    
    # Validate device
    valid_devices = ["CPU", "GPU", "AUTO"]
    if new_device.upper() not in valid_devices:
        raise HTTPException(status_code=400, detail=f"Invalid device: {new_device}. Must be one of: {valid_devices}")
    
    # Validate pixel limits
    if new_max_pixels < new_min_pixels:
        raise HTTPException(status_code=400, detail="max_pixels must be greater than or equal to min_pixels")
    if new_max_pixels < 28 * 28:
        raise HTTPException(status_code=400, detail="max_pixels must be at least 784 (28x28)")
    
    # Check if model needs to be reloaded (device or model_path changed)
    old_device = config.vision_llm.device
    old_model_path = config.vision_llm.model_path
    needs_reload = (new_device.upper() != old_device) or (new_model_path != old_model_path)
    
    config.vision_llm = VisionLLMConfig(
        enabled=new_enabled,
        model_path=new_model_path,
        device=new_device.upper(),
        max_pixels=new_max_pixels,
        min_pixels=new_min_pixels,
        prompt_de=new_prompt_de,
        prompt_en=new_prompt_en
    )
    config.save()
    
    # Reset the Vision LLM singleton if device or model changed so it reloads with new settings
    if needs_reload and new_enabled:
        logger.info(f"Vision LLM config changed (device: {old_device} -> {new_device.upper()}, path: {old_model_path} -> {new_model_path}), resetting model...")
        VisionLLMClient.reset_instance()
        # Also reset the module-level global
        import vision_llm
        vision_llm._vision_client = None
    
    return {"status": "ok", "message": "Vision LLM configuration saved successfully"}


@app.post("/api/config/vision-llm/reset-prompts")
async def reset_vision_llm_prompts() -> dict:
    """Reset Vision LLM prompts to defaults (keeps other settings)."""
    from config import DEFAULT_VISION_PROMPT_DE, DEFAULT_VISION_PROMPT_EN
    
    config = AppConfig.load()
    config.vision_llm.prompt_de = DEFAULT_VISION_PROMPT_DE
    config.vision_llm.prompt_en = DEFAULT_VISION_PROMPT_EN
    config.save()
    return {"status": "ok", "message": "Vision LLM prompts reset to defaults"}


@app.get("/api/vision-llm/status")
async def get_vision_llm_status_endpoint() -> dict:
    """Get detailed Vision LLM status."""
    from vision_llm import get_vision_llm_status, is_vision_llm_available
    
    try:
        status = get_vision_llm_status()
        available = is_vision_llm_available()
        
        return {
            "available": available,
            "status": status.get("status", "unknown"),
            "message": status.get("message", ""),
            "model_loaded": status.get("model_loaded", False),
            "model_path": status.get("model_path", ""),
            "device": status.get("device", "")
        }
    except Exception as e:
        logger.error(f"Error getting Vision LLM status: {e}")
        return {
            "available": False,
            "status": "error",
            "message": str(e),
            "model_loaded": False
        }


@app.get("/api/vision-llm/models")
async def list_vision_llm_models() -> dict:
    """List available Vision LLM models."""
    config = AppConfig.load()
    
    models = []
    for model_id, display_name, size_gb, description in VISION_LLM_MODELS:
        # Check if this model is installed
        is_installed = False
        is_current = False
        if config.vision_llm.model_path:
            model_path = Path(config.vision_llm.model_path)
            if model_path.exists() and model_id.lower() in str(model_path).lower():
                is_installed = True
                is_current = True
        
        models.append({
            "id": model_id,
            "name": display_name,
            "size_gb": size_gb,
            "description": description,
            "installed": is_installed,
            "current": is_current,
            "download_command": f"huggingface-cli download {model_id} --local-dir models/{model_id.split('/')[-1]}"
        })
    
    return {
        "models": models,
        "current_model_path": config.vision_llm.model_path,
        "enabled": config.vision_llm.enabled
    }


# ============================================================================
# Model Management API Routes
# ============================================================================

@app.get("/api/models")
async def list_models() -> ModelsListResponse:
    """List all recommended models with their installation status."""
    from llm import get_current_model, list_available_models, get_model_pull_status
    from config import RECOMMENDED_MODELS
    
    config = AppConfig.load()
    current_model = get_current_model()
    
    # Get installed models from Ollama (for text models)
    installed_models = []
    ollama_available = True
    try:
        installed_models = await list_available_models()
        installed_names = {m["name"] for m in installed_models}
        # Also match without tag (e.g., "qwen3:0.6b" matches "qwen3:0.6b")
        installed_base_names = {m["name"].split(":")[0] for m in installed_models}
    except Exception as e:
        logger.error(f"Error checking Ollama: {e}")
        installed_names = set()
        installed_base_names = set()
        ollama_available = False
    
    models = []
    for model_tuple in RECOMMENDED_MODELS:
        # Handle both 4-tuple (legacy) and 5-tuple (with model_type) formats
        if len(model_tuple) >= 5:
            model_id, display_name, size_mb, description, model_type = model_tuple[:5]
        else:
            model_id, display_name, size_mb, description = model_tuple[:4]
            model_type = "text"
        
        is_installed = False
        is_current = False
        is_pulling = False
        pull_progress = 0
        
        if model_type == "vision":
            # Vision models: check if downloaded via HuggingFace
            # Model is installed if the model_path exists and contains OpenVINO model files
            model_dir_name = model_id.split("/")[-1]  # e.g., "Qwen3-VL-2B-Instruct-int4"
            model_path = Path("models") / model_dir_name
            # Check for any openvino model file (language model is the main one)
            is_installed = model_path.exists() and (
                (model_path / "openvino_model.xml").exists() or 
                (model_path / "openvino_language_model.xml").exists()
            )
            
            # Vision model is "current" if vision_llm is enabled and using this model
            if config.vision_llm.enabled and model_dir_name.lower() in config.vision_llm.model_path.lower():
                is_current = True
        else:
            # Text models: check Ollama
            is_installed = model_id in installed_names or any(
                m["name"].startswith(model_id.split(":")[0] + ":") and model_id in m["name"]
                for m in installed_models
            )
            # Text model is "current" only if vision LLM is NOT enabled
            is_current = (model_id == current_model) and not config.vision_llm.enabled
            
            # Check if this model is currently being pulled
            pull_status = get_model_pull_status(model_id)
            is_pulling = pull_status is not None and pull_status.get("status") not in ("success", "error", None)
            pull_progress = pull_status.get("percent", 0) if is_pulling else 0
        
        models.append(ModelInfo(
            id=model_id,
            name=display_name,
            size_mb=size_mb,
            description=description,
            model_type=model_type,
            installed=is_installed,
            current=is_current,
            pulling=is_pulling,
            pull_progress=pull_progress
        ))
    
    return ModelsListResponse(
        models=models,
        current_model=current_model,
        ollama_available=ollama_available
    )


@app.post("/api/models/{model_id:path}/pull")
async def pull_model_endpoint(model_id: str):
    """
    Pull/download a model.
    For text models: uses Ollama registry.
    For vision models: uses HuggingFace CLI.
    Returns a Server-Sent Events stream with progress updates.
    """
    from fastapi.responses import StreamingResponse
    from llm import pull_model, get_model_pull_status
    from config import RECOMMENDED_MODELS
    import json
    import asyncio
    import subprocess
    
    # Determine if this is a vision model
    model_type = "text"
    for model_tuple in RECOMMENDED_MODELS:
        if model_tuple[0] == model_id:
            if len(model_tuple) >= 5:
                model_type = model_tuple[4]
            break
    
    if model_type == "vision":
        # Vision model: download via HuggingFace CLI
        async def generate_hf():
            try:
                model_dir_name = model_id.split("/")[-1]
                local_dir = Path("models") / model_dir_name
                local_dir.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Starting vision model download: {model_id} -> {local_dir}")
                yield f"data: {json.dumps({'status': 'downloading', 'message': f'Starting download of {model_id}...', 'percent': 0})}\n\n"
                
                # Run hf download (newer huggingface_hub CLI)
                process = await asyncio.create_subprocess_exec(
                    "hf", "download", model_id,
                    "--local-dir", str(local_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
                
                # Stream output
                output_lines = []
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    line_text = line.decode().strip()
                    output_lines.append(line_text)
                    logger.debug(f"HF download: {line_text}")
                    
                    # Try to parse progress from HuggingFace output
                    # HF CLI shows progress like "Downloading: 45%|████..."
                    percent = 50  # Default to 50% during download
                    if "%" in line_text:
                        try:
                            # Extract percentage from output
                            import re
                            match = re.search(r'(\d+)%', line_text)
                            if match:
                                percent = int(match.group(1))
                        except:
                            pass
                    
                    yield f"data: {json.dumps({'status': 'downloading', 'message': line_text, 'percent': percent})}\n\n"
                
                await process.wait()
                logger.info(f"HF download process exited with code: {process.returncode}")
                
                if process.returncode == 0:
                    # Verify download - check for OpenVINO model files
                    has_openvino = (
                        (local_dir / "openvino_model.xml").exists() or 
                        (local_dir / "openvino_language_model.xml").exists()
                    )
                    if has_openvino:
                        logger.info(f"Vision model {model_id} downloaded and verified successfully")
                        yield f"data: {json.dumps({'status': 'success', 'message': f'Vision model {model_id} downloaded successfully', 'percent': 100})}\n\n"
                    else:
                        logger.warning(f"Vision model {model_id} downloaded but openvino_model.xml not found")
                        yield f"data: {json.dumps({'status': 'success', 'message': f'Model downloaded to {local_dir}. Note: May need conversion to OpenVINO format.', 'percent': 100})}\n\n"
                else:
                    error_msg = output_lines[-1] if output_lines else "Download failed"
                    logger.error(f"Vision model download failed: {error_msg}")
                    yield f"data: {json.dumps({'status': 'error', 'error': error_msg})}\n\n"
                    
            except FileNotFoundError:
                error_msg = 'hf CLI not found. Install with: pip install huggingface_hub'
                logger.error(f"Vision model download error: {error_msg}")
                yield f"data: {json.dumps({'status': 'error', 'error': error_msg})}\n\n"
            except Exception as e:
                logger.error(f"Vision model download exception: {e}", exc_info=True)
                yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_hf(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Text model: use Ollama
        # Check if already pulling
        existing_status = get_model_pull_status(model_id)
        if existing_status and existing_status.get("status") not in ("success", "error", None):
            raise HTTPException(status_code=409, detail=f"Model {model_id} is already being downloaded")
        
        async def generate():
            try:
                async for progress in pull_model(model_id):
                    # Format as SSE
                    data = json.dumps(progress)
                    yield f"data: {data}\n\n"
                    
                    if progress.get("status") in ("success", "error"):
                        break
            except Exception as e:
                error_data = json.dumps({"status": "error", "error": str(e)})
                yield f"data: {error_data}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )


@app.get("/api/models/{model_id:path}/pull/status")
async def get_pull_status(model_id: str) -> dict:
    """Get the current pull/download status for a model."""
    from llm import get_model_pull_status
    
    status = get_model_pull_status(model_id)
    if status is None:
        return {"status": "not_pulling", "model": model_id}
    return {"model": model_id, **status}


@app.delete("/api/models/{model_id:path}")
async def delete_model_endpoint(model_id: str) -> dict:
    """Delete a model (Ollama for text, filesystem for vision)."""
    from llm import delete_model, get_current_model
    from config import RECOMMENDED_MODELS
    import shutil
    
    # Determine if this is a vision model
    model_type = "text"
    for model_tuple in RECOMMENDED_MODELS:
        if model_tuple[0] == model_id:
            if len(model_tuple) >= 5:
                model_type = model_tuple[4]
            break
    
    config = AppConfig.load()
    
    if model_type == "vision":
        # Vision model: check if it's currently active
        model_dir_name = model_id.split("/")[-1]
        if config.vision_llm.enabled and model_dir_name.lower() in config.vision_llm.model_path.lower():
            raise HTTPException(
                status_code=400,
                detail="Cannot delete the currently active vision model. Switch to another model first."
            )
        
        # Delete the model directory
        model_path = Path("models") / model_dir_name
        if model_path.exists():
            try:
                shutil.rmtree(model_path)
                return {"status": "ok", "message": f"Vision model {model_id} deleted successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to delete vision model: {str(e)}")
        else:
            raise HTTPException(status_code=404, detail=f"Vision model {model_id} not found")
    else:
        # Text model: use Ollama
        current = get_current_model()
        if model_id == current:
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete the currently active model. Switch to another model first."
            )
        
        success = await delete_model(model_id)
        if success:
            return {"status": "ok", "message": f"Model {model_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete model {model_id}")


@app.post("/api/models/{model_id:path}/switch")
async def switch_model(model_id: str, request: Optional[ModelSwitchRequest] = None) -> dict:
    """
    Switch to a different LLM model.
    For text models: switches Ollama model.
    For vision models: enables vision LLM mode and sets model path.
    Optionally downloads the model if not installed.
    """
    from llm import (
        set_current_model, is_model_available, get_current_model,
        pull_model, get_model_pull_status
    )
    from config import RECOMMENDED_MODELS, VisionLLMConfig
    
    if request is None:
        request = ModelSwitchRequest()
    
    # Determine if this is a vision model
    model_type = "text"
    for model_tuple in RECOMMENDED_MODELS:
        if model_tuple[0] == model_id:
            if len(model_tuple) >= 5:
                model_type = model_tuple[4]
            break
    
    config = AppConfig.load()
    
    if model_type == "vision":
        # Vision model: update vision_llm config
        model_dir_name = model_id.split("/")[-1]  # e.g., "Qwen3-VL-2B-Instruct-int4"
        model_path = Path("models") / model_dir_name
        
        # Check if vision model is installed (check for either openvino file naming convention)
        is_installed = model_path.exists() and (
            (model_path / "openvino_model.xml").exists() or 
            (model_path / "openvino_language_model.xml").exists()
        )
        
        if not is_installed:
            if request.download_if_missing:
                return {
                    "status": "download_required",
                    "message": f"Vision model {model_id} is not installed. Use /api/models/{model_id}/pull to download it.",
                    "model": model_id,
                    "model_type": "vision"
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Vision model {model_id} is not installed."
                )
        
        # Enable vision LLM and set model path
        old_enabled = config.vision_llm.enabled
        old_path = config.vision_llm.model_path
        
        config.vision_llm = VisionLLMConfig(
            enabled=True,
            model_path=str(model_path),
            device=config.vision_llm.device
        )
        config.save()
        
        return {
            "status": "ok",
            "message": f"Switched to vision model {model_id}",
            "model": model_id,
            "model_type": "vision",
            "vision_llm_enabled": True,
            "model_path": str(model_path)
        }
    else:
        # Text model: check if installed in Ollama
        is_available = await is_model_available(model_id)
        
        if not is_available:
            if request.download_if_missing:
                return {
                    "status": "download_required",
                    "message": f"Model {model_id} is not installed. Use /api/models/{model_id}/pull to download it.",
                    "model": model_id,
                    "model_type": "text"
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {model_id} is not installed. Set download_if_missing=true to download."
                )
        
        # Switch to the text model and disable vision LLM
        old_model = get_current_model()
        await set_current_model(model_id)
        
        # Disable vision LLM when switching to a text model
        if config.vision_llm.enabled:
            config.vision_llm = VisionLLMConfig(
                enabled=False,
                model_path=config.vision_llm.model_path,
                device=config.vision_llm.device
            )
            config.save()
        
        return {
            "status": "ok",
            "message": f"Switched from {old_model} to {model_id}",
            "old_model": old_model,
            "new_model": model_id,
            "model_type": "text",
            "vision_llm_enabled": False
        }


# ============================================================================
# Server-Sent Events (SSE) for Real-time Updates
# ============================================================================

@app.get("/api/events")
async def subscribe_to_events():
    """
    Subscribe to real-time events via Server-Sent Events (SSE).
    
    Events include:
    - processing_started: A file started processing
    - processing_progress: Processing step update
    - processing_completed: A file finished processing
    - processing_error: A file failed processing
    - queue_updated: Queue status changed
    - stats_updated: Dashboard stats changed
    - document_approved/rejected: Document status changed
    - heartbeat: Keep-alive signal
    
    Client should reconnect if connection drops.
    """
    from fastapi.responses import StreamingResponse
    
    broadcaster = get_broadcaster()
    
    return StreamingResponse(
        broadcaster.subscribe(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/api/config/processing")
async def get_processing_config() -> ProcessingConfigResponse:
    """Get processing configuration."""
    config = AppConfig.load()
    return ProcessingConfigResponse(
        concurrent_workers=config.processing.concurrent_workers,
        batch_size=config.processing.batch_size
    )


@app.post("/api/config/processing")
async def save_processing_config(request: ProcessingConfigRequest) -> dict:
    """Save processing configuration."""
    # Validate values
    if request.concurrent_workers < 1:
        raise HTTPException(status_code=400, detail="concurrent_workers must be at least 1")
    if request.concurrent_workers > 10:
        raise HTTPException(status_code=400, detail="concurrent_workers cannot exceed 10")
    if request.batch_size < 1:
        raise HTTPException(status_code=400, detail="batch_size must be at least 1")
    if request.batch_size > 50:
        raise HTTPException(status_code=400, detail="batch_size cannot exceed 50")
    
    config = AppConfig.load()
    config.processing = ProcessingConfig(
        concurrent_workers=request.concurrent_workers,
        batch_size=request.batch_size
    )
    config.save()
    return {"status": "ok", "message": "Processing configuration saved successfully"}


@app.get("/api/processing/progress")
async def get_processing_progress() -> ProcessingProgressResponse:
    """Get current processing progress for the progress bar."""
    from llm import get_current_model
    from vision_llm import is_vision_llm_available
    
    config = AppConfig.load()
    tracker = get_processing_tracker()
    
    # Determine which model to show in progress bar
    # If Vision LLM is enabled and available, show the vision model name
    use_vision = config.vision_llm.enabled and is_vision_llm_available()
    if use_vision:
        # Extract model name from path (e.g., "models/Qwen2.5-VL-3B-Instruct-openvino-int4" -> "Qwen2.5-VL-3B")
        model_path = config.vision_llm.model_path
        model_name = model_path.split("/")[-1] if "/" in model_path else model_path
        # Simplify name for display
        if "Qwen" in model_name:
            # Extract Qwen version (e.g., "Qwen2.5-VL-3B-Instruct-openvino-int4" -> "Qwen2.5-VL-3B")
            parts = model_name.split("-")
            display_parts = []
            for part in parts:
                if part.lower() in ("instruct", "openvino", "int4", "int8", "fp16"):
                    break
                display_parts.append(part)
            llm_model = "-".join(display_parts) if display_parts else model_name
        else:
            llm_model = model_name
        llm_model = f"{llm_model} (Vision)"
    else:
        llm_model = get_current_model()
    
    if not config.incoming_share.path:
        return ProcessingProgressResponse(
            is_active=False,
            queued=0, processing=0, completed=0, error=0, total=0,
            current_files=[],
            processing_details=[],
            llm_model=llm_model
        )
    
    try:
        creds = config.incoming_share.credentials
        success, _, effective_path = get_effective_path(
            config.incoming_share.path,
            username=creds.username,
            password=creds.password,
            domain=creds.domain
        )
        
        if not success or effective_path is None:
            return ProcessingProgressResponse(
                is_active=False,
                queued=0, processing=0, completed=0, error=0, total=0,
                current_files=[],
                processing_details=[],
                llm_model=llm_model
            )
        
        # Get all files in incoming folder
        files = get_document_files(effective_path)
        
        # Get all queue items
        queue_items = await Database.get_all_queue_items()
        queue_by_path = {item['file_path']: item for item in queue_items}
        
        # Count by status
        counts = {'new': 0, 'queued': 0, 'processing': 0, 'completed': 0, 'error': 0}
        current_files = []
        
        for file_info in files:
            queue_item = queue_by_path.get(file_info.path)
            
            if not queue_item:
                counts['new'] += 1
            else:
                status = queue_item['status']
                if status == 'pending':
                    counts['queued'] += 1
                elif status == 'processing':
                    counts['processing'] += 1
                    current_files.append(file_info.name)
                elif status == 'completed':
                    counts['completed'] += 1
                elif status == 'error':
                    counts['error'] += 1
        
        is_active = counts['processing'] > 0 or counts['queued'] > 0
        
        # Get detailed processing status from tracker
        processing_statuses = await tracker.get_all_status()
        processing_details = [
            {
                "file_name": status.file_name,
                "step": status.step.value,
                "step_detail": status.step_detail,
                "progress_percent": status.progress_percent
            }
            for status in processing_statuses
        ]
        
        return ProcessingProgressResponse(
            is_active=is_active,
            queued=counts['new'] + counts['queued'],
            processing=counts['processing'],
            completed=counts['completed'],
            error=counts['error'],
            total=len(files),
            current_files=current_files,
            processing_details=processing_details,
            llm_model=llm_model
        )
        
    except Exception as e:
        logger.error(f"Error getting processing progress: {e}")
        return ProcessingProgressResponse(
            is_active=False,
            queued=0, processing=0, completed=0, error=0, total=0,
            current_files=[],
            processing_details=[],
            llm_model=llm_model
        )


@app.post("/api/processing/cancel")
async def cancel_processing(request: CancelProcessingRequest) -> CancelProcessingResponse:
    """
    Cancel processing for a specific file or all currently processing files.
    
    If file_path is provided, cancels only that file.
    If file_path is None, cancels all currently processing files.
    """
    tracker = get_processing_tracker()
    
    try:
        result = await tracker.cancel_processing(request.file_path)
        
        if result['count'] == 0:
            return CancelProcessingResponse(
                success=True,
                message="No files were being processed",
                cancelled_count=0,
                cancelled_files=[]
            )
        
        file_names = [Path(fp).name for fp in result['cancelled']]
        message = f"Cancelled processing for {result['count']} file(s): {', '.join(file_names)}"
        
        return CancelProcessingResponse(
            success=True,
            message=message,
            cancelled_count=result['count'],
            cancelled_files=result['cancelled']
        )
        
    except Exception as e:
        logger.error(f"Error cancelling processing: {e}")
        return CancelProcessingResponse(
            success=False,
            message=f"Failed to cancel processing: {str(e)}",
            cancelled_count=0,
            cancelled_files=[]
        )


@app.post("/api/test-connection")
async def test_connection(request: TestConnectionRequest) -> TestConnectionResponse:
    """Test connection to a path (local or network share)."""
    path = request.path
    creds = request.credentials
    
    try:
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
        
        if not effective_path.is_dir():
            return TestConnectionResponse(
                success=False,
                message=f"Path is not a directory: {effective_path}",
                writable=False
            )
        
        writable = os.access(str(effective_path), os.W_OK)
        
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


# ============================================================================
# Health & Stats API Routes
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "paperlinse"}


@app.get("/api/logs/{source}")
async def get_logs(
    source: str,
    lines: int = Query(500, ge=10, le=5000, description="Number of lines to return"),
    level: str = Query("all", description="Filter by log level: all, error, warning, info")
):
    """
    Get log file contents.
    
    Args:
        source: Log source - 'paperlinse' or 'ipex-ollama'
        lines: Number of lines to return (default 500, max 5000)
        level: Filter by log level
    """
    import re
    from collections import deque
    
    # Map source to log file
    log_files = {
        "paperlinse": Path(__file__).parent.parent / "logs" / "paperlinse.log",
        "ipex-ollama": Path(__file__).parent.parent / "logs" / "ipex-ollama.log"
    }
    
    if source not in log_files:
        raise HTTPException(status_code=400, detail=f"Invalid log source: {source}. Use 'paperlinse' or 'ipex-ollama'")
    
    log_file = log_files[source]
    
    if not log_file.exists():
        return {
            "source": source,
            "file": str(log_file),
            "exists": False,
            "lines": [],
            "total_lines": 0,
            "message": f"Log file not found: {log_file}"
        }
    
    try:
        # Read last N lines efficiently using deque
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = deque(f, maxlen=lines)
        
        log_entries = []
        
        # Parse log lines and apply level filter
        level_patterns = {
            "error": re.compile(r'\b(ERROR|CRITICAL|FATAL|Exception|Traceback)\b', re.IGNORECASE),
            "warning": re.compile(r'\b(WARNING|WARN|ERROR|CRITICAL|FATAL|Exception|Traceback)\b', re.IGNORECASE),
            "info": re.compile(r'\b(INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b', re.IGNORECASE),
        }
        
        for line in all_lines:
            line = line.rstrip('\n\r')
            if not line:
                continue
            
            # Determine log level from line content
            entry_level = "debug"
            if re.search(r'\bERROR\b|\bCRITICAL\b|\bFATAL\b', line, re.IGNORECASE):
                entry_level = "error"
            elif re.search(r'\bWARNING\b|\bWARN\b', line, re.IGNORECASE):
                entry_level = "warning"
            elif re.search(r'\bINFO\b', line, re.IGNORECASE):
                entry_level = "info"
            elif re.search(r'Exception|Traceback', line):
                entry_level = "error"
            
            # Apply filter
            if level != "all":
                pattern = level_patterns.get(level)
                if pattern and not pattern.search(line):
                    # Keep continuation lines (indented or starting with specific chars)
                    if not (line.startswith(' ') or line.startswith('\t') or line.startswith('|')):
                        continue
            
            log_entries.append({
                "text": line,
                "level": entry_level
            })
        
        return {
            "source": source,
            "file": str(log_file),
            "exists": True,
            "lines": log_entries,
            "total_lines": len(log_entries),
            "requested_lines": lines,
            "level_filter": level
        }
        
    except Exception as e:
        logger.error(f"Error reading log file {log_file}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")


@app.get("/api/stats")
async def get_stats() -> StatsResponse:
    """Get dashboard statistics based on actual files in incoming folder."""
    config = AppConfig.load()
    
    # Initialize counts
    counts = {
        'new': 0,           # Files not in queue
        'queued': 0,        # In queue, pending processing
        'processing': 0,    # Currently being processed
        'awaiting_approval': 0,
        'approved': 0,
        'rejected': 0,
        'error': 0,
    }
    
    if not config.incoming_share.path:
        return StatsResponse(
            pending=0, processing=0, awaiting_approval=0,
            approved=0, rejected=0, error=0, total=0
        )
    
    try:
        creds = config.incoming_share.credentials
        success, _, effective_path = get_effective_path(
            config.incoming_share.path,
            username=creds.username,
            password=creds.password,
            domain=creds.domain
        )
        
        if not success or effective_path is None:
            return StatsResponse(
                pending=0, processing=0, awaiting_approval=0,
                approved=0, rejected=0, error=0, total=0
            )
        
        # Get all files in incoming folder
        files = get_document_files(effective_path)
        
        # Get all queue items
        queue_items = await Database.get_all_queue_items()
        queue_by_path = {item['file_path']: item for item in queue_items}
        
        # Count statuses based on actual files
        for file_info in files:
            queue_item = queue_by_path.get(file_info.path)
            
            if not queue_item:
                counts['new'] += 1
            else:
                queue_status = queue_item['status']
                doc_id = queue_item.get('document_id')
                
                # If there's a linked document, use its status
                if doc_id:
                    doc = await Database.get_document(doc_id)
                    if doc:
                        doc_status = doc.status.value
                        if doc_status in counts:
                            counts[doc_status] += 1
                        else:
                            counts['processing'] += 1
                    else:
                        # Document deleted but queue item exists
                        counts[queue_status] += 1 if queue_status in counts else 0
                else:
                    # No document yet, use queue status
                    if queue_status == 'pending':
                        counts['queued'] += 1
                    elif queue_status == 'error':
                        counts['error'] += 1
                    elif queue_status == 'processing':
                        counts['processing'] += 1
                    elif queue_status in counts:
                        counts[queue_status] += 1
        
        total = sum(counts.values())
        
        return StatsResponse(
            pending=counts['new'] + counts['queued'],
            processing=counts['processing'],
            awaiting_approval=counts['awaiting_approval'],
            approved=counts['approved'],
            rejected=counts['rejected'],
            error=counts['error'],
            total=total
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return StatsResponse(
            pending=0, processing=0, awaiting_approval=0,
            approved=0, rejected=0, error=0, total=0
        )


# ============================================================================
# Incoming Files API Routes
# ============================================================================

@app.get("/api/incoming/files")
async def list_incoming_files() -> IncomingFilesResponse:
    """List all document files in the incoming share."""
    config = AppConfig.load()
    
    if not config.incoming_share.path:
        return IncomingFilesResponse(
            success=False,
            message="Incoming share not configured",
            files=[],
            count=0
        )
    
    try:
        creds = config.incoming_share.credentials
        success, message, effective_path = get_effective_path(
            config.incoming_share.path,
            username=creds.username,
            password=creds.password,
            domain=creds.domain
        )
        
        if not success or effective_path is None:
            return IncomingFilesResponse(
                success=False,
                message=message,
                files=[],
                count=0
            )
        
        files = get_document_files(effective_path)
        return IncomingFilesResponse(
            success=True,
            message=f"Found {len(files)} document(s) in incoming share",
            files=files,
            count=len(files)
        )
        
    except Exception as e:
        return IncomingFilesResponse(
            success=False,
            message=f"Error accessing incoming share: {str(e)}",
            files=[],
            count=0
        )


@app.get("/api/incoming/status")
async def list_incoming_files_with_status() -> IncomingFilesStatusResponse:
    """List all incoming files with their processing status."""
    config = AppConfig.load()
    
    if not config.incoming_share.path:
        return IncomingFilesStatusResponse(
            success=False,
            message="Incoming share not configured",
            files=[],
            count=0
        )
    
    try:
        creds = config.incoming_share.credentials
        success, message, effective_path = get_effective_path(
            config.incoming_share.path,
            username=creds.username,
            password=creds.password,
            domain=creds.domain
        )
        
        if not success or effective_path is None:
            return IncomingFilesStatusResponse(
                success=False,
                message=message,
                files=[],
                count=0
            )
        
        # Get all files in incoming folder
        files = get_document_files(effective_path)
        
        # Get all queue items
        queue_items = await Database.get_all_queue_items()
        queue_by_path = {item['file_path']: item for item in queue_items}
        
        # Build status list
        result_files = []
        for file_info in files:
            file_path = file_info.path
            queue_item = queue_by_path.get(file_path)
            
            status_info = IncomingFileStatus(
                name=file_info.name,
                path=file_info.path,
                size=file_info.size,
                modified=file_info.modified,
                thumbnail_url=f"/api/incoming/thumbnail?path={file_path}"
            )
            
            if queue_item:
                status_info.queue_status = queue_item['status']
                status_info.error_message = queue_item.get('error_message')
                
                # If there's a linked document, get its info
                doc_id = queue_item.get('document_id')
                if doc_id:
                    status_info.document_id = doc_id
                    doc = await Database.get_document(doc_id)
                    if doc:
                        status_info.document_status = doc.status.value
                        status_info.topic = doc.topic
                        status_info.sender = doc.sender
                        if doc.processing_error:
                            status_info.error_message = doc.processing_error
            else:
                status_info.queue_status = "new"
            
            result_files.append(status_info)
        
        # Sort by modified time (newest first)
        result_files.sort(key=lambda x: x.modified, reverse=True)
        
        return IncomingFilesStatusResponse(
            success=True,
            message=f"Found {len(result_files)} file(s)",
            files=result_files,
            count=len(result_files)
        )
        
    except Exception as e:
        logger.error(f"Error listing incoming files: {e}")
        return IncomingFilesStatusResponse(
            success=False,
            message=f"Error: {str(e)}",
            files=[],
            count=0
        )


@app.get("/api/incoming/thumbnail")
async def get_thumbnail(path: str = Query(..., description="File path")):
    """Generate and return a thumbnail for a document."""
    from fastapi.responses import Response
    from pathlib import Path as PathLib
    import io
    
    file_path = PathLib(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Use PIL to generate thumbnail
        from PIL import Image
        
        suffix = file_path.suffix.lower()
        
        if suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            # Direct image
            img = Image.open(file_path)
        elif suffix in ['.tif', '.tiff']:
            # TIFF - get first page
            img = Image.open(file_path)
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                img.seek(0)
        elif suffix == '.pdf':
            # PDF - need pdf2image
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(file_path, first_page=1, last_page=1, dpi=72)
                if images:
                    img = images[0]
                else:
                    raise HTTPException(status_code=500, detail="Could not convert PDF")
            except ImportError:
                # Fallback - return a placeholder
                raise HTTPException(status_code=501, detail="PDF thumbnails not supported")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Create thumbnail (max 200x200)
        img.thumbnail((200, 200), Image.Resampling.LANCZOS)
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=80)
        buffer.seek(0)
        
        return Response(
            content=buffer.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating thumbnail for {path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/incoming/preview")
async def get_document_preview(
    path: str = Query(..., description="File path"),
    page: int = Query(1, ge=1, description="Page number for multi-page documents")
):
    """Return full document for preview. Converts TIFF to JPEG for browser compatibility."""
    from fastapi.responses import FileResponse, Response
    from pathlib import Path as PathLib
    import io
    
    file_path = PathLib(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    suffix = file_path.suffix.lower()
    
    # For browser-compatible formats, serve directly
    if suffix in ['.jpg', '.jpeg', '.png', '.gif', '.pdf']:
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.pdf': 'application/pdf'
        }
        return FileResponse(
            path=file_path,
            media_type=media_types.get(suffix, 'application/octet-stream'),
            filename=file_path.name
        )
    
    # For TIFF and BMP, convert to JPEG for browser display
    if suffix in ['.tif', '.tiff', '.bmp']:
        try:
            from PIL import Image
            
            img = Image.open(file_path)
            
            # Handle multi-page TIFF
            if suffix in ['.tif', '.tiff'] and hasattr(img, 'n_frames') and img.n_frames > 1:
                # Seek to requested page (0-indexed internally)
                page_index = min(page - 1, img.n_frames - 1)
                img.seek(page_index)
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P', 'LA', 'L', 'I', 'F'):
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to buffer as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            
            # Include page info in headers for multi-page documents
            headers = {"Cache-Control": "public, max-age=300"}
            if suffix in ['.tif', '.tiff']:
                try:
                    img_check = Image.open(file_path)
                    if hasattr(img_check, 'n_frames'):
                        headers["X-Total-Pages"] = str(img_check.n_frames)
                        headers["X-Current-Page"] = str(page)
                except:
                    pass
            
            return Response(
                content=buffer.getvalue(),
                media_type="image/jpeg",
                headers=headers
            )
            
        except Exception as e:
            logger.error(f"Error converting {path} for preview: {e}")
            raise HTTPException(status_code=500, detail=f"Error converting file: {str(e)}")
    
    # Unsupported format
    raise HTTPException(status_code=400, detail=f"Preview not supported for {suffix} files")


@app.post("/api/incoming/reprocess")
async def reprocess_file(path: str = Query(..., description="File path")) -> dict:
    """
    Reprocess a file - deletes existing document data and re-queues for processing.
    """
    from pathlib import Path as PathLib
    
    file_path = PathLib(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Get existing queue item
        queue_item = await Database.get_queue_item(path)
        
        if queue_item:
            # If there's a linked document, delete it
            doc_id = queue_item.get('document_id')
            if doc_id:
                await Database.delete_document(doc_id)
                logger.info(f"Deleted document {doc_id} for reprocessing")
            
            # Remove from queue
            await Database.remove_queue_item(path)
        
        # Add back to queue with fresh state
        processor = get_processor()
        file_hash = processor.compute_file_hash(path)
        await Database.add_to_queue(path, file_hash)
        
        logger.info(f"File queued for reprocessing: {path}")
        
        return {
            "status": "ok",
            "message": "File queued for reprocessing"
        }
        
    except Exception as e:
        logger.error(f"Error reprocessing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Document API Routes
# ============================================================================

@app.get("/api/documents")
async def list_documents(
    status: Optional[str] = Query(None, description="Filter by status"),
    sender: Optional[str] = Query(None, description="Filter by sender"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
) -> DocumentListResponse:
    """List all processed documents with optional filters."""
    try:
        # Parse status
        doc_status = None
        if status:
            try:
                doc_status = DocumentStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        # Parse dates
        from_date = None
        to_date = None
        if date_from:
            try:
                from_date = datetime.strptime(date_from, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format")
        if date_to:
            try:
                to_date = datetime.strptime(date_to, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format")
        
        documents = await Database.list_documents(
            status=doc_status,
            sender=sender,
            date_from=from_date,
            date_to=to_date,
            limit=limit,
            offset=offset
        )
        
        return DocumentListResponse(
            documents=[document_to_response(d) for d in documents],
            total=len(documents)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/search")
async def search_documents(
    q: str = Query(..., description="Search query"),
    limit: int = Query(50, ge=1, le=200)
) -> DocumentListResponse:
    """Full-text search across documents."""
    try:
        documents = await Database.search_documents(q, limit)
        return DocumentListResponse(
            documents=[document_to_response(d) for d in documents],
            total=len(documents)
        )
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: int) -> DocumentResponse:
    """Get a document by ID with all its pages."""
    try:
        doc = await Database.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return document_to_response(doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_id}/related")
async def get_related_documents(doc_id: int) -> dict:
    """
    Find documents that share identifiers with the given document.
    
    This enables document linking - for example, finding all documents
    with the same Vorgangsnummer (transaction number) or Aktenzeichen (case number).
    """
    try:
        # Verify the document exists
        doc = await Database.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        related = await Database.get_related_documents(doc_id)
        
        return {
            "document_id": doc_id,
            "related_count": len(related),
            "related_documents": related
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding related documents for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/identifiers/search")
async def search_by_identifier(
    value: str = Query(..., description="Identifier value to search for"),
    identifier_type: Optional[str] = Query(None, description="Filter by identifier type"),
    limit: int = Query(50, ge=1, le=200)
) -> dict:
    """
    Search documents by identifier value.
    
    Supports partial matching. For example, searching for "2024" will find
    all documents with identifiers containing "2024".
    """
    try:
        results = await Database.search_by_identifier(value, identifier_type, limit)
        
        return {
            "query": value,
            "identifier_type": identifier_type,
            "count": len(results),
            "documents": results
        }
    except Exception as e:
        logger.error(f"Error searching by identifier: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/documents/{doc_id}")
async def update_document(doc_id: int, request: DocumentUpdateRequest) -> DocumentResponse:
    """Update document metadata."""
    try:
        doc = await Database.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Build update dict
        updates = {}
        if request.topic is not None:
            updates['topic'] = request.topic
        if request.summary is not None:
            updates['summary'] = request.summary
        if request.sender is not None:
            updates['sender'] = request.sender
        if request.receiver is not None:
            updates['receiver'] = request.receiver
        if request.document_type is not None:
            updates['document_type'] = request.document_type
        if request.document_date is not None:
            try:
                updates['document_date'] = datetime.strptime(request.document_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format")
        
        if updates:
            await Database.update_document(doc_id, **updates)
        
        # Return updated document
        doc = await Database.get_document(doc_id)
        return document_to_response(doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int) -> dict:
    """Delete a document."""
    try:
        success = await Database.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "ok", "message": f"Document {doc_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/{doc_id}/approve")
async def approve_document(doc_id: int) -> DocumentResponse:
    """Approve a document and move it to storage."""
    try:
        processor = get_processor()
        success = await processor.approve_document(doc_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Could not approve document")
        
        doc = await Database.get_document(doc_id)
        
        # Emit events
        await emit_document_approved(doc_id)
        await emit_stats_updated()
        await emit_queue_updated()
        
        return document_to_response(doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/{doc_id}/reject")
async def reject_document(doc_id: int, reason: str = "") -> DocumentResponse:
    """Reject a document."""
    try:
        processor = get_processor()
        success = await processor.reject_document(doc_id, reason)
        
        if not success:
            raise HTTPException(status_code=400, detail="Could not reject document")
        
        doc = await Database.get_document(doc_id)
        
        # Emit events
        await emit_document_rejected(doc_id)
        await emit_stats_updated()
        await emit_queue_updated()
        
        return document_to_response(doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: int,
    background_tasks: BackgroundTasks
) -> dict:
    """Queue a document for reprocessing."""
    try:
        doc = await Database.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Reset status to pending
        await Database.update_document(
            doc_id,
            status=DocumentStatus.PENDING,
            processing_error=""
        )
        
        return {"status": "ok", "message": f"Document {doc_id} queued for reprocessing"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Processing API Routes
# ============================================================================

@app.get("/api/processing/mode")
async def get_processing_mode() -> ProcessingModeResponse:
    """Get current processing mode (manual or automatic)."""
    try:
        mode = await Database.get_setting("processing_mode", "manual")
        return ProcessingModeResponse(mode=mode)
    except Exception as e:
        logger.error(f"Error getting processing mode: {e}")
        return ProcessingModeResponse(mode="manual")


@app.put("/api/processing/mode")
async def set_processing_mode(request: ProcessingModeRequest) -> ProcessingModeResponse:
    """Set processing mode (manual or automatic)."""
    if request.mode not in ["manual", "automatic"]:
        raise HTTPException(status_code=400, detail="Mode must be 'manual' or 'automatic'")
    
    try:
        await Database.set_setting("processing_mode", request.mode)
        return ProcessingModeResponse(mode=request.mode)
    except Exception as e:
        logger.error(f"Error setting processing mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/processing/process")
async def process_file(
    request: ProcessFileRequest,
    background_tasks: BackgroundTasks
) -> dict:
    """Manually trigger processing of a specific file."""
    try:
        processor = get_processor()
        
        # Add to processing in background
        background_tasks.add_task(
            processor.process_single_file,
            request.file_path,
            request.auto_approve
        )
        
        return {
            "status": "ok",
            "message": f"Processing started for {request.file_path}"
        }
        
    except Exception as e:
        logger.error(f"Error starting processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/processing/scan")
async def trigger_scan() -> dict:
    """Manually trigger a scan of the incoming folder."""
    try:
        processor = get_processor()
        
        # Get current queue count before scan
        queue_before = await Database.get_all_queue_items(statuses=['queued'])
        pending_before = len(queue_before)
        
        new_files = await processor.scan_incoming_folder()
        
        # Get queue count after scan
        queue_after = await Database.get_all_queue_items(statuses=['queued'])
        pending_after = len(queue_after)
        
        # Calculate actually new files added to queue
        newly_added = pending_after - pending_before
        
        # Determine the appropriate message
        if newly_added > 0:
            message = f"Found {newly_added} new file{'s' if newly_added != 1 else ''}"
        elif pending_after > 0:
            message = f"{pending_after} file{'s' if pending_after != 1 else ''} pending in queue"
        else:
            message = "No new files found"
        
        return {
            "status": "ok",
            "message": message,
            "files": new_files,
            "newly_added": newly_added,
            "total_pending": pending_after
        }
        
    except Exception as e:
        logger.error(f"Error scanning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/processing/process-queue")
async def process_queue(batch_size: int = 5) -> dict:
    """Manually trigger processing of queued items."""
    try:
        processor = get_processor()
        processed = await processor.process_queue(batch_size)
        
        return {
            "status": "ok",
            "message": f"Processed {processed} documents",
            "processed_count": processed
        }
        
    except Exception as e:
        logger.error(f"Error processing queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Static Files (Frontend)
# ============================================================================

FRONTEND_DIR = Path(__file__).parent / "frontend"

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
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce uvicorn access log noise - only show errors
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
