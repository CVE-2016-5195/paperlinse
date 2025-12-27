"""Document processing pipeline for Paperlinse."""

import os
import asyncio
import hashlib
import shutil
import logging
import mimetypes
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from config import AppConfig
from shares import get_effective_path
from database import Database, DocumentModel, PageModel, DocumentStatus
from ocr import PaddleOCREngine, OCRResult
from llm import OllamaClient, DocumentMetadata, get_ollama_client
from vision_llm import (
    extract_metadata_vision, is_vision_llm_available, get_vision_llm_status
)
from events import (
    emit_processing_started, emit_processing_progress,
    emit_processing_completed, emit_processing_error,
    emit_queue_updated, emit_stats_updated,
    emit_document_created, emit_scan_started, emit_scan_completed
)

logger = logging.getLogger(__name__)

# Processing configuration
SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}


class CancelledError(Exception):
    """Raised when processing is cancelled by user."""
    pass


class ProcessingStep(str, Enum):
    """Processing steps for tracking progress."""
    QUEUED = "queued"
    STARTING = "starting"
    OCR = "ocr"
    LLM = "llm"
    SAVING = "saving"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class FileProcessingStatus:
    """Status of a file being processed."""
    file_path: str
    file_name: str
    step: ProcessingStep
    step_detail: str = ""
    progress_percent: int = 0
    error_message: str = ""
    started_at: datetime = field(default_factory=datetime.now)


class ProcessingTracker:
    """Tracks the status of files being processed."""
    
    _instance: Optional["ProcessingTracker"] = None
    
    def __init__(self):
        self._processing: dict[str, FileProcessingStatus] = {}
        self._cancelled: set[str] = set()  # Track cancelled file paths
        self._lock = asyncio.Lock()
    
    @classmethod
    def get_instance(cls) -> "ProcessingTracker":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def start_processing(self, file_path: str) -> None:
        """Mark a file as starting processing."""
        file_name = Path(file_path).name
        async with self._lock:
            self._processing[file_path] = FileProcessingStatus(
                file_path=file_path,
                file_name=file_name,
                step=ProcessingStep.STARTING,
                step_detail="Initializing...",
                progress_percent=0
            )
        # Emit event
        await emit_processing_started(file_path, file_name)
    
    async def update_step(
        self,
        file_path: str,
        step: ProcessingStep,
        detail: str = "",
        progress: int = 0
    ) -> None:
        """Update the processing step for a file."""
        file_name = ""
        async with self._lock:
            if file_path in self._processing:
                self._processing[file_path].step = step
                self._processing[file_path].step_detail = detail
                self._processing[file_path].progress_percent = progress
                file_name = self._processing[file_path].file_name
        # Emit event
        if file_name:
            await emit_processing_progress(file_path, file_name, step.value, detail, progress)
    
    async def set_error(self, file_path: str, error: str) -> None:
        """Mark a file as having an error."""
        file_name = ""
        async with self._lock:
            if file_path in self._processing:
                self._processing[file_path].step = ProcessingStep.ERROR
                self._processing[file_path].error_message = error
                self._processing[file_path].step_detail = f"Error: {error[:100]}"
                file_name = self._processing[file_path].file_name
        # Emit event
        if file_name:
            await emit_processing_error(file_path, file_name, error)
    
    async def complete_processing(self, file_path: str) -> None:
        """Mark a file as completed and remove from tracking."""
        async with self._lock:
            if file_path in self._processing:
                del self._processing[file_path]
    
    async def get_status(self, file_path: str) -> Optional[FileProcessingStatus]:
        """Get status of a specific file."""
        async with self._lock:
            return self._processing.get(file_path)
    
    async def get_all_status(self) -> list[FileProcessingStatus]:
        """Get status of all files being processed."""
        async with self._lock:
            return list(self._processing.values())
    
    async def clear_all(self) -> None:
        """Clear all tracking (used on startup)."""
        async with self._lock:
            self._processing.clear()
            self._cancelled.clear()
    
    async def cancel_processing(self, file_path: Optional[str] = None) -> dict:
        """
        Cancel processing for a specific file or all files.
        
        Args:
            file_path: Specific file to cancel, or None to cancel all
        
        Returns:
            Dict with 'cancelled' list and 'count'
        """
        cancelled_files = []
        async with self._lock:
            if file_path:
                # Cancel specific file
                if file_path in self._processing:
                    self._cancelled.add(file_path)
                    self._processing[file_path].step_detail = "Cancelling... (will stop after current step)"
                    cancelled_files.append(file_path)
            else:
                # Cancel all currently processing files
                for fp in self._processing.keys():
                    self._cancelled.add(fp)
                    self._processing[fp].step_detail = "Cancelling... (will stop after current step)"
                    cancelled_files.append(fp)
        
        return {'cancelled': cancelled_files, 'count': len(cancelled_files)}
    
    def is_cancelled(self, file_path: str) -> bool:
        """
        Check if a file's processing has been cancelled.
        
        Note: This is not async to allow quick checks in processing loop.
        """
        return file_path in self._cancelled
    
    async def clear_cancelled(self, file_path: str) -> None:
        """Remove a file from the cancelled set after cleanup."""
        async with self._lock:
            self._cancelled.discard(file_path)


# Global tracker instance
def get_processing_tracker() -> ProcessingTracker:
    """Get the global processing tracker."""
    return ProcessingTracker.get_instance()


class DocumentProcessor:
    """Main document processing pipeline."""
    
    def __init__(self):
        self.ocr_engine = PaddleOCREngine.get_instance()
        self._llm_client: Optional[OllamaClient] = None
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
    
    async def get_llm_client(self) -> OllamaClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = await get_ollama_client()
        return self._llm_client
    
    def get_page_images(self, file_path: str, dpi: int = 150) -> list[str]:
        """
        Get image paths for each page of a document.
        
        For images, returns the original file path.
        For PDFs, converts each page to a temporary JPEG file.
        
        Args:
            file_path: Path to the document file
            dpi: Resolution for PDF to image conversion
        
        Returns:
            List of image file paths (may be temp files for PDFs)
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        # For image files, just return the original path
        if suffix in {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}:
            return [file_path]
        
        # For PDFs, convert each page to an image
        if suffix == '.pdf':
            try:
                import pdf2image
                
                images = pdf2image.convert_from_path(
                    file_path,
                    dpi=dpi,
                    fmt='jpeg'
                )
                
                image_paths = []
                for i, image in enumerate(images):
                    temp_path = f"/tmp/paperlinse_page_{i}_{path.stem}.jpg"
                    image.save(temp_path, "JPEG", quality=95)
                    image_paths.append(temp_path)
                
                logger.debug(f"Converted PDF to {len(image_paths)} page images")
                return image_paths
                
            except Exception as e:
                logger.error(f"Failed to convert PDF to images: {e}")
                return []
        
        return []
    
    def cleanup_temp_images(self, image_paths: list[str], original_path: str) -> None:
        """Clean up temporary image files (not the original)."""
        for img_path in image_paths:
            if img_path != original_path and img_path.startswith("/tmp/paperlinse_"):
                try:
                    os.unlink(img_path)
                except Exception:
                    pass
    
    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    async def process_single_file(
        self,
        file_path: str,
        auto_approve: bool = False
    ) -> Optional[int]:
        """
        Process a single document file.
        
        Args:
            file_path: Path to the document file
            auto_approve: If True, automatically approve and move to storage
        
        Returns:
            Document ID if successful, None if failed
        """
        path = Path(file_path)
        tracker = get_processing_tracker()
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {path.suffix}")
            return None
        
        logger.info(f"Processing file: {file_path}")
        
        # Start tracking
        await tracker.start_processing(file_path)
        
        doc_id = None
        try:
            # Step 1: Collect file metadata
            await tracker.update_step(file_path, ProcessingStep.STARTING, "Reading file metadata...", 5)
            
            file_stat = path.stat()
            file_size = file_stat.st_size
            file_modified = datetime.fromtimestamp(file_stat.st_mtime)
            file_hash = self.compute_file_hash(file_path)
            
            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "application/octet-stream"
            
            logger.info(f"File metadata: size={file_size}, modified={file_modified}, hash={file_hash[:12]}...")
            
            # Create initial document record with file metadata
            doc = DocumentModel(
                status=DocumentStatus.PROCESSING,
                source_path=file_path,
                file_size=file_size,
                file_modified=file_modified,
                file_hash=file_hash,
                mime_type=mime_type
            )
            doc_id = await Database.create_document(doc)
            
            # Update queue status
            await Database.update_queue_status(
                file_path, "processing", document_id=doc_id
            )
            
            # Step 2: Run OCR
            await tracker.update_step(file_path, ProcessingStep.OCR, "Running OCR text extraction...", 15)
            logger.info(f"Running OCR on {file_path}")
            ocr_results = self.ocr_engine.process_document(file_path)
            
            if not ocr_results:
                raise Exception("OCR returned no results")
            
            await tracker.update_step(file_path, ProcessingStep.OCR, f"OCR complete - extracted {len(ocr_results)} page(s)", 40)
            
            # Check for cancellation after OCR (heavy operation complete)
            if tracker.is_cancelled(file_path):
                raise CancelledError(f"Processing cancelled for {file_path}")
            
            # Combine all OCR text for metadata extraction
            combined_text = "\n\n".join(r.text for r in ocr_results if r.text)
            
            # Detect language from OCR results
            detected_lang = ocr_results[0].language if ocr_results else "de"
            if combined_text:
                detected_lang = self.ocr_engine.detect_language(combined_text)
            
            # Step 3: Extract metadata using LLM (text-based or vision-based)
            config = AppConfig.load()
            use_vision = config.vision_llm.enabled and is_vision_llm_available()
            
            # Check for cancellation before LLM (expensive operation)
            if tracker.is_cancelled(file_path):
                raise CancelledError(f"Processing cancelled for {file_path}")
            
            # Always get the text LLM client (needed for folder name generation)
            llm_client = await self.get_llm_client()
            
            page_images: list[str] = []
            metadata: Optional[DocumentMetadata] = None
            try:
                if use_vision:
                    # Vision-based extraction: send images to Qwen3-VL
                    await tracker.update_step(
                        file_path, ProcessingStep.LLM,
                        f"Analyzing document with Vision LLM ({detected_lang})...", 50
                    )
                    logger.info(f"Extracting metadata using Vision LLM for document {doc_id}")
                    
                    # Get page images for vision processing
                    page_images = self.get_page_images(file_path)
                    if not page_images:
                        logger.warning("No page images available, falling back to text-based extraction")
                        use_vision = False
                    else:
                        metadata = await extract_metadata_vision(page_images, detected_lang)
                
                if not use_vision:
                    # Text-based extraction: send OCR text to Ollama
                    await tracker.update_step(
                        file_path, ProcessingStep.LLM,
                        f"Analyzing document with text LLM ({detected_lang})...", 50
                    )
                    logger.info(f"Extracting metadata using text LLM for document {doc_id}")
                    metadata = await llm_client.extract_metadata(combined_text, detected_lang)
            finally:
                # Clean up temp page images
                if page_images:
                    self.cleanup_temp_images(page_images, file_path)
            
            # Check for cancellation after LLM (the heavy operation is done)
            if tracker.is_cancelled(file_path):
                raise CancelledError(f"Processing cancelled for {file_path}")
            
            # Ensure we have metadata
            if metadata is None:
                raise Exception("Failed to extract metadata from document")
            
            extraction_method = "Vision LLM" if use_vision else "Text LLM"
            await tracker.update_step(file_path, ProcessingStep.LLM, f"{extraction_method} analysis complete", 80)
            
            # If LLM didn't find a date, use the file modification date
            if metadata.document_date is None:
                metadata.document_date = file_modified
                logger.info(f"Using file modification date as document date: {file_modified}")
            
            # Step 4: Save to database
            await tracker.update_step(file_path, ProcessingStep.SAVING, "Saving to database...", 85)
            
            # Final cancellation check before committing to database
            if tracker.is_cancelled(file_path):
                raise CancelledError(f"Processing cancelled for {file_path}")
            
            # Save pages to database
            page_count = len(ocr_results)
            for i, ocr_result in enumerate(ocr_results):
                page = PageModel(
                    document_id=doc_id,
                    page_number=i + 1,
                    ocr_text=ocr_result.text,
                    confidence=ocr_result.confidence,
                    original_filename=path.name
                )
                await Database.add_page(page)
            
            # Generate folder name
            folder_name = await llm_client.generate_folder_name(metadata)
            
            # Determine final status
            if auto_approve:
                final_status = DocumentStatus.APPROVED
            else:
                final_status = DocumentStatus.AWAITING_APPROVAL
            
            # Prepare metadata_raw for storage
            metadata_raw = metadata.raw_response if metadata.raw_response else None
            
            # Update document with extracted metadata (including enhanced fields)
            await Database.update_document(
                doc_id,
                topic=metadata.topic,
                summary=metadata.summary,
                document_date=metadata.document_date,
                sender=metadata.sender,
                receiver=metadata.receiver,
                document_type=metadata.document_type,
                language=detected_lang,
                folder_path=folder_name,
                page_count=page_count,
                iban=metadata.iban,
                bic=metadata.bic,
                due_date=metadata.due_date,
                metadata_raw=json.dumps(metadata_raw) if metadata_raw else None,
                status=final_status,
                processed_at=datetime.now()
            )
            
            # Save identifiers for document linking
            if metadata.identifiers:
                identifiers_list = [
                    {"type": ident.type, "value": ident.value}
                    for ident in metadata.identifiers
                ]
                added_count = await Database.add_identifiers_batch(doc_id, identifiers_list)
                if added_count > 0:
                    logger.info(f"Added {added_count} identifiers to document {doc_id}")
            
            # If auto-approved, move to storage
            if auto_approve:
                await tracker.update_step(file_path, ProcessingStep.SAVING, "Moving to storage...", 90)
                await self.move_to_storage(doc_id)
            
            # Update queue status
            await Database.update_queue_status(
                file_path, "completed", document_id=doc_id
            )
            
            await tracker.update_step(file_path, ProcessingStep.COMPLETED, "Processing complete", 100)
            logger.info(f"Successfully processed document {doc_id}: {metadata.topic}")
            
            # Remove from tracker and emit events
            await tracker.complete_processing(file_path)
            await emit_processing_completed(file_path, path.name, doc_id)
            await emit_queue_updated()
            await emit_stats_updated()
            
            return doc_id
            
        except CancelledError as e:
            logger.info(f"Processing cancelled for {file_path}")
            
            # Update tracker to show cancelled status
            async with tracker._lock:
                if file_path in tracker._processing:
                    tracker._processing[file_path].step = ProcessingStep.CANCELLED
                    tracker._processing[file_path].step_detail = "Cancelled by user"
            
            # Clean up: delete partial document record if created
            try:
                if doc_id:
                    await Database.delete_document(doc_id)
                    logger.info(f"Deleted partial document record {doc_id}")
            except Exception:
                pass
            
            # Reset queue status to allow reprocessing later
            await Database.update_queue_status(
                file_path, "pending", error_message="Cancelled by user"
            )
            
            # Clear cancellation flag
            await tracker.clear_cancelled(file_path)
            
            # Emit events
            await emit_queue_updated()
            
            # Brief delay so UI can see cancelled state
            await asyncio.sleep(1)
            await tracker.complete_processing(file_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            
            # Track the error
            await tracker.set_error(file_path, str(e))
            
            # Update document status to error if we created one
            try:
                if doc_id:
                    await Database.update_document(
                        doc_id,
                        status=DocumentStatus.ERROR,
                        processing_error=str(e)
                    )
            except NameError:
                pass  # doc_id was never assigned
            
            # Update queue status
            await Database.update_queue_status(
                file_path, "error", error_message=str(e)
            )
            
            # Emit events for error state
            await emit_queue_updated()
            await emit_stats_updated()
            
            # Remove from tracker after a delay (so UI can see the error)
            await asyncio.sleep(2)
            await tracker.complete_processing(file_path)
            
            return None
    
    async def move_to_storage(self, doc_id: int) -> bool:
        """
        Move a processed document to the storage location.
        
        Args:
            doc_id: Document ID
        
        Returns:
            True if successful
        """
        try:
            doc = await Database.get_document(doc_id)
            if not doc:
                logger.error(f"Document {doc_id} not found")
                return False
            
            source_path = doc.source_path
            if not source_path:
                logger.error(f"Document {doc_id} has no source_path")
                return False
            
            if not Path(source_path).exists():
                logger.error(f"Source file no longer exists: {source_path}")
                return False
            
            config = AppConfig.load()
            if not config.storage.path:
                logger.error("Storage path not configured")
                return False
            
            # Get effective storage path (handles network shares)
            creds = config.storage.credentials
            success, message, storage_path = get_effective_path(
                config.storage.path,
                username=creds.username,
                password=creds.password,
                domain=creds.domain
            )
            
            if not success or storage_path is None:
                logger.error(f"Cannot access storage: {message}")
                return False
            
            # Create document folder
            folder_name = doc.folder_path or f"{datetime.now().strftime('%Y-%m-%d')}_Document-{doc_id}"
            doc_folder = storage_path / folder_name
            
            # Handle duplicate folder names
            counter = 1
            original_folder = doc_folder
            while doc_folder.exists():
                doc_folder = original_folder.with_name(f"{original_folder.name}_{counter}")
                counter += 1
            
            doc_folder.mkdir(parents=True, exist_ok=True)
            
            # Copy file to storage
            source = Path(source_path)
            dest = doc_folder / source.name
            shutil.copy2(source_path, dest)
            
            # Create metadata.json
            metadata = {
                "id": doc_id,
                "topic": doc.topic,
                "summary": doc.summary,
                "sender": doc.sender,
                "receiver": doc.receiver,
                "document_type": doc.document_type,
                "document_date": doc.document_date.isoformat() if doc.document_date else None,
                "language": doc.language,
                "file_size": doc.file_size,
                "file_hash": doc.file_hash,
                "mime_type": doc.mime_type,
                "page_count": doc.page_count,
                "file_modified": doc.file_modified.isoformat() if doc.file_modified else None,
                "processed_at": datetime.now().isoformat(),
                "identifiers": [
                    {"type": ident.identifier_type, "value": ident.identifier_value}
                    for ident in doc.identifiers
                ],
                "iban": doc.iban,
                "bic": doc.bic,
                "due_date": doc.due_date.isoformat() if doc.due_date else None,
                "pages": [
                    {
                        "page_number": p.page_number,
                        "ocr_text": p.ocr_text,
                        "confidence": p.confidence,
                        "original_filename": p.original_filename
                    }
                    for p in doc.pages
                ]
            }
            
            metadata_path = doc_folder / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
            
            # Update document with final storage path
            await Database.update_document(
                doc_id,
                folder_path=str(doc_folder),
                status=DocumentStatus.APPROVED
            )
            
            # Remove original file from incoming folder
            try:
                source.unlink()
                logger.info(f"Removed original file: {source_path}")
            except Exception as e:
                logger.warning(f"Could not remove original file: {e}")
            
            logger.info(f"Document {doc_id} moved to storage: {doc_folder}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving document {doc_id} to storage: {e}")
            return False
    
    async def approve_document(self, doc_id: int) -> bool:
        """
        Approve a document and move it to storage.
        
        Args:
            doc_id: Document ID to approve
        
        Returns:
            True if successful
        """
        doc = await Database.get_document(doc_id)
        if not doc:
            logger.error(f"Document {doc_id} not found")
            return False
        
        if doc.status != DocumentStatus.AWAITING_APPROVAL:
            logger.warning(f"Document {doc_id} is not awaiting approval (status: {doc.status})")
            return False
        
        # Move to storage
        success = await self.move_to_storage(doc_id)
        if not success:
            logger.error(f"Failed to move document {doc_id} to storage")
            return False
        
        logger.info(f"Document {doc_id} approved and moved to storage")
        return True
    
    async def reject_document(self, doc_id: int, reason: str = "") -> bool:
        """
        Reject a document.
        
        Args:
            doc_id: Document ID to reject
            reason: Optional rejection reason
        
        Returns:
            True if successful
        """
        doc = await Database.get_document(doc_id)
        if not doc:
            return False
        
        await Database.update_document(
            doc_id,
            status=DocumentStatus.REJECTED,
            processing_error=f"Rejected: {reason}" if reason else "Rejected by user"
        )
        logger.info(f"Document {doc_id} rejected: {reason}")
        return True
    
    async def scan_incoming_folder(self) -> list[str]:
        """
        Scan the incoming folder for new documents.
        
        Returns:
            List of new file paths found
        """
        config = AppConfig.load()
        if not config.incoming_share.path:
            return []
        
        creds = config.incoming_share.credentials
        success, message, incoming_path = get_effective_path(
            config.incoming_share.path,
            username=creds.username,
            password=creds.password,
            domain=creds.domain
        )
        
        if not success or incoming_path is None:
            logger.error(f"Cannot access incoming folder: {message}")
            return []
        
        new_files = []
        
        for item in incoming_path.iterdir():
            if not item.is_file():
                continue
            
            if item.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            
            file_path = str(item)
            
            # Check if already in queue
            existing = await Database.get_queue_item(file_path)
            if existing and existing['status'] in ['processing', 'completed']:
                continue
            
            # Add to queue if new
            if not existing:
                file_hash = self.compute_file_hash(file_path)
                await Database.add_to_queue(file_path, file_hash)
            
            new_files.append(file_path)
        
        return new_files
    
    async def process_queue(self, batch_size: Optional[int] = None, max_concurrent: Optional[int] = None) -> int:
        """
        Process pending items in the queue with concurrent workers.
        
        Args:
            batch_size: Number of items to process in this batch (from config if None)
            max_concurrent: Max concurrent processing tasks (from config if None)
        
        Returns:
            Number of items processed
        """
        # Load config for defaults
        config = AppConfig.load()
        if batch_size is None:
            batch_size = config.processing.batch_size
        if max_concurrent is None:
            max_concurrent = config.processing.concurrent_workers
        
        # Get processing mode
        mode = await Database.get_setting("processing_mode", "manual")
        auto_approve = (mode == "automatic")
        
        # Get pending items
        pending = await Database.get_pending_queue_items(batch_size)
        
        if not pending:
            return 0
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                file_path = item['file_path']
                
                # Check if file still exists
                if not Path(file_path).exists():
                    await Database.update_queue_status(
                        file_path, "error", error_message="File no longer exists"
                    )
                    return None
                
                return await self.process_single_file(file_path, auto_approve)
        
        # Process all items concurrently (limited by semaphore)
        tasks = [process_with_semaphore(item) for item in pending]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful processing
        processed = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        
        return processed
    
    async def cleanup_missing_files(self) -> dict:
        """
        Remove queue items and documents for files that no longer exist.
        Also resets stuck 'processing' items back to 'queued'.
        
        This should be called on startup to clean up stale entries
        for files that have been manually deleted from the incoming folder.
        
        Also removes documents with empty source_path that can't be processed.
        
        Returns:
            Dict with 'queue_removed', 'documents_removed', 'queue_reset', 'documents_reset' counts
        """
        logger.info("Checking for missing files and stuck processing items...")
        
        result = {'queue_removed': 0, 'documents_removed': 0, 'queue_reset': 0, 'documents_reset': 0}
        
        # 0. First reset stuck 'processing' items back to 'queued'
        reset_result = await Database.reset_stuck_processing()
        result['queue_reset'] = reset_result.get('queue_reset', 0)
        result['documents_reset'] = reset_result.get('documents_reset', 0)
        
        if result['queue_reset'] > 0 or result['documents_reset'] > 0:
            logger.info(f"Reset {result['queue_reset']} stuck queue items and {result['documents_reset']} stuck documents")
        
        # 1. Clean up processing queue for missing files
        queue_items = await Database.get_all_queue_items(
            statuses=['queued', 'error']
        )
        
        if queue_items:
            missing_queue_files = []
            for item in queue_items:
                file_path = item['file_path']
                if not Path(file_path).exists():
                    missing_queue_files.append(file_path)
                    logger.debug(f"Queue file no longer exists: {file_path}")
            
            if missing_queue_files:
                result['queue_removed'] = await Database.remove_queue_items_batch(missing_queue_files)
                logger.info(f"Removed {result['queue_removed']} stale queue items")
        
        # 2. Clean up documents awaiting approval whose source files are gone or empty
        awaiting_docs = await Database.list_documents(
            status=DocumentStatus.AWAITING_APPROVAL,
            limit=1000
        )
        
        if awaiting_docs:
            invalid_doc_ids = []
            for doc in awaiting_docs:
                # Remove if source_path is empty (can't be moved to storage)
                if not doc.source_path:
                    invalid_doc_ids.append(doc.id)
                    logger.debug(f"Document {doc.id} has no source_path")
                # Remove if source file no longer exists
                elif not Path(doc.source_path).exists():
                    invalid_doc_ids.append(doc.id)
                    logger.debug(f"Document {doc.id} source file missing: {doc.source_path}")
            
            if invalid_doc_ids:
                removed = await Database.delete_documents_batch(invalid_doc_ids)
                result['documents_removed'] += removed
                logger.info(f"Removed {removed} awaiting approval documents (missing/empty source)")
        
        # 3. Also clean up documents still in PENDING status with missing files
        pending_docs = await Database.list_documents(
            status=DocumentStatus.PENDING,
            limit=1000
        )
        
        if pending_docs:
            stale_pending_ids = []
            for doc in pending_docs:
                # Remove if source_path is empty or file doesn't exist
                if not doc.source_path or not Path(doc.source_path).exists():
                    stale_pending_ids.append(doc.id)
                    logger.debug(f"Stale pending document {doc.id}")
            
            if stale_pending_ids:
                removed = await Database.delete_documents_batch(stale_pending_ids)
                result['documents_removed'] += removed
                logger.info(f"Removed {removed} stale pending documents")
        
        total_actions = sum(result.values())
        if total_actions == 0:
            logger.info("All queued files and pending documents are valid")
        
        return result
    
    async def start_background_processing(self, interval: int = 30):
        """
        Start background processing loop.
        
        Args:
            interval: Seconds between processing cycles
        """
        if self._running:
            logger.warning("Background processing already running")
            return
        
        self._running = True
        logger.info(f"Starting background processing (interval: {interval}s)")
        
        while self._running:
            try:
                # Scan for new files
                new_files = await self.scan_incoming_folder()
                if new_files:
                    logger.info(f"Found {len(new_files)} new files")
                
                # Process queue
                processed = await self.process_queue()
                if processed:
                    logger.info(f"Processed {processed} documents")
                
            except Exception as e:
                logger.error(f"Error in background processing: {e}")
            
            await asyncio.sleep(interval)
    
    def stop_background_processing(self):
        """Stop background processing."""
        self._running = False
        if self._process_task:
            self._process_task.cancel()
        logger.info("Background processing stopped")


# Global processor instance
_processor: Optional[DocumentProcessor] = None


def get_processor() -> DocumentProcessor:
    """Get the global document processor instance."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor


async def process_document(file_path: str, auto_approve: bool = False) -> Optional[int]:
    """Convenience function to process a single document."""
    processor = get_processor()
    return await processor.process_single_file(file_path, auto_approve)
