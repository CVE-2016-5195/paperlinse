"""Server-Sent Events (SSE) system for real-time updates.

This module provides a centralized event broadcasting system that allows
the frontend to receive real-time updates about:
- Document processing progress
- Queue status changes
- Stats updates
- Model operations
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, AsyncGenerator, Set
from weakref import WeakSet

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events that can be broadcast."""
    # Processing events
    PROCESSING_STARTED = "processing_started"
    PROCESSING_PROGRESS = "processing_progress"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_ERROR = "processing_error"
    
    # Queue events
    QUEUE_UPDATED = "queue_updated"
    FILE_QUEUED = "file_queued"
    FILE_REMOVED = "file_removed"
    
    # Document events
    DOCUMENT_CREATED = "document_created"
    DOCUMENT_UPDATED = "document_updated"
    DOCUMENT_APPROVED = "document_approved"
    DOCUMENT_REJECTED = "document_rejected"
    
    # Stats events
    STATS_UPDATED = "stats_updated"
    
    # System events
    SCAN_STARTED = "scan_started"
    SCAN_COMPLETED = "scan_completed"
    
    # Heartbeat to keep connection alive
    HEARTBEAT = "heartbeat"


@dataclass
class Event:
    """An event to be broadcast to connected clients."""
    type: EventType
    data: dict
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        payload = {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp
        }
        return f"data: {json.dumps(payload)}\n\n"


class EventBroadcaster:
    """
    Singleton broadcaster that manages SSE connections and event distribution.
    
    Uses asyncio.Queue for each connected client to ensure non-blocking
    event delivery and proper backpressure handling.
    """
    
    _instance: Optional["EventBroadcaster"] = None
    
    def __init__(self):
        # Set of active client queues
        self._clients: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
    
    @classmethod
    def get_instance(cls) -> "EventBroadcaster":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def start(self):
        """Start the heartbeat task."""
        if self._running:
            return
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Event broadcaster started")
    
    async def stop(self):
        """Stop the heartbeat task."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info("Event broadcaster stopped")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to keep connections alive."""
        while self._running:
            try:
                await asyncio.sleep(15)  # Send heartbeat every 15 seconds
                if self._clients:
                    await self.broadcast(Event(
                        type=EventType.HEARTBEAT,
                        data={"clients": len(self._clients)}
                    ))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def subscribe(self) -> AsyncGenerator[str, None]:
        """
        Subscribe to events. Returns an async generator that yields SSE formatted events.
        
        Usage:
            async for event in broadcaster.subscribe():
                yield event
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        async with self._lock:
            self._clients.add(queue)
            client_count = len(self._clients)
        
        logger.debug(f"Client subscribed, total clients: {client_count}")
        
        try:
            # Send initial connection event
            yield Event(
                type=EventType.HEARTBEAT,
                data={"status": "connected", "clients": client_count}
            ).to_sse()
            
            while True:
                try:
                    # Wait for events with timeout to allow cleanup
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event.to_sse()
                except asyncio.TimeoutError:
                    # Send keepalive comment (not a full event)
                    yield ": keepalive\n\n"
                except asyncio.CancelledError:
                    break
        finally:
            async with self._lock:
                self._clients.discard(queue)
            logger.debug(f"Client unsubscribed, total clients: {len(self._clients)}")
    
    async def broadcast(self, event: Event):
        """Broadcast an event to all connected clients."""
        if not self._clients:
            return
        
        async with self._lock:
            # Copy to avoid modification during iteration
            clients = list(self._clients)
        
        for queue in clients:
            try:
                # Non-blocking put - drop event if queue is full
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Client queue full, dropping event")
    
    @property
    def client_count(self) -> int:
        """Get the number of connected clients."""
        return len(self._clients)


# Global broadcaster instance
def get_broadcaster() -> EventBroadcaster:
    """Get the global event broadcaster."""
    return EventBroadcaster.get_instance()


# Convenience functions for broadcasting common events

async def emit_processing_started(file_path: str, file_name: str):
    """Emit event when processing starts for a file."""
    await get_broadcaster().broadcast(Event(
        type=EventType.PROCESSING_STARTED,
        data={"file_path": file_path, "file_name": file_name}
    ))


async def emit_processing_progress(
    file_path: str,
    file_name: str,
    step: str,
    step_detail: str,
    progress_percent: int
):
    """Emit event for processing progress update."""
    await get_broadcaster().broadcast(Event(
        type=EventType.PROCESSING_PROGRESS,
        data={
            "file_path": file_path,
            "file_name": file_name,
            "step": step,
            "step_detail": step_detail,
            "progress_percent": progress_percent
        }
    ))


async def emit_processing_completed(file_path: str, file_name: str, document_id: int):
    """Emit event when processing completes successfully."""
    await get_broadcaster().broadcast(Event(
        type=EventType.PROCESSING_COMPLETED,
        data={
            "file_path": file_path,
            "file_name": file_name,
            "document_id": document_id
        }
    ))


async def emit_processing_error(file_path: str, file_name: str, error: str):
    """Emit event when processing fails."""
    await get_broadcaster().broadcast(Event(
        type=EventType.PROCESSING_ERROR,
        data={
            "file_path": file_path,
            "file_name": file_name,
            "error": error
        }
    ))


async def emit_queue_updated():
    """Emit event when queue status changes."""
    await get_broadcaster().broadcast(Event(
        type=EventType.QUEUE_UPDATED,
        data={}
    ))


async def emit_stats_updated():
    """Emit event when stats should be refreshed."""
    await get_broadcaster().broadcast(Event(
        type=EventType.STATS_UPDATED,
        data={}
    ))


async def emit_document_created(document_id: int):
    """Emit event when a document is created."""
    await get_broadcaster().broadcast(Event(
        type=EventType.DOCUMENT_CREATED,
        data={"document_id": document_id}
    ))


async def emit_document_updated(document_id: int, status: Optional[str] = None):
    """Emit event when a document is updated."""
    data: dict = {"document_id": document_id}
    if status:
        data["status"] = status
    await get_broadcaster().broadcast(Event(
        type=EventType.DOCUMENT_UPDATED,
        data=data
    ))


async def emit_document_approved(document_id: int):
    """Emit event when a document is approved."""
    await get_broadcaster().broadcast(Event(
        type=EventType.DOCUMENT_APPROVED,
        data={"document_id": document_id}
    ))


async def emit_document_rejected(document_id: int):
    """Emit event when a document is rejected."""
    await get_broadcaster().broadcast(Event(
        type=EventType.DOCUMENT_REJECTED,
        data={"document_id": document_id}
    ))


async def emit_scan_started():
    """Emit event when folder scan starts."""
    await get_broadcaster().broadcast(Event(
        type=EventType.SCAN_STARTED,
        data={}
    ))


async def emit_scan_completed(files_found: int):
    """Emit event when folder scan completes."""
    await get_broadcaster().broadcast(Event(
        type=EventType.SCAN_COMPLETED,
        data={"files_found": files_found}
    ))
