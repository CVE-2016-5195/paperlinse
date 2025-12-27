"""Database models and connection management for Paperlinse."""

import os
import json
from datetime import datetime
from enum import Enum
from typing import Optional
from contextlib import asynccontextmanager

import asyncpg
from pydantic import BaseModel, Field


# Database configuration from environment
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://paperlinse:paperlinse@localhost:5432/paperlinse"
)


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"           # In incoming folder, not yet processed
    PROCESSING = "processing"     # Currently being processed
    AWAITING_APPROVAL = "awaiting_approval"  # Processed, waiting for manual approval
    APPROVED = "approved"         # Approved and moved to storage
    REJECTED = "rejected"         # Rejected by user
    ERROR = "error"               # Processing failed


class PageModel(BaseModel):
    """Model for a document page."""
    id: Optional[int] = None
    document_id: Optional[int] = None
    page_number: int
    ocr_text: str = ""
    confidence: float = 0.0
    image_path: str = ""
    original_filename: str = ""
    created_at: Optional[datetime] = None


class DocumentIdentifier(BaseModel):
    """Model for a document identifier (reference number, invoice number, etc.)."""
    id: Optional[int] = None
    document_id: Optional[int] = None
    identifier_type: str = ""  # e.g., "Vorgangsnummer", "Rechnungsnummer", "Aktenzeichen"
    identifier_value: str = ""
    created_at: Optional[datetime] = None


class DocumentModel(BaseModel):
    """Model for a processed document."""
    id: Optional[int] = None
    topic: str = ""
    summary: str = ""
    document_date: Optional[datetime] = None
    sender: str = ""
    receiver: str = ""
    document_type: str = ""  # invoice, letter, contract, etc.
    language: str = ""       # de, en, etc.
    folder_path: str = ""    # Path in storage
    source_path: str = ""    # Original file path (for moving after approval)
    file_size: int = 0       # Original file size in bytes
    file_modified: Optional[datetime] = None  # Original file modification time
    file_hash: str = ""      # SHA-256 hash for duplicate detection
    mime_type: str = ""      # File MIME type
    page_count: int = 0      # Number of pages
    # Enhanced metadata from LLM
    identifiers: list[DocumentIdentifier] = Field(default_factory=list)  # Reference numbers, etc.
    iban: str = ""           # Bank account IBAN if detected
    bic: str = ""            # Bank BIC/SWIFT if detected
    due_date: Optional[datetime] = None  # Due date for invoices
    metadata_raw: Optional[dict] = None  # Complete raw LLM response for debugging
    status: DocumentStatus = DocumentStatus.PENDING
    processing_error: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    pages: list[PageModel] = Field(default_factory=list)


class Database:
    """Database connection and operations manager."""
    
    _pool: Optional[asyncpg.Pool] = None
    
    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        """Get or create database connection pool."""
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
        return cls._pool
    
    @classmethod
    async def close(cls) -> None:
        """Close database connection pool."""
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None
    
    @classmethod
    @asynccontextmanager
    async def connection(cls):
        """Get a database connection from the pool."""
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            yield conn
    
    @classmethod
    async def init_schema(cls) -> None:
        """Initialize database schema."""
        async with cls.connection() as conn:
            # Create enum type for document status
            await conn.execute("""
                DO $$ BEGIN
                    CREATE TYPE document_status AS ENUM (
                        'pending', 'processing', 'awaiting_approval',
                        'approved', 'rejected', 'error'
                    );
                EXCEPTION
                    WHEN duplicate_object THEN null;
                END $$;
            """)
            
            # Create documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    topic VARCHAR(500) DEFAULT '',
                    summary TEXT DEFAULT '',
                    document_date TIMESTAMP,
                    sender VARCHAR(500) DEFAULT '',
                    receiver VARCHAR(500) DEFAULT '',
                    document_type VARCHAR(100) DEFAULT '',
                    language VARCHAR(10) DEFAULT '',
                    folder_path VARCHAR(1000) DEFAULT '',
                    source_path VARCHAR(1000) DEFAULT '',
                    file_size BIGINT DEFAULT 0,
                    file_modified TIMESTAMP,
                    status document_status DEFAULT 'pending',
                    processing_error TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    processed_at TIMESTAMP
                );
            """)
            
            # Add new columns if they don't exist (for existing databases)
            await conn.execute("""
                DO $$ BEGIN
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_path VARCHAR(1000) DEFAULT '';
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_size BIGINT DEFAULT 0;
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_modified TIMESTAMP;
                    -- Enhanced metadata columns
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64) DEFAULT '';
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS mime_type VARCHAR(100) DEFAULT '';
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_count INTEGER DEFAULT 0;
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS iban VARCHAR(50) DEFAULT '';
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS bic VARCHAR(20) DEFAULT '';
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS due_date TIMESTAMP;
                    ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata_raw JSONB;
                EXCEPTION
                    WHEN duplicate_column THEN null;
                END $$;
            """)
            
            # Create document_identifiers table for reference numbers, invoice numbers, etc.
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_identifiers (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    identifier_type VARCHAR(100) NOT NULL,
                    identifier_value VARCHAR(500) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create indexes for identifier lookups (for document linking)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_identifiers_value 
                ON document_identifiers(identifier_value);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_identifiers_type_value 
                ON document_identifiers(identifier_type, identifier_value);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_identifiers_document_id 
                ON document_identifiers(document_id);
            """)
            
            # Create pages table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS pages (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL,
                    ocr_text TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    image_path VARCHAR(1000) DEFAULT '',
                    original_filename VARCHAR(500) DEFAULT '',
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create indexes for full-text search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pages_ocr_text_search 
                ON pages USING GIN (to_tsvector('german', ocr_text));
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_topic_search 
                ON documents USING GIN (to_tsvector('german', topic));
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_sender_search 
                ON documents USING GIN (to_tsvector('german', sender));
            """)
            
            # Create indexes for common queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_status 
                ON documents(status);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_date 
                ON documents(document_date);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_sender 
                ON documents(sender);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pages_document_id 
                ON pages(document_id);
            """)
            
            # Create processing queue table for tracking incoming files
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_queue (
                    id SERIAL PRIMARY KEY,
                    file_path VARCHAR(1000) UNIQUE NOT NULL,
                    file_hash VARCHAR(64),
                    status VARCHAR(50) DEFAULT 'queued',
                    document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
                    error_message TEXT DEFAULT '',
                    attempts INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_processing_queue_status 
                ON processing_queue(status);
            """)
            
            # Create settings table for processing mode etc.
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key VARCHAR(100) PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Insert default settings
            await conn.execute("""
                INSERT INTO settings (key, value) 
                VALUES ('processing_mode', 'manual')
                ON CONFLICT (key) DO NOTHING;
            """)
    
    # Document operations
    @classmethod
    async def create_document(cls, doc: DocumentModel) -> int:
        """Create a new document and return its ID."""
        async with cls.connection() as conn:
            row = await conn.fetchrow("""
                INSERT INTO documents (
                    topic, summary, document_date, sender, receiver,
                    document_type, language, folder_path, source_path,
                    file_size, file_modified, file_hash, mime_type, page_count,
                    iban, bic, due_date, metadata_raw, status, processing_error
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                RETURNING id
            """, doc.topic, doc.summary, doc.document_date, doc.sender,
                doc.receiver, doc.document_type, doc.language, doc.folder_path,
                doc.source_path, doc.file_size, doc.file_modified,
                doc.file_hash, doc.mime_type, doc.page_count,
                doc.iban, doc.bic, doc.due_date,
                json.dumps(doc.metadata_raw) if doc.metadata_raw else None,
                doc.status.value, doc.processing_error)
            return row['id']
    
    @classmethod
    async def update_document(cls, doc_id: int, **kwargs) -> None:
        """Update document fields."""
        if not kwargs:
            return
        
        # Handle status enum conversion
        if 'status' in kwargs and isinstance(kwargs['status'], DocumentStatus):
            kwargs['status'] = kwargs['status'].value
        
        set_clauses = []
        values = []
        for i, (key, value) in enumerate(kwargs.items(), 1):
            set_clauses.append(f"{key} = ${i}")
            values.append(value)
        
        values.append(doc_id)
        query = f"""
            UPDATE documents 
            SET {', '.join(set_clauses)}, updated_at = NOW()
            WHERE id = ${len(values)}
        """
        
        async with cls.connection() as conn:
            await conn.execute(query, *values)
    
    @classmethod
    async def get_document(cls, doc_id: int) -> Optional[DocumentModel]:
        """Get a document by ID with its pages and identifiers."""
        async with cls.connection() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM documents WHERE id = $1
            """, doc_id)
            
            if not row:
                return None
            
            pages = await conn.fetch("""
                SELECT * FROM pages WHERE document_id = $1 ORDER BY page_number
            """, doc_id)
            
            identifiers = await conn.fetch("""
                SELECT * FROM document_identifiers WHERE document_id = $1
            """, doc_id)
            
            # Parse metadata_raw from JSON if present
            metadata_raw = None
            if row.get('metadata_raw'):
                try:
                    metadata_raw = json.loads(row['metadata_raw']) if isinstance(row['metadata_raw'], str) else row['metadata_raw']
                except (json.JSONDecodeError, TypeError):
                    pass
            
            doc = DocumentModel(
                id=row['id'],
                topic=row['topic'],
                summary=row['summary'],
                document_date=row['document_date'],
                sender=row['sender'],
                receiver=row['receiver'],
                document_type=row['document_type'],
                language=row['language'],
                folder_path=row['folder_path'],
                source_path=row.get('source_path', ''),
                file_size=row.get('file_size', 0),
                file_modified=row.get('file_modified'),
                file_hash=row.get('file_hash', ''),
                mime_type=row.get('mime_type', ''),
                page_count=row.get('page_count', 0),
                iban=row.get('iban', ''),
                bic=row.get('bic', ''),
                due_date=row.get('due_date'),
                metadata_raw=metadata_raw,
                identifiers=[DocumentIdentifier(
                    id=i['id'],
                    document_id=i['document_id'],
                    identifier_type=i['identifier_type'],
                    identifier_value=i['identifier_value'],
                    created_at=i['created_at']
                ) for i in identifiers],
                status=DocumentStatus(row['status']),
                processing_error=row['processing_error'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                processed_at=row['processed_at'],
                pages=[PageModel(
                    id=p['id'],
                    document_id=p['document_id'],
                    page_number=p['page_number'],
                    ocr_text=p['ocr_text'],
                    confidence=p['confidence'],
                    image_path=p['image_path'],
                    original_filename=p['original_filename'],
                    created_at=p['created_at']
                ) for p in pages]
            )
            return doc
    
    @classmethod
    async def list_documents(
        cls,
        status: Optional[DocumentStatus] = None,
        sender: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> list[DocumentModel]:
        """List documents with optional filters."""
        conditions = []
        values = []
        param_count = 0
        
        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            values.append(status.value)
        
        if sender:
            param_count += 1
            conditions.append(f"sender ILIKE ${param_count}")
            values.append(f"%{sender}%")
        
        if date_from:
            param_count += 1
            conditions.append(f"document_date >= ${param_count}")
            values.append(date_from)
        
        if date_to:
            param_count += 1
            conditions.append(f"document_date <= ${param_count}")
            values.append(date_to)
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        param_count += 1
        limit_param = param_count
        param_count += 1
        offset_param = param_count
        values.extend([limit, offset])
        
        query = f"""
            SELECT * FROM documents
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${limit_param} OFFSET ${offset_param}
        """
        
        async with cls.connection() as conn:
            rows = await conn.fetch(query, *values)
            
            documents = []
            for row in rows:
                documents.append(DocumentModel(
                    id=row['id'],
                    topic=row['topic'],
                    summary=row['summary'],
                    document_date=row['document_date'],
                    sender=row['sender'],
                    receiver=row['receiver'],
                    document_type=row['document_type'],
                    language=row['language'],
                    folder_path=row['folder_path'],
                    source_path=row.get('source_path', ''),
                    file_size=row.get('file_size', 0),
                    file_modified=row.get('file_modified'),
                    status=DocumentStatus(row['status']),
                    processing_error=row['processing_error'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    processed_at=row['processed_at']
                ))
            return documents
    
    @classmethod
    async def search_documents(cls, query: str, limit: int = 50) -> list[DocumentModel]:
        """Full-text search across documents and pages."""
        async with cls.connection() as conn:
            # Search in both document fields and page OCR text
            rows = await conn.fetch("""
                SELECT DISTINCT d.* FROM documents d
                LEFT JOIN pages p ON d.id = p.document_id
                WHERE 
                    to_tsvector('german', d.topic) @@ plainto_tsquery('german', $1)
                    OR to_tsvector('german', d.sender) @@ plainto_tsquery('german', $1)
                    OR to_tsvector('german', d.summary) @@ plainto_tsquery('german', $1)
                    OR to_tsvector('german', p.ocr_text) @@ plainto_tsquery('german', $1)
                    OR d.topic ILIKE $2
                    OR d.sender ILIKE $2
                    OR p.ocr_text ILIKE $2
                ORDER BY d.created_at DESC
                LIMIT $3
            """, query, f"%{query}%", limit)
            
            documents = []
            for row in rows:
                documents.append(DocumentModel(
                    id=row['id'],
                    topic=row['topic'],
                    summary=row['summary'],
                    document_date=row['document_date'],
                    sender=row['sender'],
                    receiver=row['receiver'],
                    document_type=row['document_type'],
                    language=row['language'],
                    folder_path=row['folder_path'],
                    source_path=row.get('source_path', ''),
                    file_size=row.get('file_size', 0),
                    file_modified=row.get('file_modified'),
                    status=DocumentStatus(row['status']),
                    processing_error=row['processing_error'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    processed_at=row['processed_at']
                ))
            return documents
    
    @classmethod
    async def delete_document(cls, doc_id: int) -> bool:
        """Delete a document and its pages."""
        async with cls.connection() as conn:
            result = await conn.execute("""
                DELETE FROM documents WHERE id = $1
            """, doc_id)
            return result == "DELETE 1"
    
    @classmethod
    async def delete_documents_batch(cls, doc_ids: list[int]) -> int:
        """Delete multiple documents and their pages."""
        if not doc_ids:
            return 0
        async with cls.connection() as conn:
            result = await conn.execute("""
                DELETE FROM documents WHERE id = ANY($1)
            """, doc_ids)
            # Result is like "DELETE 5"
            try:
                return int(result.split()[-1])
            except (IndexError, ValueError):
                return 0
    
    @classmethod
    async def get_documents_with_source_path(
        cls, 
        status: Optional[DocumentStatus] = None
    ) -> list[dict]:
        """Get documents with their source paths, optionally filtered by status."""
        async with cls.connection() as conn:
            if status:
                rows = await conn.fetch("""
                    SELECT id, source_path, status FROM documents 
                    WHERE status = $1 AND source_path IS NOT NULL AND source_path != ''
                """, status.value)
            else:
                rows = await conn.fetch("""
                    SELECT id, source_path, status FROM documents 
                    WHERE source_path IS NOT NULL AND source_path != ''
                """)
            return [dict(row) for row in rows]
    
    # Page operations
    @classmethod
    async def add_page(cls, page: PageModel) -> int:
        """Add a page to a document."""
        async with cls.connection() as conn:
            row = await conn.fetchrow("""
                INSERT INTO pages (
                    document_id, page_number, ocr_text, confidence,
                    image_path, original_filename
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """, page.document_id, page.page_number, page.ocr_text,
                page.confidence, page.image_path, page.original_filename)
            return row['id']
    
    # Document identifier operations
    @classmethod
    async def add_identifier(cls, identifier: DocumentIdentifier) -> int:
        """Add an identifier to a document."""
        async with cls.connection() as conn:
            row = await conn.fetchrow("""
                INSERT INTO document_identifiers (
                    document_id, identifier_type, identifier_value
                )
                VALUES ($1, $2, $3)
                RETURNING id
            """, identifier.document_id, identifier.identifier_type, 
                identifier.identifier_value)
            return row['id']
    
    @classmethod
    async def add_identifiers_batch(
        cls, 
        document_id: int, 
        identifiers: list[dict]
    ) -> int:
        """
        Add multiple identifiers to a document.
        
        Args:
            document_id: Document ID
            identifiers: List of dicts with 'type' and 'value' keys
        
        Returns:
            Number of identifiers added
        """
        if not identifiers:
            return 0
        
        async with cls.connection() as conn:
            count = 0
            for ident in identifiers:
                ident_type = ident.get('type', '').strip()[:100]  # Truncate to DB column size
                ident_value = ident.get('value', '').strip()[:500]  # Truncate to DB column size
                if ident_type and ident_value:
                    await conn.execute("""
                        INSERT INTO document_identifiers (
                            document_id, identifier_type, identifier_value
                        )
                        VALUES ($1, $2, $3)
                    """, document_id, ident_type, ident_value)
                    count += 1
            return count
    
    @classmethod
    async def get_related_documents(cls, doc_id: int) -> list[dict]:
        """
        Find documents that share identifiers with the given document.
        
        Returns:
            List of dicts with document info and matching identifiers
        """
        async with cls.connection() as conn:
            # Get identifiers for this document
            doc_identifiers = await conn.fetch("""
                SELECT identifier_type, identifier_value 
                FROM document_identifiers 
                WHERE document_id = $1
            """, doc_id)
            
            if not doc_identifiers:
                return []
            
            # Find other documents with matching identifiers
            # Using a CTE to get all matching identifiers and their documents
            rows = await conn.fetch("""
                WITH doc_ids AS (
                    SELECT identifier_type, identifier_value 
                    FROM document_identifiers 
                    WHERE document_id = $1
                )
                SELECT DISTINCT 
                    d.id, d.topic, d.sender, d.document_date, d.document_type, d.status,
                    di.identifier_type, di.identifier_value
                FROM documents d
                JOIN document_identifiers di ON d.id = di.document_id
                WHERE (di.identifier_type, di.identifier_value) IN (
                    SELECT identifier_type, identifier_value FROM doc_ids
                )
                AND d.id != $1
                ORDER BY d.document_date DESC
            """, doc_id)
            
            # Group by document
            docs_map = {}
            for row in rows:
                doc_id_row = row['id']
                if doc_id_row not in docs_map:
                    docs_map[doc_id_row] = {
                        'id': doc_id_row,
                        'topic': row['topic'],
                        'sender': row['sender'],
                        'document_date': row['document_date'].isoformat() if row['document_date'] else None,
                        'document_type': row['document_type'],
                        'status': row['status'],
                        'matching_identifiers': []
                    }
                docs_map[doc_id_row]['matching_identifiers'].append({
                    'type': row['identifier_type'],
                    'value': row['identifier_value']
                })
            
            return list(docs_map.values())
    
    @classmethod
    async def search_by_identifier(
        cls, 
        value: str, 
        identifier_type: Optional[str] = None,
        limit: int = 50
    ) -> list[dict]:
        """
        Search documents by identifier value.
        
        Args:
            value: Identifier value to search for (partial match)
            identifier_type: Optional type filter
            limit: Maximum results
        
        Returns:
            List of documents with matching identifiers
        """
        async with cls.connection() as conn:
            if identifier_type:
                rows = await conn.fetch("""
                    SELECT DISTINCT 
                        d.id, d.topic, d.sender, d.document_date, d.document_type, d.status,
                        di.identifier_type, di.identifier_value
                    FROM documents d
                    JOIN document_identifiers di ON d.id = di.document_id
                    WHERE di.identifier_value ILIKE $1
                    AND di.identifier_type = $2
                    ORDER BY d.document_date DESC
                    LIMIT $3
                """, f"%{value}%", identifier_type, limit)
            else:
                rows = await conn.fetch("""
                    SELECT DISTINCT 
                        d.id, d.topic, d.sender, d.document_date, d.document_type, d.status,
                        di.identifier_type, di.identifier_value
                    FROM documents d
                    JOIN document_identifiers di ON d.id = di.document_id
                    WHERE di.identifier_value ILIKE $1
                    ORDER BY d.document_date DESC
                    LIMIT $2
                """, f"%{value}%", limit)
            
            # Group by document
            docs_map = {}
            for row in rows:
                doc_id = row['id']
                if doc_id not in docs_map:
                    docs_map[doc_id] = {
                        'id': doc_id,
                        'topic': row['topic'],
                        'sender': row['sender'],
                        'document_date': row['document_date'].isoformat() if row['document_date'] else None,
                        'document_type': row['document_type'],
                        'status': row['status'],
                        'matching_identifiers': []
                    }
                docs_map[doc_id]['matching_identifiers'].append({
                    'type': row['identifier_type'],
                    'value': row['identifier_value']
                })
            
            return list(docs_map.values())
    
    # Processing queue operations
    @classmethod
    async def add_to_queue(cls, file_path: str, file_hash: str = "") -> int:
        """Add a file to the processing queue."""
        async with cls.connection() as conn:
            row = await conn.fetchrow("""
                INSERT INTO processing_queue (file_path, file_hash)
                VALUES ($1, $2)
                ON CONFLICT (file_path) DO UPDATE SET updated_at = NOW()
                RETURNING id
            """, file_path, file_hash)
            return row['id']
    
    @classmethod
    async def get_queue_item(cls, file_path: str) -> Optional[dict]:
        """Get a queue item by file path."""
        async with cls.connection() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM processing_queue WHERE file_path = $1
            """, file_path)
            return dict(row) if row else None
    
    @classmethod
    async def update_queue_status(
        cls, file_path: str, status: str,
        document_id: Optional[int] = None,
        error_message: str = ""
    ) -> None:
        """Update processing queue item status."""
        async with cls.connection() as conn:
            await conn.execute("""
                UPDATE processing_queue 
                SET status = $2, document_id = $3, error_message = $4,
                    attempts = attempts + 1, updated_at = NOW()
                WHERE file_path = $1
            """, file_path, status, document_id, error_message)
    
    @classmethod
    async def get_pending_queue_items(cls, limit: int = 10) -> list[dict]:
        """Get pending items from processing queue."""
        async with cls.connection() as conn:
            rows = await conn.fetch("""
                SELECT * FROM processing_queue 
                WHERE status = 'queued' AND attempts < 3
                ORDER BY created_at ASC
                LIMIT $1
            """, limit)
            return [dict(row) for row in rows]
    
    @classmethod
    async def get_all_queue_items(cls, statuses: Optional[list[str]] = None) -> list[dict]:
        """Get all items from processing queue, optionally filtered by status."""
        async with cls.connection() as conn:
            if statuses:
                rows = await conn.fetch("""
                    SELECT * FROM processing_queue 
                    WHERE status = ANY($1)
                    ORDER BY created_at ASC
                """, statuses)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM processing_queue 
                    ORDER BY created_at ASC
                """)
            return [dict(row) for row in rows]
    
    @classmethod
    async def remove_queue_item(cls, file_path: str) -> bool:
        """Remove an item from the processing queue."""
        async with cls.connection() as conn:
            result = await conn.execute("""
                DELETE FROM processing_queue WHERE file_path = $1
            """, file_path)
            return result == "DELETE 1"
    
    @classmethod
    async def remove_queue_items_batch(cls, file_paths: list[str]) -> int:
        """Remove multiple items from the processing queue."""
        if not file_paths:
            return 0
        async with cls.connection() as conn:
            result = await conn.execute("""
                DELETE FROM processing_queue WHERE file_path = ANY($1)
            """, file_paths)
            # Result is like "DELETE 5"
            try:
                return int(result.split()[-1])
            except (IndexError, ValueError):
                return 0
    
    @classmethod
    async def reset_stuck_processing(cls) -> dict:
        """
        Reset stuck 'processing' items back to 'queued' status.
        Also resets stuck documents in PROCESSING status.
        
        Returns:
            Dict with 'queue_reset' and 'documents_reset' counts
        """
        result = {'queue_reset': 0, 'documents_reset': 0}
        
        async with cls.connection() as conn:
            # Reset stuck queue items
            queue_result = await conn.execute("""
                UPDATE processing_queue 
                SET status = 'queued', updated_at = NOW()
                WHERE status = 'processing'
            """)
            try:
                result['queue_reset'] = int(queue_result.split()[-1])
            except (IndexError, ValueError):
                pass
            
            # Reset stuck documents
            doc_result = await conn.execute("""
                UPDATE documents 
                SET status = 'pending', updated_at = NOW()
                WHERE status = 'processing'
            """)
            try:
                result['documents_reset'] = int(doc_result.split()[-1])
            except (IndexError, ValueError):
                pass
        
        return result
    
    # Settings operations
    @classmethod
    async def get_setting(cls, key: str, default: str = "") -> str:
        """Get a setting value."""
        async with cls.connection() as conn:
            row = await conn.fetchrow("""
                SELECT value FROM settings WHERE key = $1
            """, key)
            return row['value'] if row else default
    
    @classmethod
    async def set_setting(cls, key: str, value: str) -> None:
        """Set a setting value."""
        async with cls.connection() as conn:
            await conn.execute("""
                INSERT INTO settings (key, value, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()
            """, key, value)
    
    # Statistics
    @classmethod
    async def get_document_counts(cls) -> dict:
        """Get document counts by status."""
        async with cls.connection() as conn:
            rows = await conn.fetch("""
                SELECT status, COUNT(*) as count FROM documents GROUP BY status
            """)
            counts = {status.value: 0 for status in DocumentStatus}
            for row in rows:
                counts[row['status']] = row['count']
            return counts
