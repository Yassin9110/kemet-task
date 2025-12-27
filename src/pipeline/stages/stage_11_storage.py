"""
Stage 11: Storage

Persists all data: documents, parent chunks, child chunks,
vectors, and semantic edges.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from logging import Logger

from src.config.settings import PipelineConfig
from src.models.enums import DocumentFormat, IngestionStatus
from src.models.document import Document
from src.models.chunks import ParentChunk, ChildChunk
from src.models.edges import SemanticEdge
from src.storage.json_storage import StorageManager
from src.storage.vector_storage import create_vector_storage_from_config


@dataclass
class StorageInput:
    """Input for the storage stage."""
    document_id: str
    original_path: str
    stored_path: str
    file_hash: str
    format: DocumentFormat
    parent_chunks: List[ParentChunk]
    child_chunks: List[ChildChunk]
    semantic_edges: List[SemanticEdge]
    hierarchy_depth: int
    total_tokens: int
    language_distribution: Dict[str, float]
    upload_metadata: Dict
    processing_start_time: datetime


@dataclass
class StorageOutput:
    """Output from the storage stage."""
    document_id: str
    document: Document
    storage_stats: dict


class StorageError(Exception):
    """Raised when storage operations fail."""
    pass


class StorageStage:
    """
    Stage 11: Storage
    
    Responsibilities:
    - Create and persist Document record
    - Persist parent chunks to JSON
    - Persist child chunks to JSON (backup)
    - Store child chunk vectors in Chroma
    - Persist semantic edges to JSON
    - Update document status
    
    Storage Locations:
    - Raw files: Already stored in Stage 1
    - Document metadata: documents.json
    - Parent chunks: parents.json
    - Child chunks: children.json (backup) + Chroma (vectors)
    - Semantic edges: edges.json
    """
    
    STAGE_NAME = "Storage"
    STAGE_NUMBER = 11
    TOTAL_STAGES = 11
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the storage stage.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.storage = StorageManager.from_config(config)
        self.vector_storage = create_vector_storage_from_config(config)
    
    def execute(
        self,
        input_data: StorageInput,
        logger: Logger
    ) -> StorageOutput:
        """
        Execute the storage stage.
        
        Args:
            input_data: Storage input with all data to persist.
            logger: Logger instance for progress tracking.
            
        Returns:
            StorageOutput with document record and statistics.
            
        Raises:
            StorageError: If storage operations fail.
        """
        start_time = time.time()
        
        # Statistics
        stats = {
            'document_stored': False,
            'parents_stored': 0,
            'children_stored_json': 0,
            'children_stored_vector': 0,
            'edges_stored': 0,
            'vector_db_total_count': 0,
        }
        
        try:
            # Calculate processing duration
            processing_duration_ms = int(
                (datetime.now() - input_data.processing_start_time).total_seconds() * 1000
            )
            
            # Create Document record
            document = Document(
                document_id=input_data.document_id,
                file_path=input_data.original_path,
                stored_path=str(input_data.stored_path),
                file_hash=input_data.file_hash,
                format=input_data.format,
                total_tokens=input_data.total_tokens,
                language_distribution=input_data.language_distribution,
                hierarchy_depth=input_data.hierarchy_depth,
                upload_metadata=input_data.upload_metadata,
                ingestion_status=IngestionStatus.COMPLETED,
                processing_duration_ms=processing_duration_ms
            )

            # Store document metadata
            self._store_document(document)
            stats['document_stored'] = True
            
            # Store parent chunks
            if input_data.parent_chunks:
                self._store_parents(input_data.parent_chunks)
                stats['parents_stored'] = len(input_data.parent_chunks)
            
            # Store child chunks (JSON backup)
            if input_data.child_chunks:
                self._store_children_json(input_data.child_chunks)
                stats['children_stored_json'] = len(input_data.child_chunks)
            
            # Store child chunks (vector storage)
            if input_data.child_chunks:
                self._store_children_vectors(input_data.child_chunks)
                stats['children_stored_vector'] = len(input_data.child_chunks)
            
            # Store semantic edges
            if input_data.semantic_edges:
                self._store_edges(input_data.semantic_edges)
                stats['edges_stored'] = len(input_data.semantic_edges)
            
            # Get total vector count
            stats['vector_db_total_count'] = self.vector_storage.count()
            
        except Exception as e:
            # Update document status to failed if we created it
            logger.error(f"  → Storage failed: {str(e)}")
            raise StorageError(f"Failed to store document data: {str(e)}") from e
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log completion
        logger.info(
            f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
            f"{self.STAGE_NAME} ✓ ({duration_ms}ms)"
        )
        logger.info("  → Vectors stored in Chroma")
        logger.info("  → Metadata stored in JSON")
        
        return StorageOutput(
            document_id=input_data.document_id,
            document=document,
            storage_stats=stats
        )
    
    def _store_document(self, document: Document) -> None:
        """
        Store document metadata.
        
        Args:
            document: Document to store.
        """
        # Load existing documents
        try:
            existing_documents = self.storage.documents.load_all()
        except FileNotFoundError:
            existing_documents = []
        
        # Add new document
        existing_documents.append(document)
        
        # Save all documents
        self.storage.documents.save_many(existing_documents)
    
    def _store_parents(self, parents: List[ParentChunk]) -> None:
        """
        Store parent chunks to JSON.
        
        Args:
            parents: List of parent chunks to store.
        """
        # Load existing parents
        try:
            existing_parents = self.storage.parents.load_all()
        except FileNotFoundError:
            existing_parents = []
        
        # Add new parents
        existing_parents.extend(parents)
        
        # Save all parents
        self.storage.parents.save_many(existing_parents)
    
    def _store_children_json(self, children: List[ChildChunk]) -> None:
        """
        Store child chunks to JSON (backup).
        
        Args:
            children: List of child chunks to store.
        """
        # Load existing children
        try:
            existing_children = self.storage.children.load_all()
        except FileNotFoundError:
            existing_children = []
        
        # Add new children
        existing_children.extend(children)
        
        # Save all children
        self.storage.children.save_many(existing_children)
    
    def _store_children_vectors(self, children: List[ChildChunk]) -> None:
        """
        Store child chunks in vector storage.
        
        Args:
            children: List of child chunks to store.
        """
        # Filter chunks with embeddings
        chunks_with_embeddings = [
            chunk for chunk in children
            if chunk.embedding is not None
        ]

        # Add each chunk individually to vector storage
        for chunk in chunks_with_embeddings:
            self.vector_storage.add(chunk)
    
    def _store_edges(self, edges: List[SemanticEdge]) -> None:
        """
        Store semantic edges to JSON.
        
        Args:
            edges: List of semantic edges to store.
        """
        # Load existing edges
        try:
            existing_edges = self.storage.edges.load_all()
        except FileNotFoundError:
            existing_edges = []
        
        # Add new edges
        existing_edges.extend(edges)
        
        # Save all edges
        self.storage.edges.save_many(existing_edges)


class StorageCleanup:
    """
    Utility class for cleaning up storage on failed ingestion.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize storage cleanup.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.storage = StorageManager.from_config(config)
        self.vector_storage = create_vector_storage_from_config(config)
    
    def cleanup_document(self, document_id: str) -> bool:
        """
        Remove all data associated with a document.
        
        Args:
            document_id: Document ID to clean up.
            
        Returns:
            True if cleanup was successful.
        """
        try:
            # Remove from vector storage
            self.vector_storage.delete(document_id)
            
            # Remove from JSON storage
            self._remove_document_from_json(document_id)
            self._remove_parents_from_json(document_id)
            self._remove_children_from_json(document_id)
            self._remove_edges_from_json(document_id)
            
            return True
            
        except Exception:
            return False
    
    def _remove_document_from_json(self, document_id: str) -> None:
        """Remove document from documents.json."""
        try:
            documents = self.storage.documents.load_all()
            documents = [d for d in documents if d.document_id != document_id]
            self.storage.documents.save_many(documents)
        except FileNotFoundError:
            pass
    
    def _remove_parents_from_json(self, document_id: str) -> None:
        """Remove parents from parents.json."""
        try:
            parents = self.storage.parents.load_all
            parents = [p for p in parents if p.document_id != document_id]
            self.storage.parents.save_many(parents)
        except FileNotFoundError:
            pass
    
    def _remove_children_from_json(self, document_id: str) -> None:
        """Remove children from children.json."""
        try:
            children = self.storage.children.save_many()
            children = [c for c in children if c.document_id != document_id]
            self.storage.children.save_many(children)
        except FileNotFoundError:
            pass
    
    def _remove_edges_from_json(self, document_id: str) -> None:
        """
        Remove edges associated with document's chunks.
        
        Args:
            document_id: Document ID.
        """
        try:
            # Get chunk IDs for this document
            children = self.storage.children.load_all()
            chunk_ids = {c.chunk_id for c in children if c.document_id == document_id}
            
            # Remove edges involving these chunks
            edges = self.storage.edges.load_all()
            edges = [
                e for e in edges
                if e.source_chunk_id not in chunk_ids and e.target_chunk_id not in chunk_ids
            ]
            self.storage.edges.save_many(edges)
        except FileNotFoundError:
            pass