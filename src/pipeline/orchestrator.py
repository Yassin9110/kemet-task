"""
Pipeline Orchestrator

Main controller that coordinates all pipeline stages
and provides the primary API for document ingestion.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
from logging import Logger
import traceback

from src.config.settings import PipelineConfig
from src.models.document import Document
from src.models.chunks import ParentChunk, ChildChunk
from src.models.edges import SemanticEdge
from src.models.enums import IngestionStatus
from src.logging.pipeline_logger import create_logger_from_config

# Import all stages
from src.pipeline.stages.stage_01_upload import (
    UploadStage,
    UploadInput,
    UploadOutput,
    FileNotFoundError,
    FileTooLargeError,
)
from src.pipeline.stages.stage_02_validation import (
    ValidationStage,
    ValidationInput,
    ValidationOutput,
    UnsupportedFormatError,
)
from src.pipeline.stages.stage_03_format_detection import (
    FormatDetectionStage,
    FormatDetectionInput,
    FormatDetectionOutput,
)
from src.pipeline.stages.stage_04_extraction import (
    ExtractionStage,
    ExtractionInput,
    ExtractionOutput,
    ExtractionError,
    NoContentError,
)
from src.pipeline.stages.stage_05_normalization import (
    NormalizationStage,
    NormalizationInput,
    NormalizationOutput,
)
from src.pipeline.stages.stage_06_language_detection import (
    LanguageDetectionStage,
    LanguageDetectionInput,
    LanguageDetectionOutput,
)
from src.pipeline.stages.stage_07_structure_parsing import (
    StructureParsingStage,
    StructureParsingInput,
    StructureParsingOutput,
)
from src.pipeline.stages.stage_08_chunking import (
    ChunkingStage,
    ChunkingInput,
    ChunkingOutput,
)
from src.pipeline.stages.stage_09_embedding import (
    EmbeddingStage,
    EmbeddingInput,
    EmbeddingOutput,
)
from src.pipeline.stages.stage_10_graph_building import (
    GraphBuildingStage,
    GraphBuildingInput,
    GraphBuildingOutput,
)
from src.pipeline.stages.stage_11_storage import (
    StorageStage,
    StorageInput,
    StorageOutput,
    StorageCleanup,
)

# Import storage and retrieval
from src.storage.json_storage import StorageManager
from src.storage.vector_storage import VectorStorage, create_vector_storage_from_config
from src.storage.file_storage import FileStorage


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    document_id: str
    status: IngestionStatus
    document: Optional[Document]
    parent_count: int
    child_count: int
    edge_count: int
    total_tokens: int
    hierarchy_depth: int
    processing_duration_ms: int
    error_message: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from vector search."""
    chunk: ChildChunk
    score: float
    parent: Optional[ParentChunk]


@dataclass
class ExpandedContext:
    """Expanded context for a chunk."""
    chunk: ChildChunk
    parent: Optional[ParentChunk]
    prev_sibling: Optional[ChildChunk]
    next_sibling: Optional[ChildChunk]
    semantic_neighbors: List[ChildChunk]


class PipelineOrchestrator:
    """
    Main pipeline orchestrator.
    
    Coordinates all stages of document ingestion and provides
    the primary API for the RAG system.
    
    Usage:
        config = PipelineConfig(cohere_api_key="...")
        pipeline = PipelineOrchestrator(config)
        result = pipeline.ingest("/path/to/document.pdf")
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self.logger = create_logger_from_config(self.config)
        
        # Initialize stages
        self._init_stages()
        
        # Initialize storage
        self.json_storage = StorageManager.from_config(self.config)
        self.vector_storage = create_vector_storage_from_config(self.config)
        self.file_storage = FileStorage(self.config.raw_files_path)
        self.storage_cleanup = StorageCleanup(self.config)
    
    def _init_stages(self) -> None:
        """Initialize all pipeline stages."""
        self.stage_upload = UploadStage(self.config)
        self.stage_validation = ValidationStage(self.config)
        self.stage_format_detection = FormatDetectionStage(self.config)
        self.stage_extraction = ExtractionStage(self.config)
        self.stage_normalization = NormalizationStage(self.config)
        self.stage_language_detection = LanguageDetectionStage(self.config)
        self.stage_structure_parsing = StructureParsingStage(self.config)
        self.stage_chunking = ChunkingStage(self.config)
        self.stage_embedding = EmbeddingStage(self.config)
        self.stage_graph_building = GraphBuildingStage(self.config)
        self.stage_storage = StorageStage(self.config)
    
    def ingest(self, file_path: str, metadata: Optional[Dict] = None) -> IngestionResult:
        """
        Ingest a document through the complete pipeline.
        
        Args:
            file_path: Path to the document file.
            metadata: Optional metadata (user_id, source, tags).
            
        Returns:
            IngestionResult with status and statistics.
        """
        start_time = datetime.now()
        document_id = None
        
        # Log start
        self.logger.info("═" * 60)
        self.logger.info(f"Starting ingestion: {file_path}")
        
        try:
            # Stage 1: Upload
            upload_output = self.stage_upload.execute(UploadInput(file_path=file_path, metadata=metadata), self.logger)
            document_id = upload_output.document_id
            self.logger.info(f"Document ID: {document_id}")
            self.logger.info("═" * 60)
            # Stage 2: Validation
            validation_output = self.stage_validation.execute(
                ValidationInput(document_id=document_id, stored_file=upload_output.stored_file, file_extension=upload_output.file_extension),
                self.logger
            )
            
            # Stage 3: Format Detection
            format_output = self.stage_format_detection.execute(
                FormatDetectionInput(
                    document_id=document_id,
                    stored_path=validation_output.stored_path,
                    file_extension=validation_output.file_extension,
                    mime_type=validation_output.mime_type
                ),
                self.logger
            )
            
            # Stage 4: Extraction
            extraction_output = self.stage_extraction.execute(
                ExtractionInput(
                    document_id=document_id,
                    stored_path=format_output.stored_path,
                    format=format_output.format
                ),
                self.logger
            )
            
            # Stage 5: Normalization
            normalization_output = self.stage_normalization.execute(
                NormalizationInput(
                    document_id=document_id,
                    stored_path=extraction_output.stored_path,
                    format=extraction_output.format,
                    blocks=extraction_output.blocks
                ),
                self.logger
            )
            
            # Stage 6: Language Detection
            language_output = self.stage_language_detection.execute(
                LanguageDetectionInput(
                    document_id=document_id,
                    stored_path=normalization_output.stored_path,
                    format=normalization_output.format,
                    blocks=normalization_output.blocks
                ),
                self.logger
            )
            
            # Stage 7: Structure Parsing
            structure_output = self.stage_structure_parsing.execute(
                StructureParsingInput(
                    document_id=document_id,
                    stored_path=language_output.stored_path,
                    format=language_output.format,
                    blocks=language_output.blocks,
                    primary_language=language_output.primary_language
                ),
                self.logger
            )
            
            # Stage 8: Chunking
            chunking_output = self.stage_chunking.execute(
                ChunkingInput(
                    document_id=document_id,
                    stored_path=structure_output.stored_path,
                    format=structure_output.format,
                    blocks=structure_output.blocks,
                    primary_language=structure_output.primary_language
                ),
                self.logger
            )
            
            # Stage 9: Embedding
            embedding_output = self.stage_embedding.execute(
                EmbeddingInput(
                    document_id=document_id,
                    stored_path=chunking_output.stored_path,
                    format=chunking_output.format,
                    parent_chunks=chunking_output.parent_chunks,
                    child_chunks=chunking_output.child_chunks,
                    hierarchy_depth=chunking_output.hierarchy_depth,
                    total_tokens=chunking_output.total_tokens
                ),
                self.logger
            )
            
            # Stage 10: Graph Building
            graph_output = self.stage_graph_building.execute(
                GraphBuildingInput(
                    document_id=document_id,
                    stored_path=embedding_output.stored_path,
                    format=embedding_output.format,
                    parent_chunks=embedding_output.parent_chunks,
                    child_chunks=embedding_output.child_chunks,
                    hierarchy_depth=embedding_output.hierarchy_depth,
                    total_tokens=embedding_output.total_tokens
                ),
                self.logger
            )
            
            # Stage 11: Storage
            storage_output = self.stage_storage.execute(
                StorageInput(
                    document_id=document_id,
                    original_path=upload_output.original_path,
                    stored_path=graph_output.stored_path,
                    file_hash=validation_output.file_hash,
                    format=graph_output.format,
                    parent_chunks=graph_output.parent_chunks,
                    child_chunks=graph_output.child_chunks,
                    semantic_edges=graph_output.semantic_edges,
                    hierarchy_depth=graph_output.hierarchy_depth,
                    total_tokens=graph_output.total_tokens,
                    language_distribution=language_output.language_distribution,
                    upload_metadata=upload_output.metadata,
                    processing_start_time=start_time
                ),
                self.logger
            )
            
            # Calculate final duration
            end_time = datetime.now()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Log completion
            self.logger.info("═" * 60)
            self.logger.info("✓ Ingestion complete")
            self.logger.info(f"  Total time: {duration_ms / 1000:.1f}s")
            self.logger.info(f"  Document: {document_id}")
            self.logger.info(
                f"  Parents: {len(graph_output.parent_chunks)} | "
                f"Children: {len(graph_output.child_chunks)} | "
                f"Edges: {len(graph_output.semantic_edges)}"
            )
            self.logger.info("═" * 60)
            
            return IngestionResult(
                document_id=document_id,
                status=IngestionStatus.COMPLETED,
                document=storage_output.document,
                parent_count=len(graph_output.parent_chunks),
                child_count=len(graph_output.child_chunks),
                edge_count=len(graph_output.semantic_edges),
                total_tokens=graph_output.total_tokens,
                hierarchy_depth=graph_output.hierarchy_depth,
                processing_duration_ms=duration_ms
            )
            
        except (FileNotFoundError, FileTooLargeError, UnsupportedFormatError) as e:
            # Input validation errors
            self.logger.error(f"Validation error: {str(e)}")
            return self._create_failed_result(
                document_id=document_id,
                start_time=start_time,
                error_message=str(e)
            )
            
        except (ExtractionError, NoContentError) as e:
            # Extraction errors
            self.logger.error(f"Extraction error: {str(e)}")
            self._cleanup_on_failure(document_id)
            return self._create_failed_result(
                document_id=document_id,
                start_time=start_time,
                error_message=str(e)
            )
            
        except Exception as e:
            # Unexpected errors
            self.logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            self._cleanup_on_failure(document_id)
            return self._create_failed_result(
                document_id=document_id,
                start_time=start_time,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def _create_failed_result(
        self,
        document_id: Optional[str],
        start_time: datetime,
        error_message: str
    ) -> IngestionResult:
        """
        Create a failed ingestion result.
        
        Args:
            document_id: Document ID (if assigned).
            start_time: Processing start time.
            error_message: Error message.
            
        Returns:
            IngestionResult with failed status.
        """
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        self.logger.info("═" * 60)
        self.logger.error("✗ Ingestion failed")
        self.logger.error(f"  Error: {error_message}")
        self.logger.info("═" * 60)
        
        return IngestionResult(
            document_id=document_id or "unknown",
            status=IngestionStatus.FAILED,
            document=None,
            parent_count=0,
            child_count=0,
            edge_count=0,
            total_tokens=0,
            hierarchy_depth=0,
            processing_duration_ms=duration_ms,
            error_message=error_message
        )
    
    def _cleanup_on_failure(self, document_id: Optional[str]) -> None:
        """
        Clean up any stored data on failure.
        
        Args:
            document_id: Document ID to clean up.
        """
        if document_id:
            try:
                self.storage_cleanup.cleanup_document(document_id)
                self.logger.info(f"  → Cleaned up partial data for {document_id}")
            except Exception as e:
                self.logger.warning(f"  → Cleanup failed: {str(e)}")
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve document metadata by ID.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Document or None if not found.
        """
        try:
            documents = self.json_storage.documents.load_all()
            for doc in documents:
                if doc.document_id == document_id:
                    return doc
            return None
        except FileNotFoundError:
            return None
    
    def get_chunks(self, document_id: str) -> List[ChildChunk]:
        """
        Retrieve all child chunks for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            List of child chunks.
        """
        try:
            return self.json_storage.children.load_by_document(document_id=document_id)
        except FileNotFoundError:
            return []
    
    def get_parents(self, document_id: str) -> List[ParentChunk]:
        """
        Retrieve all parent chunks for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            List of parent chunks.
        """
        try:
            return self.json_storage.parents.load_by_document(document_id=document_id)
        except FileNotFoundError:
            return []
    
    def get_edges(self, document_id: str) -> List[SemanticEdge]:
        """
        Retrieve all semantic edges for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            List of semantic edges.
        """
        try:
            return self.json_storage.edges.load_by_document(document_id=document_id)
        except FileNotFoundError:
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all associated data.
        
        Args:
            document_id: Document ID to delete.
            
        Returns:
            True if deletion was successful.
        """
        self.logger.info(f"Deleting document: {document_id}")
        
        try:
            # Delete from file storage
            self.file_storage.delete(document_id)
            
            # Delete from all storage
            success = self.storage_cleanup.cleanup_document(document_id)
            
            if success:
                self.logger.info(f"  → Successfully deleted {document_id}")
            else:
                self.logger.warning(f"  → Partial deletion for {document_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"  → Deletion failed: {str(e)}")
            return False
    
    def list_documents(self) -> List[Document]:
        """
        List all ingested documents.
        
        Returns:
            List of all documents.
        """
        try:
            return self.json_storage.documents.load_all()
        except FileNotFoundError:
            return []
    
    def reprocess_all(self) -> List[IngestionResult]:
        """
        Reprocess all documents with current pipeline configuration.
        
        Returns:
            List of ingestion results.
        """
        self.logger.info("Starting reprocessing of all documents...")
        results: List[IngestionResult] = []
        
        try:
            documents = self.json_storage.documents.load_all()
        except FileNotFoundError:
            self.logger.info("No documents to reprocess")
            return results
        
        for doc in documents:
            self.logger.info(f"Reprocessing: {doc.document_id}")
            
            # Delete existing data
            self.delete_document(doc.document_id)
            
            # Re-ingest from original file
            result = self.ingest(
                file_path=doc.file_path,
                metadata=doc.upload_metadata
            )
            results.append(result)
        
        self.logger.info(f"Reprocessing complete. {len(results)} documents processed.")
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with statistics.
        """
        try:
            documents = self.json_storage.documents.load_all()
            parents = self.json_storage.parents.load_all()
            children = self.json_storage.children.load_all()
            edges = self.json_storage.edges.load_all()
            vector_count = self.vector_storage.count()
            
            return {
                'total_documents': len(documents),
                'total_parents': len(parents),
                'total_children': len(children),
                'total_edges': len(edges),
                'total_vectors': vector_count,
                'completed_documents': sum(
                    1 for d in documents
                    if d.ingestion_status == IngestionStatus.COMPLETED
                ),
                'failed_documents': sum(
                    1 for d in documents
                    if d.ingestion_status == IngestionStatus.FAILED
                ),
            }
        except FileNotFoundError:
            return {
                'total_documents': 0,
                'total_parents': 0,
                'total_children': 0,
                'total_edges': 0,
                'total_vectors': 0,
                'completed_documents': 0,
                'failed_documents': 0,
            }