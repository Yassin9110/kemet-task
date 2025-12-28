"""
Document dataclasses for the Multilingual RAG Ingestion Pipeline.

This module defines document-level data structures:
- Document: Complete document metadata and state
- IngestionResult: Result of pipeline processing
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any

from .enums import DocumentFormat, IngestionStatus


@dataclass
class Document:
    """
    Complete metadata for an ingested document.
    
    Represents a document throughout its lifecycle in the pipeline,
    from initial upload through completed ingestion.
    """
    
    document_id: str  # Unique identifier for this document (UUID).
    
    file_path: str # Original file path provided at upload.
    
    stored_path: str # Path to the immutable copy in storage.
    
    file_hash: str # SHA256 hash of the file contents.
    
    format: DocumentFormat # Detected document format.
    
    ingestion_status: IngestionStatus = IngestionStatus.PENDING # Current processing state.
    
    total_tokens: int = 0 # Total token count of the document.
    
    hierarchy_depth: int = 1 # Hierarchy depth (1, 2, or 3 levels).
    
    language_distribution: Dict[str, float] = field(default_factory=dict) # Language distribution, e.g., {'ar': 0.6, 'en': 0.4}.
    
    parent_count: int = 0  # Number of parent chunks created.
    
    child_count: int = 0 # Number of child chunks created.
    
    edge_count: int = 0 # Number of semantic edges created.
    
    error_message: Optional[str] = None # Error details if ingestion failed.
    
    warnings: List[str] = field(default_factory=list) # Non-fatal warnings encountered during processing.
    
    upload_metadata: Dict[str, Any] = field(default_factory=dict) # User-provided metadata (user_id, source, tags, etc.).
    
    created_at: datetime = field(default_factory=datetime.now) # Timestamp when document was uploaded.
    
    completed_at: Optional[datetime] = None # Timestamp when ingestion completed (success or failure).

    processing_duration_ms: Optional[int] = None # Total processing time in milliseconds.
    
    def __post_init__(self) -> None:
        """Validate document data after initialization."""
        errors = []
        
        if not self.document_id:
            errors.append("document_id cannot be empty")
        
        if not self.file_path:
            errors.append("file_path cannot be empty")
        
        if not self.stored_path:
            errors.append("stored_path cannot be empty")
        
        if not self.file_hash:
            errors.append("file_hash cannot be empty")
        
        if self.total_tokens < 0:
            errors.append(f"total_tokens cannot be negative, got {self.total_tokens}")
        
        if self.hierarchy_depth not in (1, 2, 3):
            errors.append(f"hierarchy_depth must be 1, 2, or 3, got {self.hierarchy_depth}")
        
        if self.parent_count < 0:
            errors.append(f"parent_count cannot be negative, got {self.parent_count}")
        
        if self.child_count < 0:
            errors.append(f"child_count cannot be negative, got {self.child_count}")
        
        if self.edge_count < 0:
            errors.append(f"edge_count cannot be negative, got {self.edge_count}")
        
        if errors:
            raise ValueError("Document validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @property
    def is_pending(self) -> bool:
        """Check if document is pending processing."""
        return self.ingestion_status == IngestionStatus.PENDING
    
    @property
    def is_processing(self) -> bool:
        """Check if document is currently being processed."""
        return self.ingestion_status == IngestionStatus.PROCESSING
    
    @property
    def is_completed(self) -> bool:
        """Check if document was successfully ingested."""
        return self.ingestion_status == IngestionStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if document ingestion failed."""
        return self.ingestion_status == IngestionStatus.FAILED
    
    @property
    def has_warnings(self) -> bool:
        """Check if there were any warnings during processing."""
        return len(self.warnings) > 0
    
    @property
    def primary_language(self) -> Optional[str]:
        """Get the primary (most common) language in the document."""
        if not self.language_distribution:
            return None
        return max(self.language_distribution, key=self.language_distribution.get)
    
    @property
    def is_multilingual(self) -> bool:
        """Check if document contains multiple languages."""
        return len(self.language_distribution) > 1
    
    @property
    def file_extension(self) -> str:
        """Get the file extension from the original path."""
        if "." in self.file_path:
            return self.file_path.rsplit(".", 1)[-1].lower()
        return ""
    
    @property
    def filename(self) -> str:
        """Get the filename from the original path."""
        return self.file_path.replace("\\", "/").rsplit("/", 1)[-1]
    
    def mark_processing(self) -> None:
        """Mark document as currently processing."""
        self.ingestion_status = IngestionStatus.PROCESSING
    
    def mark_completed(
        self,
        total_tokens: int,
        hierarchy_depth: int,
        parent_count: int,
        child_count: int,
        edge_count: int,
        language_distribution: Dict[str, float],
        processing_duration_ms: int
    ) -> None:
        """
        Mark document as successfully completed.
        
        Args:
            total_tokens: Total tokens in document.
            hierarchy_depth: Final hierarchy depth used.
            parent_count: Number of parent chunks created.
            child_count: Number of child chunks created.
            edge_count: Number of edges created.
            language_distribution: Language breakdown.
            processing_duration_ms: Processing time in ms.
        """
        self.ingestion_status = IngestionStatus.COMPLETED
        self.total_tokens = total_tokens
        self.hierarchy_depth = hierarchy_depth
        self.parent_count = parent_count
        self.child_count = child_count
        self.edge_count = edge_count
        self.language_distribution = language_distribution
        self.processing_duration_ms = processing_duration_ms
        self.completed_at = datetime.now()
    
    def mark_failed(self, error_message: str, processing_duration_ms: int) -> None:
        """
        Mark document as failed.
        
        Args:
            error_message: Description of the failure.
            processing_duration_ms: Processing time before failure.
        """
        self.ingestion_status = IngestionStatus.FAILED
        self.error_message = error_message
        self.processing_duration_ms = processing_duration_ms
        self.completed_at = datetime.now()
    
    def add_warning(self, warning: str) -> None:
        """
        Add a warning message.
        
        Args:
            warning: Warning message to add.
        """
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of this document.
        """
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "stored_path": self.stored_path,
            "file_hash": self.file_hash,
            "format": self.format.value,
            "ingestion_status": self.ingestion_status.value,
            "total_tokens": self.total_tokens,
            "hierarchy_depth": self.hierarchy_depth,
            "language_distribution": self.language_distribution,
            "parent_count": self.parent_count,
            "child_count": self.child_count,
            "edge_count": self.edge_count,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "upload_metadata": self.upload_metadata,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_duration_ms": self.processing_duration_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Create a Document from a dictionary.
        
        Args:
            data: Dictionary representation.
            
        Returns:
            A new Document instance.
        """
        return cls(
            document_id=data["document_id"],
            file_path=data["file_path"],
            stored_path=data["stored_path"],
            file_hash=data["file_hash"],
            format=DocumentFormat(data["format"]),
            ingestion_status=IngestionStatus(data["ingestion_status"]),
            total_tokens=data.get("total_tokens", 0),
            hierarchy_depth=data.get("hierarchy_depth", 1),
            language_distribution=data.get("language_distribution", {}),
            parent_count=data.get("parent_count", 0),
            child_count=data.get("child_count", 0),
            edge_count=data.get("edge_count", 0),
            error_message=data.get("error_message"),
            warnings=data.get("warnings", []),
            upload_metadata=data.get("upload_metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            processing_duration_ms=data.get("processing_duration_ms"),
        )


@dataclass
class IngestionResult:
    """
    Result of a document ingestion operation.
    
    Returned by the pipeline after processing a document,
    containing summary statistics and status information.
    """
    
    document_id: str # ID of the processed document.
    
    status: IngestionStatus # Final ingestion status.
    
    parent_count: int = 0 # Number of parent chunks created.
    
    child_count: int = 0 # Number of child chunks created.
    
    edge_count: int = 0 # Number of edges created (sibling + semantic).
    
    total_tokens: int = 0 # Total tokens processed.
    
    processing_duration_ms: int = 0 # Total processing time in milliseconds.
    
    error_message: Optional[str] = None # Error message if status is FAILED.
    
    warnings: List[str] = field(default_factory=list) # List of non-fatal warnings encountered.
    
    @property
    def is_success(self) -> bool:
        """Check if ingestion was successful."""
        return self.status == IngestionStatus.COMPLETED
    
    @property
    def is_failure(self) -> bool:
        """Check if ingestion failed."""
        return self.status == IngestionStatus.FAILED
    
    @property
    def has_warnings(self) -> bool:
        """Check if there were any warnings."""
        return len(self.warnings) > 0
    
    @property
    def total_chunks(self) -> int:
        """Total number of chunks (parent + child)."""
        return self.parent_count + self.child_count
    
    @property
    def processing_duration_seconds(self) -> float:
        """Processing duration in seconds."""
        return self.processing_duration_ms / 1000.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation.
        """
        return {
            "document_id": self.document_id,
            "status": self.status.value,
            "parent_count": self.parent_count,
            "child_count": self.child_count,
            "edge_count": self.edge_count,
            "total_tokens": self.total_tokens,
            "processing_duration_ms": self.processing_duration_ms,
            "error_message": self.error_message,
            "warnings": self.warnings,
        }
    
    @classmethod
    def from_document(cls, document: Document) -> "IngestionResult":
        """
        Create an IngestionResult from a completed Document.
        
        Args:
            document: The processed document.
            
        Returns:
            An IngestionResult summarizing the document.
        """
        return cls(
            document_id=document.document_id,
            status=document.ingestion_status,
            parent_count=document.parent_count,
            child_count=document.child_count,
            edge_count=document.edge_count,
            total_tokens=document.total_tokens,
            processing_duration_ms=document.processing_duration_ms or 0,
            error_message=document.error_message,
            warnings=document.warnings.copy(),
        )
    
    @classmethod
    def success(
        cls,
        document_id: str,
        parent_count: int,
        child_count: int,
        edge_count: int,
        total_tokens: int,
        processing_duration_ms: int,
        warnings: Optional[List[str]] = None
    ) -> "IngestionResult":
        """
        Create a successful ingestion result.
        
        Args:
            document_id: The document ID.
            parent_count: Number of parents created.
            child_count: Number of children created.
            edge_count: Number of edges created.
            total_tokens: Total tokens processed.
            processing_duration_ms: Processing time.
            warnings: Optional list of warnings.
            
        Returns:
            A successful IngestionResult.
        """
        return cls(
            document_id=document_id,
            status=IngestionStatus.COMPLETED,
            parent_count=parent_count,
            child_count=child_count,
            edge_count=edge_count,
            total_tokens=total_tokens,
            processing_duration_ms=processing_duration_ms,
            warnings=warnings or [],
        )
    
    @classmethod
    def failure(
        cls,
        document_id: str,
        error_message: str,
        processing_duration_ms: int = 0
    ) -> "IngestionResult":
        """
        Create a failed ingestion result.
        
        Args:
            document_id: The document ID.
            error_message: Description of the failure.
            processing_duration_ms: Processing time before failure.
            
        Returns:
            A failed IngestionResult.
        """
        return cls(
            document_id=document_id,
            status=IngestionStatus.FAILED,
            error_message=error_message,
            processing_duration_ms=processing_duration_ms,
        )
    
    def summary(self) -> str:
        """
        Generate a human-readable summary.
        
        Returns:
            Summary string.
        """
        if self.is_success:
            lines = [
                f"  Ingestion completed successfully",
                f"  Document ID: {self.document_id}",
                f"  Parents: {self.parent_count}",
                f"  Children: {self.child_count}",
                f"  Edges: {self.edge_count}",
                f"  Total tokens: {self.total_tokens}",
                f"  Duration: {self.processing_duration_seconds:.2f}s",
            ]
            if self.has_warnings:
                lines.append(f"  Warnings: {len(self.warnings)}")
                for warning in self.warnings:
                    lines.append(f"    - {warning}")
        else:
            lines = [
                f"  Ingestion failed",
                f"  Document ID: {self.document_id}",
                f"  Error: {self.error_message}",
                f"  Duration: {self.processing_duration_seconds:.2f}s",
            ]
        
        return "\n".join(lines)