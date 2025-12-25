"""
Chunk dataclasses for the Multilingual RAG Ingestion Pipeline.

This module defines the parent-child chunk hierarchy:
- ParentChunk: Larger context chunks for LLM context (~512 tokens)
- ChildChunk: Smaller chunks for precise retrieval (~128 tokens)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from .enums import Language, BlockType


@dataclass
class ParentChunk:
    """
    A parent chunk containing larger context for LLM consumption.
    
    Parent chunks are created from document sections or fixed windows,
    typically around 512 tokens. They contain one or more child chunks
    and are returned to the LLM for context after retrieval.
    """
    
    parent_id: str
    """Unique identifier for this parent chunk (UUID)."""
    
    document_id: str
    """ID of the source document."""
    
    text: str
    """Full text content of the parent chunk."""
    
    token_count: int
    """Number of tokens in this chunk."""
    
    section_path: List[str] = field(default_factory=list)
    """Hierarchical section path, e.g., ['Chapter 1', 'Introduction']."""
    
    page_range: Optional[Tuple[int, int]] = None
    """Start and end page numbers (1-indexed), inclusive."""
    
    language: Language = Language.EN
    """Primary language of the chunk content."""
    
    child_ids: List[str] = field(default_factory=list)
    """List of child chunk IDs contained in this parent."""
    
    created_at: datetime = field(default_factory=datetime.now)
    """Timestamp when this chunk was created."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""
    
    def __post_init__(self) -> None:
        """Validate chunk data after initialization."""
        errors = []
        
        if not self.parent_id:
            errors.append("parent_id cannot be empty")
        
        if not self.document_id:
            errors.append("document_id cannot be empty")
        
        if self.token_count < 0:
            errors.append(f"token_count cannot be negative, got {self.token_count}")
        
        if self.page_range is not None:
            start, end = self.page_range
            if start < 1:
                errors.append(f"page_range start must be >= 1, got {start}")
            if end < start:
                errors.append(f"page_range end must be >= start, got {start}-{end}")
        
        if errors:
            raise ValueError("ParentChunk validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @property
    def is_empty(self) -> bool:
        """Check if the chunk contains no meaningful text."""
        return len(self.text.strip()) == 0
    
    @property
    def text_length(self) -> int:
        """Length of the text in characters."""
        return len(self.text)
    
    @property
    def child_count(self) -> int:
        """Number of child chunks."""
        return len(self.child_ids)
    
    @property
    def has_children(self) -> bool:
        """Check if this parent has any children."""
        return len(self.child_ids) > 0
    
    @property
    def section_depth(self) -> int:
        """Depth of the section path."""
        return len(self.section_path)
    
    @property
    def section_name(self) -> Optional[str]:
        """Get the immediate section name (last element of path)."""
        if self.section_path:
            return self.section_path[-1]
        return None
    
    @property
    def is_arabic(self) -> bool:
        """Check if chunk is primarily Arabic."""
        return self.language == Language.AR
    
    @property
    def is_english(self) -> bool:
        """Check if chunk is primarily English."""
        return self.language == Language.EN
    
    @property
    def is_mixed(self) -> bool:
        """Check if chunk contains mixed languages."""
        return self.language == Language.MIXED
    
    def add_child(self, child_id: str) -> None:
        """
        Add a child chunk ID to this parent.
        
        Args:
            child_id: The ID of the child chunk to add.
        """
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of this chunk.
        """
        return {
            "parent_id": self.parent_id,
            "document_id": self.document_id,
            "text": self.text,
            "token_count": self.token_count,
            "section_path": self.section_path,
            "page_range": list(self.page_range) if self.page_range else None,
            "language": self.language.value,
            "child_ids": self.child_ids,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParentChunk":
        """
        Create a ParentChunk from a dictionary.
        
        Args:
            data: Dictionary representation.
            
        Returns:
            A new ParentChunk instance.
        """
        return cls(
            parent_id=data["parent_id"],
            document_id=data["document_id"],
            text=data["text"],
            token_count=data["token_count"],
            section_path=data.get("section_path", []),
            page_range=tuple(data["page_range"]) if data.get("page_range") else None,
            language=Language(data["language"]),
            child_ids=data.get("child_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChildChunk:
    """
    A child chunk for precise vector retrieval.
    
    Child chunks are smaller subdivisions of parent chunks, typically
    around 128 tokens. They are embedded and stored in the vector database
    for retrieval. On match, the parent chunk is returned for context.
    """
    
    chunk_id: str
    """Unique identifier for this child chunk (UUID)."""
    
    document_id: str
    """ID of the source document."""
    
    text: str
    """Text content of this chunk."""
    
    token_count: int
    """Number of tokens in this chunk."""
    
    language: Language
    """Detected language of this chunk."""
    
    block_type: BlockType
    """Structural type of the source block."""
    
    parent_id: Optional[str] = None
    """ID of parent chunk (None if document uses 1-level hierarchy)."""
    
    section_path: List[str] = field(default_factory=list)
    """Hierarchical section path."""
    
    page_number: Optional[int] = None
    """Source page number (1-indexed)."""
    
    position_in_parent: int = 0
    """Position within parent chunk (0-indexed)."""
    
    prev_chunk_id: Optional[str] = None
    """ID of the previous sibling chunk."""
    
    next_chunk_id: Optional[str] = None
    """ID of the next sibling chunk."""
    
    embedding: Optional[List[float]] = None
    """Vector embedding (populated after embedding stage)."""
    
    created_at: datetime = field(default_factory=datetime.now)
    """Timestamp when this chunk was created."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""
    
    def __post_init__(self) -> None:
        """Validate chunk data after initialization."""
        errors = []
        
        if not self.chunk_id:
            errors.append("chunk_id cannot be empty")
        
        if not self.document_id:
            errors.append("document_id cannot be empty")
        
        if self.token_count < 0:
            errors.append(f"token_count cannot be negative, got {self.token_count}")
        
        if self.page_number is not None and self.page_number < 1:
            errors.append(f"page_number must be >= 1, got {self.page_number}")
        
        if self.position_in_parent < 0:
            errors.append(f"position_in_parent cannot be negative, got {self.position_in_parent}")
        
        if errors:
            raise ValueError("ChildChunk validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @property
    def is_empty(self) -> bool:
        """Check if the chunk contains no meaningful text."""
        return len(self.text.strip()) == 0
    
    @property
    def text_length(self) -> int:
        """Length of the text in characters."""
        return len(self.text)
    
    @property
    def has_parent(self) -> bool:
        """Check if this chunk has a parent."""
        return self.parent_id is not None
    
    @property
    def has_embedding(self) -> bool:
        """Check if embedding has been computed."""
        return self.embedding is not None
    
    @property
    def embedding_dimension(self) -> Optional[int]:
        """Get the dimension of the embedding vector."""
        if self.embedding is not None:
            return len(self.embedding)
        return None
    
    @property
    def has_prev_sibling(self) -> bool:
        """Check if there is a previous sibling chunk."""
        return self.prev_chunk_id is not None
    
    @property
    def has_next_sibling(self) -> bool:
        """Check if there is a next sibling chunk."""
        return self.next_chunk_id is not None
    
    @property
    def is_first_in_parent(self) -> bool:
        """Check if this is the first chunk in its parent."""
        return self.position_in_parent == 0
    
    @property
    def section_depth(self) -> int:
        """Depth of the section path."""
        return len(self.section_path)
    
    @property
    def section_name(self) -> Optional[str]:
        """Get the immediate section name (last element of path)."""
        if self.section_path:
            return self.section_path[-1]
        return None
    
    @property
    def is_arabic(self) -> bool:
        """Check if chunk is primarily Arabic."""
        return self.language == Language.AR
    
    @property
    def is_english(self) -> bool:
        """Check if chunk is primarily English."""
        return self.language == Language.EN
    
    @property
    def is_mixed(self) -> bool:
        """Check if chunk contains mixed languages."""
        return self.language == Language.MIXED
    
    @property
    def is_atomic(self) -> bool:
        """Check if this chunk is from an atomic block (table or code)."""
        return self.block_type in (BlockType.TABLE, BlockType.CODE)
    
    def set_embedding(self, embedding: List[float]) -> None:
        """
        Set the embedding vector for this chunk.
        
        Args:
            embedding: The embedding vector.
        """
        self.embedding = embedding
    
    def link_prev(self, prev_id: str) -> None:
        """
        Link to previous sibling chunk.
        
        Args:
            prev_id: ID of the previous chunk.
        """
        self.prev_chunk_id = prev_id
    
    def link_next(self, next_id: str) -> None:
        """
        Link to next sibling chunk.
        
        Args:
            next_id: ID of the next chunk.
        """
        self.next_chunk_id = next_id
    
    def to_dict(self, include_embedding: bool = True) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Args:
            include_embedding: Whether to include the embedding vector.
            
        Returns:
            Dictionary representation of this chunk.
        """
        result = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "token_count": self.token_count,
            "language": self.language.value,
            "block_type": self.block_type.value,
            "parent_id": self.parent_id,
            "section_path": self.section_path,
            "page_number": self.page_number,
            "position_in_parent": self.position_in_parent,
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
        
        if include_embedding:
            result["embedding"] = self.embedding
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChildChunk":
        """
        Create a ChildChunk from a dictionary.
        
        Args:
            data: Dictionary representation.
            
        Returns:
            A new ChildChunk instance.
        """
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            text=data["text"],
            token_count=data["token_count"],
            language=Language(data["language"]),
            block_type=BlockType(data["block_type"]),
            parent_id=data.get("parent_id"),
            section_path=data.get("section_path", []),
            page_number=data.get("page_number"),
            position_in_parent=data.get("position_in_parent", 0),
            prev_chunk_id=data.get("prev_chunk_id"),
            next_chunk_id=data.get("next_chunk_id"),
            embedding=data.get("embedding"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )