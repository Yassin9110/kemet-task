"""
Edge dataclasses for the Multilingual RAG Ingestion Pipeline.

This module defines graph edge representations:
- SemanticEdge: Relationships between chunks (semantic similarity, siblings)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional


class EdgeType:
    """Constants for edge types."""
    
    SEMANTIC = "semantic" # Edge based on semantic similarity between chunks.
    
    SIBLING_PREV = "sibling_prev" # Edge to previous sibling chunk.
    
    SIBLING_NEXT = "sibling_next" # Edge to next sibling chunk.
    
    PARENT_CHILD = "parent_child" # Edge from parent to child chunk.
    
    @classmethod
    def all_types(cls) -> list:
        """Get all valid edge types."""
        return [cls.SEMANTIC, cls.SIBLING_PREV, cls.SIBLING_NEXT, cls.PARENT_CHILD]
    
    @classmethod
    def is_valid(cls, edge_type: str) -> bool:
        """Check if an edge type is valid."""
        return edge_type in cls.all_types()


@dataclass
class SemanticEdge:
    """
    An edge representing a relationship between two chunks.
    
    Edges can represent:
    - Semantic similarity (computed post-embedding)
    - Sibling relationships (prev/next in sequence)
    - Parent-child relationships
    """
    
    edge_id: str # Unique identifier for this edge (UUID).
    
    source_chunk_id: str # ID of the source chunk.
    
    target_chunk_id: str # ID of the target chunk.
    
    edge_type: str # Type of relationship (semantic, sibling_prev, sibling_next, parent_child).
    
    document_id: str # Document ID for filtering (source chunk's document).
    
    similarity_score: Optional[float] = None # Cosine similarity score (0-1), only for semantic edges.
    
    created_at: datetime = field(default_factory=datetime.now) # Timestamp when this edge was created.
    
    metadata: Dict[str, Any] = field(default_factory=dict) # Additional metadata.
    
    def __post_init__(self) -> None:
        """Validate edge data after initialization."""
        errors = []
        
        if not self.edge_id:
            errors.append("edge_id cannot be empty")
        
        if not self.source_chunk_id:
            errors.append("source_chunk_id cannot be empty")
        
        if not self.target_chunk_id:
            errors.append("target_chunk_id cannot be empty")
        
        if self.source_chunk_id == self.target_chunk_id:
            errors.append("source_chunk_id and target_chunk_id cannot be the same")
        
        if not self.document_id:
            errors.append("document_id cannot be empty")
        
        if not EdgeType.is_valid(self.edge_type):
            errors.append(f"Invalid edge_type: {self.edge_type}. Must be one of: {EdgeType.all_types()}")
        
        if self.similarity_score is not None:
            if not (0 <= self.similarity_score <= 1):
                errors.append(f"similarity_score must be between 0 and 1, got {self.similarity_score}")
        
        if self.edge_type == EdgeType.SEMANTIC and self.similarity_score is None:
            errors.append("similarity_score is required for semantic edges")
        
        if errors:
            raise ValueError("SemanticEdge validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @property
    def is_semantic(self) -> bool:
        """Check if this is a semantic similarity edge."""
        return self.edge_type == EdgeType.SEMANTIC
    
    @property
    def is_sibling(self) -> bool:
        """Check if this is a sibling relationship edge."""
        return self.edge_type in (EdgeType.SIBLING_PREV, EdgeType.SIBLING_NEXT)
    
    @property
    def is_sibling_prev(self) -> bool:
        """Check if this is a previous sibling edge."""
        return self.edge_type == EdgeType.SIBLING_PREV
    
    @property
    def is_sibling_next(self) -> bool:
        """Check if this is a next sibling edge."""
        return self.edge_type == EdgeType.SIBLING_NEXT
    
    @property
    def is_parent_child(self) -> bool:
        """Check if this is a parent-child edge."""
        return self.edge_type == EdgeType.PARENT_CHILD
    
    @property
    def is_strong_similarity(self) -> bool:
        """Check if similarity is strong (>= 0.9)."""
        if self.similarity_score is None:
            return False
        return self.similarity_score >= 0.9
    
    @property
    def is_moderate_similarity(self) -> bool:
        """Check if similarity is moderate (>= 0.8 and < 0.9)."""
        if self.similarity_score is None:
            return False
        return 0.8 <= self.similarity_score < 0.9
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of this edge.
        """
        return {
            "edge_id": self.edge_id,
            "source_chunk_id": self.source_chunk_id,
            "target_chunk_id": self.target_chunk_id,
            "edge_type": self.edge_type,
            "document_id": self.document_id,
            "similarity_score": self.similarity_score,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticEdge":
        """
        Create a SemanticEdge from a dictionary.
        
        Args:
            data: Dictionary representation.
            
        Returns:
            A new SemanticEdge instance.
        """
        return cls(
            edge_id=data["edge_id"],
            source_chunk_id=data["source_chunk_id"],
            target_chunk_id=data["target_chunk_id"],
            edge_type=data["edge_type"],
            document_id=data["document_id"],
            similarity_score=data.get("similarity_score"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def create_semantic(
        cls,
        edge_id: str,
        source_chunk_id: str,
        target_chunk_id: str,
        document_id: str,
        similarity_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "SemanticEdge":
        """
        Create a semantic similarity edge.
        
        Args:
            edge_id: Unique edge ID.
            source_chunk_id: Source chunk ID.
            target_chunk_id: Target chunk ID.
            document_id: Document ID.
            similarity_score: Cosine similarity (0-1).
            metadata: Optional additional metadata.
            
        Returns:
            A new SemanticEdge with type SEMANTIC.
        """
        return cls(
            edge_id=edge_id,
            source_chunk_id=source_chunk_id,
            target_chunk_id=target_chunk_id,
            edge_type=EdgeType.SEMANTIC,
            document_id=document_id,
            similarity_score=similarity_score,
            metadata=metadata or {},
        )
    
    @classmethod
    def create_sibling_next(
        cls,
        edge_id: str,
        source_chunk_id: str,
        target_chunk_id: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "SemanticEdge":
        """
        Create a next sibling edge.
        
        Args:
            edge_id: Unique edge ID.
            source_chunk_id: Source chunk ID.
            target_chunk_id: Next sibling chunk ID.
            document_id: Document ID.
            metadata: Optional additional metadata.
            
        Returns:
            A new SemanticEdge with type SIBLING_NEXT.
        """
        return cls(
            edge_id=edge_id,
            source_chunk_id=source_chunk_id,
            target_chunk_id=target_chunk_id,
            edge_type=EdgeType.SIBLING_NEXT,
            document_id=document_id,
            metadata=metadata or {},
        )
    
    @classmethod
    def create_sibling_prev(
        cls,
        edge_id: str,
        source_chunk_id: str,
        target_chunk_id: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "SemanticEdge":
        """
        Create a previous sibling edge.
        
        Args:
            edge_id: Unique edge ID.
            source_chunk_id: Source chunk ID.
            target_chunk_id: Previous sibling chunk ID.
            document_id: Document ID.
            metadata: Optional additional metadata.
            
        Returns:
            A new SemanticEdge with type SIBLING_PREV.
        """
        return cls(
            edge_id=edge_id,
            source_chunk_id=source_chunk_id,
            target_chunk_id=target_chunk_id,
            edge_type=EdgeType.SIBLING_PREV,
            document_id=document_id,
            metadata=metadata or {},
        )
    
    @classmethod
    def create_parent_child(
        cls,
        edge_id: str,
        parent_id: str,
        child_id: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "SemanticEdge":
        """
        Create a parent-child edge.
        
        Args:
            edge_id: Unique edge ID.
            parent_id: Parent chunk ID.
            child_id: Child chunk ID.
            document_id: Document ID.
            metadata: Optional additional metadata.
            
        Returns:
            A new SemanticEdge with type PARENT_CHILD.
        """
        return cls(
            edge_id=edge_id,
            source_chunk_id=parent_id,
            target_chunk_id=child_id,
            edge_type=EdgeType.PARENT_CHILD,
            document_id=document_id,
            metadata=metadata or {},
        )
    
    def reverse(self, new_edge_id: str) -> "SemanticEdge":
        """
        Create a reverse edge (swap source and target).
        
        Args:
            new_edge_id: ID for the new reversed edge.
            
        Returns:
            A new SemanticEdge with swapped source and target.
        """
        # Swap sibling direction if applicable
        new_edge_type = self.edge_type
        if self.edge_type == EdgeType.SIBLING_NEXT:
            new_edge_type = EdgeType.SIBLING_PREV
        elif self.edge_type == EdgeType.SIBLING_PREV:
            new_edge_type = EdgeType.SIBLING_NEXT
        
        return SemanticEdge(
            edge_id=new_edge_id,
            source_chunk_id=self.target_chunk_id,
            target_chunk_id=self.source_chunk_id,
            edge_type=new_edge_type,
            document_id=self.document_id,
            similarity_score=self.similarity_score,
            metadata=self.metadata.copy(),
        )