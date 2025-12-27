"""
Stage 10: Graph Building

Creates structural edges (parent-child, sibling) and
optional semantic similarity edges between chunks.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from logging import Logger

from src.config.settings import PipelineConfig
from src.models.enums import DocumentFormat
from src.models.chunks import ParentChunk, ChildChunk
from src.models.edges import SemanticEdge
from src.pipeline.helpers.id_generator import generate_edge_id


@dataclass
class GraphBuildingInput:
    """Input for the graph building stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    parent_chunks: List[ParentChunk]
    child_chunks: List[ChildChunk]
    hierarchy_depth: int
    total_tokens: int


@dataclass
class GraphBuildingOutput:
    """Output from the graph building stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    parent_chunks: List[ParentChunk]
    child_chunks: List[ChildChunk]
    semantic_edges: List[SemanticEdge]
    hierarchy_depth: int
    total_tokens: int
    graph_stats: dict


class GraphBuildingStage:
    """
    Stage 10: Graph Building
    
    Responsibilities:
    - Verify structural edges (parent-child, sibling) are in place
    - Compute semantic similarity edges (if enabled)
    - Store edges for graph-based retrieval
    
    Edge Types:
    - Parent → Child: Stored in child.parent_id
    - Child → Siblings: Stored in child.prev_chunk_id, child.next_chunk_id
    - Semantic similarity: Stored as SemanticEdge objects
    """
    
    STAGE_NAME = "Graph Building"
    STAGE_NUMBER = 10
    TOTAL_STAGES = 11
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the graph building stage.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
    
    def execute(
        self,
        input_data: GraphBuildingInput,
        logger: Logger
    ) -> GraphBuildingOutput:
        """
        Execute the graph building stage.
        
        Args:
            input_data: Graph building input with chunks.
            logger: Logger instance for progress tracking.
            
        Returns:
            GraphBuildingOutput with semantic edges.
        """
        start_time = time.time()
        
        parent_chunks = input_data.parent_chunks
        child_chunks = input_data.child_chunks
        
        # Statistics
        stats = {
            'parent_child_edges': 0,
            'sibling_edges': 0,
            'semantic_edges': 0,
            'semantic_edges_enabled': self.config.compute_semantic_edges,
            'similarity_threshold': self.config.semantic_similarity_threshold,
        }
        
        # Count structural edges
        stats['parent_child_edges'] = self._count_parent_child_edges(child_chunks)
        stats['sibling_edges'] = self._count_sibling_edges(child_chunks)
        
        # Compute semantic edges if enabled
        semantic_edges: List[SemanticEdge] = []
        
        if self.config.compute_semantic_edges:
            semantic_edges = self._compute_semantic_edges(
                child_chunks=child_chunks,
                threshold=self.config.semantic_similarity_threshold
            )
            stats['semantic_edges'] = len(semantic_edges)
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log completion
        logger.info(
            f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
            f"{self.STAGE_NAME} ✓ ({duration_ms}ms)"
        )
        logger.info(f"  → Created {stats['sibling_edges']} sibling edges")
        
        if self.config.compute_semantic_edges:
            logger.info(
                f"  → Created {stats['semantic_edges']} semantic edges "
                f"(threshold: {self.config.semantic_similarity_threshold})"
            )
        else:
            logger.info("  → Semantic edges disabled")
        
        return GraphBuildingOutput(
            document_id=input_data.document_id,
            stored_path=input_data.stored_path,
            format=input_data.format,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
            semantic_edges=semantic_edges,
            hierarchy_depth=input_data.hierarchy_depth,
            total_tokens=input_data.total_tokens,
            graph_stats=stats
        )
    
    def _count_parent_child_edges(self, child_chunks: List[ChildChunk]) -> int:
        """
        Count parent-child edges.
        
        Args:
            child_chunks: List of child chunks.
            
        Returns:
            Number of parent-child edges.
        """
        return sum(1 for chunk in child_chunks if chunk.parent_id is not None)
    
    def _count_sibling_edges(self, child_chunks: List[ChildChunk]) -> int:
        """
        Count sibling edges (prev/next links).
        
        Args:
            child_chunks: List of child chunks.
            
        Returns:
            Number of sibling edges.
        """
        prev_edges = sum(1 for chunk in child_chunks if chunk.prev_chunk_id is not None)
        next_edges = sum(1 for chunk in child_chunks if chunk.next_chunk_id is not None)
        
        # Each sibling relationship is counted twice (prev and next)
        # Return unique edge count
        return max(prev_edges, next_edges)
    
    def _compute_semantic_edges(
        self,
        child_chunks: List[ChildChunk],
        threshold: float
    ) -> List[SemanticEdge]:
        """
        Compute semantic similarity edges between chunks.
        
        Only computes intra-document edges (within same document).
        
        Args:
            child_chunks: List of child chunks with embeddings.
            threshold: Minimum similarity threshold for edge creation.
            
        Returns:
            List of semantic edges.
        """
        edges: List[SemanticEdge] = []
        
        # Filter chunks with embeddings
        chunks_with_embeddings = [
            chunk for chunk in child_chunks
            if chunk.embedding is not None
        ]
        
        if len(chunks_with_embeddings) < 2:
            return edges
        
        # Compute pairwise similarities
        n = len(chunks_with_embeddings)
        
        for i in range(n):
            chunk_i = chunks_with_embeddings[i]
            
            for j in range(i + 1, n):
                chunk_j = chunks_with_embeddings[j]
                
                # Skip if already siblings (adjacent chunks)
                if self._are_siblings(chunk_i, chunk_j):
                    continue
                
                # Compute cosine similarity
                similarity = self._cosine_similarity(
                    chunk_i.embedding,
                    chunk_j.embedding
                )
                
                # Create edge if above threshold
                if similarity >= threshold:
                    edge = SemanticEdge(
                        edge_id=generate_edge_id(),
                        source_chunk_id=chunk_i.chunk_id,
                        target_chunk_id=chunk_j.chunk_id,
                        edge_type="semantic",
                        document_id=chunk_i.document_id,
                        similarity_score=round(similarity, 4)
                    )
                    edges.append(edge)
        
        return edges
    
    def _are_siblings(self, chunk_a: ChildChunk, chunk_b: ChildChunk) -> bool:
        """
        Check if two chunks are adjacent siblings.
        
        Args:
            chunk_a: First chunk.
            chunk_b: Second chunk.
            
        Returns:
            True if chunks are adjacent siblings.
        """
        return (
            chunk_a.next_chunk_id == chunk_b.chunk_id or
            chunk_a.prev_chunk_id == chunk_b.chunk_id or
            chunk_b.next_chunk_id == chunk_a.chunk_id or
            chunk_b.prev_chunk_id == chunk_a.chunk_id
        )
    
    def _cosine_similarity(
        self,
        vec_a: List[float],
        vec_b: List[float]
    ) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec_a: First vector.
            vec_b: Second vector.
            
        Returns:
            Cosine similarity score (0 to 1).
        """
        if not vec_a or not vec_b:
            return 0.0
        
        if len(vec_a) != len(vec_b):
            return 0.0
        
        # Compute dot product and magnitudes
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = sum(a * a for a in vec_a) ** 0.5
        magnitude_b = sum(b * b for b in vec_b) ** 0.5
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)