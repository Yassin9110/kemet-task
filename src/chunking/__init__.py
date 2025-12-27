"""
Chunking package for the Multilingual RAG Ingestion Pipeline.

Provides chunking utilities:
- Atomic block detection (tables)
- Hierarchy resolution (1/2/3 levels)
- Parent chunk creation (structure-aware)
- Child chunk creation (with overlap and sibling links)
"""

from .atomic_detector import AtomicDetector

from .hierarchy_resolver import (
    HierarchyResolver,
    HierarchyDecision,
)

from .parent_chunker import (
    ParentChunker,
    ParentChunkCandidate,
)

from .child_chunker import (
    ChildChunker,
    ChunkingConfig,
    SentenceSplitter,
)


def create_chunking_components(
    max_tokens: int = 512,
    min_tokens: int = 80,
    child_target_tokens: int = 128,
    overlap_tokens: int = 50,
    arabic_overlap_multiplier: float = 1.2,
    small_doc_threshold: int = 1500,
    large_doc_threshold: int = 10000
) -> dict:
    """
    Create all chunking components with consistent configuration.
    
    Args:
        max_tokens: Maximum tokens per parent chunk.
        min_tokens: Minimum tokens for parent-child split.
        child_target_tokens: Target tokens per child chunk.
        overlap_tokens: Token overlap between children.
        arabic_overlap_multiplier: Multiplier for Arabic overlap.
        small_doc_threshold: Threshold for 1-level hierarchy.
        large_doc_threshold: Threshold for 3-level hierarchy.
        
    Returns:
        Dictionary with all chunking components.
    """
    atomic_detector = AtomicDetector()
    
    hierarchy_resolver = HierarchyResolver(
        small_doc_threshold=small_doc_threshold,
        large_doc_threshold=large_doc_threshold
    )
    
    parent_chunker = ParentChunker(
        max_tokens=max_tokens,
        min_tokens=min_tokens
    )
    
    chunking_config = ChunkingConfig(
        target_tokens=child_target_tokens,
        overlap_tokens=overlap_tokens,
        arabic_overlap_multiplier=arabic_overlap_multiplier
    )
    
    child_chunker = ChildChunker(config=chunking_config)
    
    return {
        "atomic_detector": atomic_detector,
        "hierarchy_resolver": hierarchy_resolver,
        "parent_chunker": parent_chunker,
        "child_chunker": child_chunker,
        "chunking_config": chunking_config,
    }


def create_chunking_components_from_config(config) -> dict:
    """
    Create chunking components from PipelineConfig.
    
    Args:
        config: PipelineConfig instance.
        
    Returns:
        Dictionary with all chunking components.
    """
    return create_chunking_components(
        max_tokens=config.max_tokens,
        min_tokens=config.min_tokens,
        child_target_tokens=config.child_target_tokens,
        overlap_tokens=config.overlap_tokens,
        arabic_overlap_multiplier=config.arabic_overlap_multiplier,
        small_doc_threshold=config.small_doc_threshold,
        large_doc_threshold=config.large_doc_threshold
    )


__all__ = [
    # Atomic Detection
    "AtomicDetector",
    # Hierarchy
    "HierarchyResolver",
    "HierarchyDecision",
    # Parent Chunking
    "ParentChunker",
    "ParentChunkCandidate",
    # Child Chunking
    "ChildChunker",
    "ChunkingConfig",
    "SentenceSplitter",
    # Factory Functions
    "create_chunking_components",
    "create_chunking_components_from_config",
]