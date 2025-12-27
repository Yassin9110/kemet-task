"""
Hierarchy resolver for the Multilingual RAG Ingestion Pipeline.

Determines the appropriate hierarchy depth (1, 2, or 3 levels)
based on document size and structure.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..models.blocks import NormalizedBlock
from ..config.settings import PipelineConfig
from ..models.enums import BlockType


@dataclass
class HierarchyDecision:
    """Result of hierarchy resolution."""
    
    depth: int
    """Hierarchy depth: 1, 2, or 3."""
    
    total_tokens: int
    """Total tokens in document."""
    
    reason: str
    """Explanation for the decision."""
    
    has_structure: bool
    """Whether document has heading structure."""
    
    heading_count: int
    """Number of headings found."""
    
    @property
    def uses_parents(self) -> bool:
        """Check if hierarchy uses parent chunks."""
        return self.depth >= 2
    
    @property
    def uses_sections(self) -> bool:
        """Check if hierarchy uses section-level grouping."""
        return self.depth == 3


class HierarchyResolver:
    """
    Resolves the appropriate hierarchy depth for a document.
    
    Hierarchy levels:
    - Level 1: Chunks only (small documents < 1,500 tokens)
    - Level 2: Parent → Children (medium documents 1,500 - 10,000 tokens)
    - Level 3: Document → Sections → Chunks (large documents > 10,000 tokens)
    
    The decision is based on:
    - Total token count
    - Presence of heading structure
    - Configuration thresholds
    """
    
    def __init__(self, small_doc_threshold: int = 1500, large_doc_threshold: int = 10000, min_headings_for_structure: int = 2):
        """
        Initialize hierarchy resolver.
        
        Args:
            small_doc_threshold: Token count below which to use 1 level.
            large_doc_threshold: Token count above which to use 3 levels.
            min_headings_for_structure: Minimum headings to consider structured.
        """
        self.small_doc_threshold = small_doc_threshold
        self.large_doc_threshold = large_doc_threshold
        self.min_headings_for_structure = min_headings_for_structure
    
    @classmethod
    def from_config(cls, config: PipelineConfig) -> "HierarchyResolver":
        """
        Create resolver from pipeline configuration.
        
        Args:
            config: Pipeline configuration.
            
        Returns:
            Configured HierarchyResolver.
        """
        return cls(small_doc_threshold=config.small_doc_threshold, large_doc_threshold=config.large_doc_threshold)
    
    def resolve(self, blocks: List[NormalizedBlock], total_tokens: int) -> HierarchyDecision:
        """
        Determine hierarchy depth for a document.
        
        Args:
            blocks: List of normalized blocks.
            total_tokens: Total token count of document.
            
        Returns:
            HierarchyDecision with depth and reasoning.
        """
        # Count headings
        heading_count = sum(1 for b in blocks if b.structural_hint == BlockType.HEADING)
        has_structure = heading_count >= self.min_headings_for_structure
        
        # Determine depth based on size
        if total_tokens < self.small_doc_threshold:
            depth = 1
            reason = f"Small document ({total_tokens} tokens < {self.small_doc_threshold})"
        
        elif total_tokens >= self.large_doc_threshold:
            depth = 3
            reason = f"Large document ({total_tokens} tokens >= {self.large_doc_threshold})"
        
        else:
            depth = 2
            reason = f"Medium document ({self.small_doc_threshold} <= {total_tokens} tokens < {self.large_doc_threshold})"
        
        # Add structure info to reason
        if has_structure:
            reason += f" with {heading_count} headings"
        else:
            reason += " without significant heading structure"
        
        return HierarchyDecision(
            depth=depth,
            total_tokens=total_tokens,
            reason=reason,
            has_structure=has_structure,
            heading_count=heading_count
        )
    
    def get_depth_for_tokens(self, total_tokens: int) -> int:
        """
        Get hierarchy depth based on token count only.
        
        Args:
            total_tokens: Total token count.
            
        Returns:
            Hierarchy depth (1, 2, or 3).
        """
        if total_tokens < self.small_doc_threshold:
            return 1
        elif total_tokens >= self.large_doc_threshold:
            return 3
        else:
            return 2
    
    def explain_depth(self, depth: int) -> str:
        """
        Get explanation of what a hierarchy depth means.
        
        Args:
            depth: Hierarchy depth (1, 2, or 3).
            
        Returns:
            Human-readable explanation.
        """
        explanations = {
            1: "Single level: Document → Chunks (no parent-child relationship)",
            2: "Two levels: Document → Parents → Children (parent chunks provide context)",
            3: "Three levels: Document → Sections → Parents → Children (section-aware grouping)"
        }
        return explanations.get(depth, f"Unknown depth: {depth}")