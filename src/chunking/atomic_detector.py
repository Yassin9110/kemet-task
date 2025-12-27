"""
Atomic block detector for the Multilingual RAG Ingestion Pipeline.

Detects blocks that should not be split during chunking, such as tables.
"""

from typing import List, Set

from ..models.blocks import NormalizedBlock
from ..models.enums import BlockType


class AtomicDetector:
    """
    Detects and marks blocks that should not be split.
    
    Atomic blocks include:
    - Tables (should remain intact for coherence)
    
    These blocks will be kept whole even if they exceed
    the normal chunk size limits.
    """
    
    # Block types that are always atomic
    ATOMIC_TYPES: Set[BlockType] = {
        BlockType.TABLE,
    }
    
    def __init__(
        self,
        max_atomic_tokens: int = 1024,
        force_split_oversized: bool = False
    ):
        """
        Initialize atomic detector.
        
        Args:
            max_atomic_tokens: Maximum tokens for an atomic block.
                              Larger blocks may be force-split if enabled.
            force_split_oversized: Whether to split oversized atomic blocks.
        """
        self.max_atomic_tokens = max_atomic_tokens
        self.force_split_oversized = force_split_oversized
    
    def is_atomic(self, block: NormalizedBlock) -> bool:
        """
        Check if a block should be treated as atomic.
        
        Args:
            block: Block to check.
            
        Returns:
            True if block should not be split.
        """
        return block.block_type in self.ATOMIC_TYPES
    
    def is_atomic_by_type(self, block_type: BlockType) -> bool:
        """
        Check if a block type is atomic.
        
        Args:
            block_type: Block type to check.
            
        Returns:
            True if block type is atomic.
        """
        return block_type in self.ATOMIC_TYPES
    
    def filter_atomic(
        self,
        blocks: List[NormalizedBlock]
    ) -> tuple:
        """
        Separate atomic and non-atomic blocks.
        
        Args:
            blocks: List of blocks to filter.
            
        Returns:
            Tuple of (atomic_blocks, non_atomic_blocks).
        """
        atomic = []
        non_atomic = []
        
        for block in blocks:
            if self.is_atomic(block):
                atomic.append(block)
            else:
                non_atomic.append(block)
        
        return atomic, non_atomic
    
    def mark_atomic_blocks(
        self,
        blocks: List[NormalizedBlock]
    ) -> List[NormalizedBlock]:
        """
        Mark blocks with atomic flag in metadata.
        
        Args:
            blocks: List of blocks to process.
            
        Returns:
            Same blocks with updated metadata.
        """
        for block in blocks:
            block.metadata["is_atomic"] = self.is_atomic(block)
        
        return blocks
    
    def get_atomic_block_ids(
        self,
        blocks: List[NormalizedBlock]
    ) -> Set[str]:
        """
        Get IDs of all atomic blocks.
        
        Args:
            blocks: List of blocks to check.
            
        Returns:
            Set of atomic block IDs.
        """
        return {
            block.block_id
            for block in blocks
            if self.is_atomic(block)
        }
    
    def should_force_split(
        self,
        block: NormalizedBlock,
        token_count: int
    ) -> bool:
        """
        Check if an atomic block should be force-split due to size.
        
        Args:
            block: Block to check.
            token_count: Token count of the block.
            
        Returns:
            True if block should be split despite being atomic.
        """
        if not self.force_split_oversized:
            return False
        
        if not self.is_atomic(block):
            return False
        
        return token_count > self.max_atomic_tokens
    
    def get_atomic_types(self) -> List[str]:
        """
        Get list of atomic block type names.
        
        Returns:
            List of block type names that are atomic.
        """
        return [bt.value for bt in self.ATOMIC_TYPES]