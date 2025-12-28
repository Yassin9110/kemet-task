"""
Stage 5: Normalization & Cleaning

Applies text normalization including Unicode normalization,
Arabic-specific normalization, and whitespace cleaning.
"""

import time
from dataclasses import dataclass
from typing import List, Optional
from logging import Logger

from src.config.settings import PipelineConfig
from src.models.enums import DocumentFormat
from src.models.blocks import ExtractedBlock, NormalizedBlock
from src.normalizers.base import BaseNormalizer
from src.normalizers.unicode_normalizer import UnicodeNormalizer
from src.normalizers.arabic_normalizer import ArabicNormalizer
from src.normalizers.whitespace_normalizer import WhitespaceNormalizer


@dataclass
class NormalizationInput:
    """Input for the normalization stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    blocks: List[ExtractedBlock]


@dataclass
class NormalizationOutput:
    """Output from the normalization stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    blocks: List[NormalizedBlock]
    normalization_stats: dict


class NormalizationStage:
    """
    Stage 5: Normalization & Cleaning
    
    Responsibilities:
    - Apply Unicode NFC normalization
    - Remove invisible characters
    - Apply Arabic-specific normalization (Alef, Tashkeel, Tatweel)
    - Normalize whitespace
    - Track normalization statistics
    
    NOT applied:
    - Translation
    - Rewriting
    - Summarization
    """
    
    STAGE_NAME = "Normalization"
    STAGE_NUMBER = 5
    TOTAL_STAGES = 11
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the normalization stage.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        
        # Initialize normalizers in order of application
        self._normalizers: List[BaseNormalizer] = [
            UnicodeNormalizer(config),
            ArabicNormalizer(config),
            WhitespaceNormalizer(config),
        ]
    
    def execute(
        self,
        input_data: NormalizationInput,
        logger: Logger
    ) -> NormalizationOutput:
        """
        Execute the normalization stage.
        
        Args:
            input_data: Normalization input with extracted blocks.
            logger: Logger instance for progress tracking.
            
        Returns:
            NormalizationOutput with normalized blocks and statistics.
        """
        start_time = time.time()
        
        blocks = input_data.blocks
        normalized_blocks: List[NormalizedBlock] = []
        
        # Statistics tracking
        stats = {
            'total_blocks': len(blocks),
            'total_chars_before': 0,
            'total_chars_after': 0,
            'arabic_normalization_applied': False,
            'blocks_modified': 0,
            'empty_blocks_removed': 0,
        }
        
        for block in blocks:
            original_text = block.raw_text
            stats['total_chars_before'] += len(original_text)
            
            # Apply normalization chain
            normalized_text = original_text
            for normalizer in self._normalizers:
                normalized_text = normalizer.normalize(normalized_text)
            
            # Track if Arabic normalization was meaningful
            if self._contains_arabic(original_text):
                stats['arabic_normalization_applied'] = True
            
            # Skip blocks that become empty after normalization
            if not normalized_text or not normalized_text.strip():
                stats['empty_blocks_removed'] += 1
                continue
            
            stats['total_chars_after'] += len(normalized_text)
            
            # Track modifications
            if normalized_text != original_text:
                stats['blocks_modified'] += 1
            
            # Create normalized block
            normalized_block = NormalizedBlock(
                block_id=block.block_id,
                text=normalized_text,
                original_text= original_text,
                block_type=block.block_type,
                page_number=block.page_number,
                heading_level=block.heading_level,
                source_offset=block.source_offset
            )
            
            normalized_blocks.append(normalized_block)
        
        # Calculate compression ratio
        if stats['total_chars_before'] > 0:
            stats['compression_ratio'] = round(
                stats['total_chars_after'] / stats['total_chars_before'],
                3
            )
        else:
            stats['compression_ratio'] = 1.0
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log completion
        logger.info(
            f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
            f"{self.STAGE_NAME} ✓ ({duration_ms}ms)"
        )
        
        # Log normalization details
        if stats['arabic_normalization_applied']:
            logger.info("  → Applied Arabic normalization")
        
        if stats['blocks_modified'] > 0:
            logger.info(
                f"  → Modified {stats['blocks_modified']}/{stats['total_blocks']} blocks"
            )
        
        if stats['empty_blocks_removed'] > 0:
            logger.warning(
                f"  → Removed {stats['empty_blocks_removed']} empty blocks after normalization"
            )
        
        return NormalizationOutput(
            document_id=input_data.document_id,
            stored_path=input_data.stored_path,
            format=input_data.format,
            blocks=normalized_blocks,
            normalization_stats=stats
        )
    
    def _contains_arabic(self, text: str) -> bool:
        """
        Check if text contains Arabic characters.
        
        Args:
            text: Text to check.
            
        Returns:
            True if text contains Arabic characters.
        """
        # Arabic Unicode ranges
        for char in text:
            code_point = ord(char)
            # Arabic (0600-06FF), Arabic Supplement (0750-077F),
            # Arabic Extended-A (08A0-08FF), Arabic Presentation Forms
            if (0x0600 <= code_point <= 0x06FF or
                0x0750 <= code_point <= 0x077F or
                0x08A0 <= code_point <= 0x08FF or
                0xFB50 <= code_point <= 0xFDFF or
                0xFE70 <= code_point <= 0xFEFF):
                return True
        return False


class NormalizationChain:
    """
    Utility class for creating custom normalization chains.
    
    Allows fine-grained control over which normalizers are applied
    and in what order.
    """
    
    def __init__(self):
        """Initialize an empty normalization chain."""
        self._normalizers: List[BaseNormalizer] = []
    
    def add(self, normalizer: BaseNormalizer) -> 'NormalizationChain':
        """
        Add a normalizer to the chain.
        
        Args:
            normalizer: Normalizer to add.
            
        Returns:
            Self for method chaining.
        """
        self._normalizers.append(normalizer)
        return self
    
    def normalize(self, text: str) -> str:
        """
        Apply all normalizers in order.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        result = text
        for normalizer in self._normalizers:
            result = normalizer.normalize(result)
        return result
    
    def normalize_block(self, block: ExtractedBlock) -> NormalizedBlock:
        """
        Normalize an extracted block.
        
        Args:
            block: Extracted block to normalize.
            
        Returns:
            Normalized block.
        """
        normalized_text = self.normalize(block.raw_text)
        
        return NormalizedBlock(
            block_id=block.block_id,
            text=normalized_text,
            original_text= block.raw_text,
            block_type=block.block_type,
            page_number=block.page_number,
            heading_level=block.heading_level,
            source_offset=block.source_offset
        )
    
    def normalize_blocks(self, blocks: List[ExtractedBlock]) -> List[NormalizedBlock]:
        """
        Normalize a list of extracted blocks.
        
        Args:
            blocks: List of extracted blocks.
            
        Returns:
            List of normalized blocks (empty blocks filtered out).
        """
        normalized = []
        for block in blocks:
            norm_block = self.normalize_block(block)
            if norm_block.normalized_text and norm_block.normalized_text.strip():
                normalized.append(norm_block)
        return normalized