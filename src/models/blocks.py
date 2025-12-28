"""
Block dataclasses for the Multilingual RAG Ingestion Pipeline.

This module defines text block representations at different pipeline stages:
- ExtractedBlock: Raw text from document extraction (Stage 4)
- NormalizedBlock: Cleaned text with language info (Stages 5-6)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .enums import BlockType, Language


@dataclass
class ExtractedBlock:
    """
    A block of text extracted from a document.
    
    Represents the output of Stage 4 (Text Extraction).
    Contains raw, unprocessed text with structural metadata.
    """
    
    block_id: str # Unique identifier for this block (UUID).
    
    raw_text: str # Original extracted text, unprocessed.
    
    block_type: BlockType # Structural classification of the block.
    
    source_offset: int = 0 # Character offset position in the original document.
    
    page_number: Optional[int] = None # Source page number (1-indexed), if applicable.
    
    heading_level: Optional[int] = None # Heading level 1-6 if block_type is HEADING, else None.
    
    metadata: Dict[str, Any] = field(default_factory=dict) # Additional extractor-specific metadata.
    
    def __post_init__(self) -> None:
        """Validate block data after initialization."""
        # Validate heading_level
        if self.heading_level is not None:
            if not (1 <= self.heading_level <= 6):
                raise ValueError(f"heading_level must be 1-6, got {self.heading_level}")
            if self.block_type != BlockType.HEADING:
                raise ValueError(
                    f"heading_level should only be set for HEADING blocks, "
                    f"got block_type={self.block_type}"
                )
        
        # Validate page_number
        if self.page_number is not None and self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        
        # Validate source_offset
        if self.source_offset < 0:
            raise ValueError(f"source_offset cannot be negative, got {self.source_offset}")
    
    @property
    def text_length(self) -> int:
        """Length of the raw text in characters."""
        return len(self.raw_text)
    
    @property
    def is_empty(self) -> bool:
        """Check if the block contains no meaningful text."""
        return len(self.raw_text.strip()) == 0
    
    @property
    def is_heading(self) -> bool:
        """Check if this block is a heading."""
        return self.block_type == BlockType.HEADING
    
    @property
    def is_atomic(self) -> bool:
        """Check if this block should not be split (table or code)."""
        return self.block_type in (BlockType.TABLE, BlockType.CODE)


@dataclass
class NormalizedBlock:
    """
    A normalized block of text with language information.
    
    Represents the output of Stages 5-6 (Normalization & Language Detection).
    Contains cleaned text ready for structural parsing and chunking.
    """
    
    block_id: str # Unique identifier, same as source ExtractedBlock.
    
    text: str # Normalized and cleaned text.

    block_type: BlockType # Structural classification of the block.
    
    original_text: Optional[str] = None # Original extracted text for reference.

    source_offset: int = 0 # Character offset position in the original document.
    
    page_number: Optional[int] = None # Source page number (1-indexed), if applicable.
    
    heading_level: Optional[int] = None # Heading level 1-6 if block_type is HEADING, else None.
    
    metadata: Dict[str, Any] = field(default_factory=dict) # Additional metadata from extraction and normalization.
    
    def __post_init__(self) -> None:
        """Validate block data after initialization."""
        # Validate heading_level
        if self.heading_level is not None:
            if not (1 <= self.heading_level <= 6):
                raise ValueError(f"heading_level must be 1-6, got {self.heading_level}")
            if self.block_type != BlockType.HEADING:
                raise ValueError(
                    f"heading_level should only be set for HEADING blocks, "
                    f"got block_type={self.block_type}"
                )
        
        # Validate page_number
        if self.page_number is not None and self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        
        # Validate source_offset
        if self.source_offset < 0:
            raise ValueError(f"source_offset cannot be negative, got {self.source_offset}")
    
    @property
    def text_length(self) -> int:
        """Length of the normalized text in characters."""
        return len(self.text)
    
    @property
    def is_empty(self) -> bool:
        """Check if the block contains no meaningful text."""
        return len(self.text.strip()) == 0
    
    @property
    def is_heading(self) -> bool:
        """Check if this block is a heading."""
        return self.block_type == BlockType.HEADING
    
    @property
    def is_atomic(self) -> bool:
        """Check if this block should not be split (table or code)."""
        return self.block_type in (BlockType.TABLE, BlockType.CODE)
    
    @property
    def is_arabic(self) -> bool:
        """Check if block is primarily Arabic."""
        return self.language == Language.AR
    
    @property
    def is_english(self) -> bool:
        """Check if block is primarily English."""
        return self.language == Language.EN
    
    @property
    def is_mixed(self) -> bool:
        """Check if block contains mixed languages."""
        return self.language == Language.MIXED
    
    @classmethod
    def from_extracted_block(
        cls,
        extracted: ExtractedBlock,
        normalized_text: str,
        language: Language
    ) -> "NormalizedBlock":
        """
        Create a NormalizedBlock from an ExtractedBlock.
        
        Args:
            extracted: The source ExtractedBlock.
            normalized_text: The cleaned/normalized text.
            language: The detected language.
            
        Returns:
            A new NormalizedBlock instance.
        """
        return cls(
            block_id=extracted.block_id,
            text=normalized_text,
            original_text=extracted.raw_text,
            block_type=extracted.block_type,
            language=language,
            source_offset=extracted.source_offset,
            page_number=extracted.page_number,
            heading_level=extracted.heading_level,
            metadata=extracted.metadata.copy()
        )