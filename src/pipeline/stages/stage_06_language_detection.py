"""
Stage 6: Language Detection

Detects language per block (Arabic, English, or Mixed)
based on character analysis.
"""

import time
from dataclasses import dataclass
from typing import List, Dict
from logging import Logger

from src.config.settings import PipelineConfig
from src.models.enums import DocumentFormat, Language
from src.models.blocks import NormalizedBlock
from src.models.enums import BlockType


@dataclass
class LanguageBlock:
    """Block with language information attached."""
    block_id: str
    block_type: BlockType
    normalized_text: str
    original_text: str
    page_number: int | None
    heading_level: int | None
    source_offset: int
    language: Language


@dataclass
class LanguageDetectionInput:
    """Input for the language detection stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    blocks: List[NormalizedBlock]


@dataclass
class LanguageDetectionOutput:
    """Output from the language detection stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    blocks: List[LanguageBlock]
    language_distribution: Dict[str, float]
    primary_language: Language


class LanguageDetectionStage:
    """
    Stage 6: Language Detection
    
    Responsibilities:
    - Detect language per block (not per document)
    - Support Arabic (ar), English (en), and Mixed
    - Calculate document-level language distribution
    - Determine primary language
    
    Detection is based on Unicode character ranges, not ML models.
    """
    
    STAGE_NAME = "Language Detection"
    STAGE_NUMBER = 6
    TOTAL_STAGES = 11
    
    # Threshold for considering a block as "mixed"
    # If secondary language is >= 20%, mark as mixed
    MIXED_THRESHOLD = 0.20
    
    # Minimum characters to make a confident detection
    MIN_CHARS_FOR_DETECTION = 10
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the language detection stage.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
    
    def execute(
        self,
        input_data: LanguageDetectionInput,
        logger: Logger
    ) -> LanguageDetectionOutput:
        """
        Execute the language detection stage.
        
        Args:
            input_data: Language detection input with normalized blocks.
            logger: Logger instance for progress tracking.
            
        Returns:
            LanguageDetectionOutput with language-annotated blocks.
        """
        start_time = time.time()
        
        blocks = input_data.blocks
        language_blocks: List[LanguageBlock] = []
        
        # Track language counts for distribution
        language_counts = {
            Language.AR: 0,
            Language.EN: 0,
            Language.MIXED: 0,
        }
        total_chars = 0
        arabic_chars_total = 0
        english_chars_total = 0
        
        for block in blocks:
            text = block.text
            
            # Detect language for this block
            language, arabic_ratio, english_ratio = self._detect_block_language(text)
            
            # Track character counts for document-level distribution
            char_count = len(text)
            total_chars += char_count
            arabic_chars_total += int(char_count * arabic_ratio)
            english_chars_total += int(char_count * english_ratio)
            
            language_counts[language] += 1
            
            # Create language block
            lang_block = LanguageBlock(
                block_id=block.block_id,
                block_type=block.block_type,
                normalized_text=block.text,
                original_text=block.original_text,
                page_number=block.page_number,
                heading_level=block.heading_level,
                source_offset=block.source_offset,
                language=language
            )
            language_blocks.append(lang_block)
        
        # Calculate document-level language distribution
        language_distribution = self._calculate_distribution(
            total_chars, arabic_chars_total, english_chars_total
        )
        
        # Determine primary language
        primary_language = self._determine_primary_language(language_distribution)
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log completion
        logger.info(
            f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
            f"{self.STAGE_NAME} ✓ ({duration_ms}ms)"
        )
        
        # Format distribution for logging
        dist_str = ", ".join(
            f"{lang}: {pct:.0%}"
            for lang, pct in language_distribution.items()
            if pct > 0
        )
        logger.info(f"  → Distribution: {dist_str}")
        logger.info(f"  → Primary language: {primary_language.value}")
        
        # Log low confidence warning
        low_confidence_blocks = sum(
            1 for block in language_blocks
            if len(block.normalized_text) < self.MIN_CHARS_FOR_DETECTION
        )
        if low_confidence_blocks > 0:
            logger.warning(
                f"  → {low_confidence_blocks} blocks with low-confidence detection "
                f"(< {self.MIN_CHARS_FOR_DETECTION} chars)"
            )
        
        return LanguageDetectionOutput(
            document_id=input_data.document_id,
            stored_path=input_data.stored_path,
            format=input_data.format,
            blocks=language_blocks,
            language_distribution=language_distribution,
            primary_language=primary_language
        )
    
    def _detect_block_language(self, text: str) -> tuple[Language, float, float]:
        """
        Detect language for a single block of text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Tuple of (Language, arabic_ratio, english_ratio).
        """
        if not text or not text.strip():
            return Language.EN, 0.0, 0.0  # Default to English for empty
        
        arabic_count = 0
        english_count = 0
        other_count = 0
        
        for char in text:
            if self._is_arabic_char(char):
                arabic_count += 1
            elif self._is_english_char(char):
                english_count += 1
            elif not char.isspace() and not self._is_punctuation(char):
                other_count += 1
        
        total_letters = arabic_count + english_count + other_count
        
        if total_letters == 0:
            return Language.EN, 0.0, 0.0  # Default for punctuation-only
        
        arabic_ratio = arabic_count / total_letters
        english_ratio = english_count / total_letters
        
        # Determine language based on ratios
        if arabic_ratio >= (1 - self.MIXED_THRESHOLD) and english_ratio < self.MIXED_THRESHOLD:
            return Language.AR, arabic_ratio, english_ratio
        elif english_ratio >= (1 - self.MIXED_THRESHOLD) and arabic_ratio < self.MIXED_THRESHOLD:
            return Language.EN, arabic_ratio, english_ratio
        elif arabic_ratio >= self.MIXED_THRESHOLD and english_ratio >= self.MIXED_THRESHOLD:
            return Language.MIXED, arabic_ratio, english_ratio
        elif arabic_ratio > english_ratio:
            return Language.AR, arabic_ratio, english_ratio
        else:
            return Language.EN, arabic_ratio, english_ratio
    
    def _is_arabic_char(self, char: str) -> bool:
        """
        Check if a character is Arabic.
        
        Args:
            char: Character to check.
            
        Returns:
            True if character is Arabic.
        """
        code_point = ord(char)
        
        # Arabic Unicode ranges
        return (
            0x0600 <= code_point <= 0x06FF or   # Arabic
            0x0750 <= code_point <= 0x077F or   # Arabic Supplement
            0x08A0 <= code_point <= 0x08FF or   # Arabic Extended-A
            0xFB50 <= code_point <= 0xFDFF or   # Arabic Presentation Forms-A
            0xFE70 <= code_point <= 0xFEFF      # Arabic Presentation Forms-B
        )
    
    def _is_english_char(self, char: str) -> bool:
        """
        Check if a character is English (Latin alphabet).
        
        Args:
            char: Character to check.
            
        Returns:
            True if character is English.
        """
        code_point = ord(char)
        
        # Basic Latin letters
        return (
            0x0041 <= code_point <= 0x005A or   # A-Z
            0x0061 <= code_point <= 0x007A      # a-z
        )
    
    def _is_punctuation(self, char: str) -> bool:
        """
        Check if a character is punctuation (to be excluded from ratio).
        
        Args:
            char: Character to check.
            
        Returns:
            True if character is punctuation.
        """
        code_point = ord(char)
        
        # Common punctuation ranges
        return (
            0x0020 <= code_point <= 0x002F or   # Space and basic punctuation
            0x003A <= code_point <= 0x0040 or   # : ; < = > ? @
            0x005B <= code_point <= 0x0060 or   # [ \ ] ^ _ `
            0x007B <= code_point <= 0x007E or   # { | } ~
            0x00A0 <= code_point <= 0x00BF or   # Latin punctuation
            0x060C <= code_point <= 0x060D or   # Arabic comma, date separator
            0x061B == code_point or             # Arabic semicolon
            0x061F == code_point or             # Arabic question mark
            0x0640 == code_point or             # Arabic tatweel
            char in '.,;:!?()[]{}"\'-/\\@#$%^&*+=<>~`|'
        )
    
    def _calculate_distribution(
        self,
        total_chars: int,
        arabic_chars: int,
        english_chars: int
    ) -> Dict[str, float]:
        """
        Calculate document-level language distribution.
        
        Args:
            total_chars: Total character count.
            arabic_chars: Arabic character count.
            english_chars: English character count.
            
        Returns:
            Dictionary with language distribution percentages.
        """
        if total_chars == 0:
            return {"ar": 0.0, "en": 1.0, "mixed": 0.0}
        
        arabic_pct = arabic_chars / total_chars
        english_pct = english_chars / total_chars
        other_pct = 1.0 - arabic_pct - english_pct
        
        return {
            "ar": round(arabic_pct, 3),
            "en": round(english_pct, 3),
            "other": round(max(0, other_pct), 3)
        }
    
    def _determine_primary_language(
        self,
        distribution: Dict[str, float]
    ) -> Language:
        """
        Determine the primary language of the document.
        
        Args:
            distribution: Language distribution dictionary.
            
        Returns:
            Primary Language enum value.
        """
        arabic_pct = distribution.get("ar", 0)
        english_pct = distribution.get("en", 0)
        
        # If both languages have significant presence, mark as mixed
        if arabic_pct >= self.MIXED_THRESHOLD and english_pct >= self.MIXED_THRESHOLD:
            return Language.MIXED
        
        # Otherwise, return the dominant language
        if arabic_pct > english_pct:
            return Language.AR
        else:
            return Language.EN