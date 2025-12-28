"""
Normalizers package for the Multilingual RAG Ingestion Pipeline.

Provides text normalization utilities:
- Unicode normalization (NFC, invisible characters)
- Arabic-specific normalization (Alef, Tashkeel, Tatweel)
- Whitespace normalization
"""

from .base import (
    BaseNormalizer,
    NormalizerChain,
    NormalizationResult,
)

from .unicode_normalizer import (
    UnicodeNormalizer,
    EncodingFixer,
)

from .arabic_normalizer import (
    ArabicNormalizer,
    ArabicLightStemmer,
)

from .whitespace_normalizer import (
    WhitespaceNormalizer,
    LineNormalizer,
    TextCleaner,
)


def create_default_normalizer_chain(
    include_arabic: bool = True,
    arabic_normalize_alef: bool = True,
    arabic_remove_tashkeel: bool = True,
    arabic_remove_tatweel: bool = True
) -> NormalizerChain:
    """
    Create a normalizer chain with default settings.
    
    The default chain includes:
    1. Unicode normalization (NFC, remove invisible chars)
    2. Arabic normalization (if enabled)
    3. Whitespace normalization
    
    Args:
        include_arabic: Whether to include Arabic normalizer.
        arabic_normalize_alef: Normalize Arabic Alef variants.
        arabic_remove_tashkeel: Remove Arabic diacritics.
        arabic_remove_tatweel: Remove Arabic Tatweel.
        
    Returns:
        Configured NormalizerChain.
    """
    chain = NormalizerChain()
    
    # Unicode normalization first
    chain.add(UnicodeNormalizer(
        apply_nfc=True,
        remove_zero_width=True,
        remove_control_chars=True,
        fix_encoding=True
    ))
    
    # Arabic normalization
    if include_arabic:
        chain.add(ArabicNormalizer(
            normalize_alef=arabic_normalize_alef,
            remove_tashkeel=arabic_remove_tashkeel,
            remove_tatweel=arabic_remove_tatweel,
            normalize_yeh=False,
            normalize_heh=False,
        ))
    
    # Whitespace normalization last
    chain.add(WhitespaceNormalizer(
        collapse_spaces=True,
        normalize_line_endings=True,
        collapse_newlines=False,
        trim=True,
        normalize_special_whitespace=True,
        preserve_paragraph_breaks=True
    ))
    
    return chain


def create_minimal_normalizer_chain() -> NormalizerChain:
    """
    Create a minimal normalizer chain.
    
    Only includes essential normalizations:
    - Unicode NFC
    - Basic whitespace cleanup
    
    Returns:
        Minimal NormalizerChain.
    """
    chain = NormalizerChain()
    
    chain.add(UnicodeNormalizer(
        apply_nfc=True,
        remove_zero_width=True,
        remove_control_chars=True,
        fix_encoding=False
    ))
    
    chain.add(WhitespaceNormalizer(
        collapse_spaces=True,
        normalize_line_endings=True,
        collapse_newlines=False,
        trim=True,
        normalize_special_whitespace=True
    ))
    
    return chain


def create_aggressive_normalizer_chain() -> NormalizerChain:
    """
    Create an aggressive normalizer chain.
    
    Includes all normalizations for maximum consistency:
    - Full Unicode normalization
    - Full Arabic normalization (including Yeh/Heh)
    - Aggressive whitespace normalization
    - Text cleaning
    
    Warning: May lose some semantic meaning.
    
    Returns:
        Aggressive NormalizerChain.
    """
    chain = NormalizerChain()
    
    # Text cleaner first
    chain.add(TextCleaner(
        strip=True,
        collapse_whitespace=True,
        remove_null_bytes=True,
        normalize_quotes=True,
        normalize_dashes=True
    ))
    
    # Unicode normalization
    chain.add(UnicodeNormalizer(
        apply_nfc=True,
        remove_zero_width=True,
        remove_control_chars=True,
        fix_encoding=True
    ))
    
    # Full Arabic normalization
    chain.add(ArabicNormalizer(
        normalize_alef=True,
        remove_tashkeel=True,
        remove_tatweel=True,
        normalize_yeh=True,
        normalize_heh=True,
        normalize_punctuation=True,
        normalize_numerals=True
    ))
    
    # Aggressive whitespace
    chain.add(WhitespaceNormalizer(
        collapse_spaces=True,
        normalize_line_endings=True,
        collapse_newlines=True,
        max_consecutive_newlines=1,
        trim=True,
        normalize_special_whitespace=True,
        preserve_paragraph_breaks=False
    ))
    
    return chain


def normalize_text(
    text: str,
    chain: NormalizerChain = None
) -> str:
    """
    Normalize text using default or provided chain.
    
    Args:
        text: Text to normalize.
        chain: Optional normalizer chain (uses default if None).
        
    Returns:
        Normalized text.
    """
    if chain is None:
        chain = create_default_normalizer_chain()
    
    return chain.normalize(text)


__all__ = [
    # Base
    "BaseNormalizer",
    "NormalizerChain",
    "NormalizationResult",
    # Unicode
    "UnicodeNormalizer",
    "EncodingFixer",
    # Arabic
    "ArabicNormalizer",
    "ArabicLightStemmer",
    # Whitespace
    "WhitespaceNormalizer",
    "LineNormalizer",
    "TextCleaner",
    # Factory functions
    "create_default_normalizer_chain",
    "create_minimal_normalizer_chain",
    "create_aggressive_normalizer_chain",
    "normalize_text",
]