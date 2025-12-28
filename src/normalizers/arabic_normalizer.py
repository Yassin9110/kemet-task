"""
Arabic normalizer for the Multilingual RAG Ingestion Pipeline.

Handles Arabic-specific text normalization including:
- Alef variants normalization
- Tashkeel (diacritics) removal
- Tatweel (kashida) removal
- Other Arabic character normalizations
"""

import re
from typing import Dict, Set, Optional

from .base import BaseNormalizer


class ArabicNormalizer(BaseNormalizer):
    """
    Normalizes Arabic text for consistent processing.
    
    Operations:
    - Normalize Alef variants (أ إ آ ا)
    - Remove Tashkeel (diacritics/harakat)
    - Remove Tatweel (kashida/stretching)
    - Normalize Yeh variants
    - Normalize Heh variants
    - Remove Arabic-specific punctuation marks (optional)
    """
    
    NAME = "arabic"
    
    # Alef variants to normalize
    ALEF_VARIANTS: Dict[str, str] = {
        'أ': 'ا',  # Alef with Hamza above
        'إ': 'ا',  # Alef with Hamza below
        'آ': 'ا',  # Alef with Madda
        'ٱ': 'ا',  # Alef Wasla
        'ٲ': 'ا',  # Alef with wavy Hamza above
        'ٳ': 'ا',  # Alef with wavy Hamza below
    }
    
    # Tashkeel (diacritics) characters
    TASHKEEL: Set[str] = {
        '\u064b',  # Fathatan
        '\u064c',  # Dammatan
        '\u064d',  # Kasratan
        '\u064e',  # Fatha
        '\u064f',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u0653',  # Maddah above
        '\u0654',  # Hamza above
        '\u0655',  # Hamza below
        '\u0656',  # Subscript Alef
        '\u0657',  # Inverted Damma
        '\u0658',  # Mark noon ghunna
        '\u0659',  # Zwarakay
        '\u065a',  # Vowel sign small v above
        '\u065b',  # Vowel sign inverted small v above
        '\u065c',  # Vowel sign dot below
        '\u065d',  # Reversed damma
        '\u065e',  # Fatha with two dots
        '\u065f',  # Wavy hamza below
        '\u0670',  # Superscript Alef
    }
    
    # Tatweel character
    TATWEEL = '\u0640'
    
    # Yeh variants
    YEH_VARIANTS: Dict[str, str] = {
        'ى': 'ي',  # Alef Maksura to Yeh
        'ۍ': 'ي',  # Yeh with tail
        'ې': 'ي',  # E with two dots
        'ے': 'ي',  # Yeh barree
        'ۓ': 'ي',  # Yeh barree with hamza
    }
    
    # Heh variants
    HEH_VARIANTS: Dict[str, str] = {
        'ة': 'ه',  # Teh Marbuta to Heh
        'ۀ': 'ه',  # Heh with Yeh above
        'ۂ': 'ه',  # Heh goal with hamza
        'ھ': 'ه',  # Heh Doachashmee
    }
    
    # Waw variants
    WAW_VARIANTS: Dict[str, str] = {
        'ؤ': 'و',  # Waw with Hamza
        'ۄ': 'و',  # Waw with ring
        'ۆ': 'و',  # Oe
        'ۇ': 'و',  # U
        'ۈ': 'و',  # Yu
        'ۉ': 'و',  # Kirghiz Yu
        'ۊ': 'و',  # Waw with two dots above
        'ۋ': 'و',  # Ve
    }
    
    # Qaf/Kaf variants
    QAF_KAF_VARIANTS: Dict[str, str] = {
        'ڨ': 'ق',  # Qaf with three dots above (Maghrebi)
        'ڧ': 'ف',  # Feh with three dots below
        'ک': 'ك',  # Keheh (Persian)
        'گ': 'ك',  # Gaf (Persian)
        'ڪ': 'ك',  # Swash Kaf
    }
    
    # Arabic punctuation that might need normalization
    ARABIC_PUNCTUATION: Dict[str, str] = {
        '،': ',',  # Arabic comma
        '؛': ';',  # Arabic semicolon
        '؟': '?',  # Arabic question mark
        '٪': '%',  # Arabic percent
        '٫': '.',  # Arabic decimal separator
        '٬': ',',  # Arabic thousands separator
    }
    
    # Arabic-Indic numerals to Western
    ARABIC_NUMERALS: Dict[str, str] = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
    }
    
    def __init__(
        self,
        enabled: bool = True,
        normalize_alef: bool = True,
        remove_tashkeel: bool = True,
        remove_tatweel: bool = True,
        normalize_yeh: bool = False,
        normalize_heh: bool = False,
        normalize_waw: bool = False,
        normalize_qaf_kaf: bool = False,
        normalize_punctuation: bool = False,
        normalize_numerals: bool = False
    ):
        """
        Initialize Arabic normalizer.
        
        Args:
            enabled: Whether normalizer is active.
            normalize_alef: Normalize Alef variants to bare Alef.
            remove_tashkeel: Remove diacritics (harakat).
            remove_tatweel: Remove kashida/stretching character.
            normalize_yeh: Normalize Yeh variants (may lose meaning).
            normalize_heh: Normalize Heh/Teh Marbuta variants (may lose meaning).
            normalize_waw: Normalize Waw variants.
            normalize_qaf_kaf: Normalize Qaf/Kaf regional variants.
            normalize_punctuation: Convert Arabic punctuation to Western.
            normalize_numerals: Convert Arabic-Indic numerals to Western.
        """
        super().__init__(enabled)
        self.normalize_alef = normalize_alef
        self.remove_tashkeel = remove_tashkeel
        self.remove_tatweel = remove_tatweel
        self.normalize_yeh = normalize_yeh
        self.normalize_heh = normalize_heh
        self.normalize_waw = normalize_waw
        self.normalize_qaf_kaf = normalize_qaf_kaf
        self.normalize_punctuation = normalize_punctuation
        self.normalize_numerals = normalize_numerals
        
        # Build combined replacement map
        self._build_replacement_map()
    
    def _build_replacement_map(self) -> None:
        """Build the combined character replacement map."""
        self._replacements: Dict[str, str] = {}
        
        if self.normalize_alef:
            self._replacements.update(self.ALEF_VARIANTS)
        
        if self.normalize_yeh:
            self._replacements.update(self.YEH_VARIANTS)
        
        if self.normalize_heh:
            self._replacements.update(self.HEH_VARIANTS)
        
        if self.normalize_waw:
            self._replacements.update(self.WAW_VARIANTS)
        
        if self.normalize_qaf_kaf:
            self._replacements.update(self.QAF_KAF_VARIANTS)
        
        if self.normalize_punctuation:
            self._replacements.update(self.ARABIC_PUNCTUATION)
        
        if self.normalize_numerals:
            self._replacements.update(self.ARABIC_NUMERALS)
    
    def normalize(self, text: str) -> str:
        """
        Apply Arabic normalization.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        if not text:
            return text
        
        result = text
        
        # Remove tashkeel (diacritics)
        if self.remove_tashkeel:
            result = self._remove_tashkeel(result)
        
        # Remove tatweel
        if self.remove_tatweel:
            result = result.replace(self.TATWEEL, '')
        
        # Apply character replacements
        if self._replacements:
            result = self._apply_replacements(result)
        
        return result
    
    def _remove_tashkeel(self, text: str) -> str:
        """
        Remove tashkeel (diacritics) from text.
        
        Args:
            text: Text to clean.
            
        Returns:
            Text without diacritics.
        """
        return ''.join(
            char for char in text
            if char not in self.TASHKEEL
        )
    
    def _apply_replacements(self, text: str) -> str:
        """
        Apply character replacements.
        
        Args:
            text: Text to process.
            
        Returns:
            Processed text.
        """
        result = []
        
        for char in text:
            if char in self._replacements:
                result.append(self._replacements[char])
            else:
                result.append(char)
        
        return ''.join(result)
    
    @staticmethod
    def is_arabic(text: str) -> bool:
        """
        Check if text contains Arabic characters.
        
        Args:
            text: Text to check.
            
        Returns:
            True if text contains Arabic characters.
        """
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        return bool(arabic_pattern.search(text))
    
    @staticmethod
    def get_arabic_ratio(text: str) -> float:
        """
        Calculate the ratio of Arabic characters in text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Ratio of Arabic characters (0.0 to 1.0).
        """
        if not text:
            return 0.0
        
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        arabic_chars = len(arabic_pattern.findall(text))
        
        # Count non-whitespace characters
        total_chars = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        if total_chars == 0:
            return 0.0
        
        return arabic_chars / total_chars


class ArabicLightStemmer(BaseNormalizer):
    """
    Light stemming for Arabic text.
    
    Removes common prefixes and suffixes without full morphological analysis.
    This is optional and may affect retrieval - use with caution.
    """
    
    NAME = "arabic_stemmer"
    
    # Common prefixes
    PREFIXES = ['ال', 'وال', 'بال', 'كال', 'فال', 'لل']
    
    # Common suffixes
    SUFFIXES = ['ها', 'هم', 'هن', 'ك', 'كم', 'كن', 'نا', 'ي', 'ه', 'ون', 'ين', 'ات', 'ة', 'ا']
    
    def __init__(self, enabled: bool = False):
        """
        Initialize Arabic light stemmer.
        
        Note: Disabled by default as it may affect retrieval quality.
        
        Args:
            enabled: Whether normalizer is active.
        """
        super().__init__(enabled)
        
        # Sort by length (longest first) for proper matching
        self._prefixes = sorted(self.PREFIXES, key=len, reverse=True)
        self._suffixes = sorted(self.SUFFIXES, key=len, reverse=True)
    
    def normalize(self, text: str) -> str:
        """
        Apply light stemming to Arabic words.
        
        Args:
            text: Text to stem.
            
        Returns:
            Stemmed text.
        """
        if not text:
            return text
        
        words = text.split()
        stemmed_words = [self._stem_word(word) for word in words]
        
        return ' '.join(stemmed_words)
    
    def _stem_word(self, word: str) -> str:
        """
        Apply light stemming to a single word.
        
        Args:
            word: Word to stem.
            
        Returns:
            Stemmed word.
        """
        # Only stem if word is long enough
        if len(word) < 4:
            return word
        
        result = word
        
        # Remove prefix
        for prefix in self._prefixes:
            if result.startswith(prefix) and len(result) - len(prefix) >= 2:
                result = result[len(prefix):]
                break
        
        # Remove suffix
        for suffix in self._suffixes:
            if result.endswith(suffix) and len(result) - len(suffix) >= 2:
                result = result[:-len(suffix)]
                break
        
        return result