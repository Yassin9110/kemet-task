"""
Unicode normalizer for the Multilingual RAG Ingestion Pipeline.

Handles Unicode normalization (NFC) and removal of invisible/control characters.
"""

import unicodedata
import re
from typing import Set

from .base import BaseNormalizer


class UnicodeNormalizer(BaseNormalizer):
    """
    Normalizes Unicode text using NFC form and removes problematic characters.
    
    Operations:
    - Apply NFC (Canonical Decomposition, followed by Canonical Composition)
    - Remove zero-width characters
    - Remove control characters (except newlines and tabs)
    - Fix common encoding issues
    """
    
    NAME = "unicode"
    
    # Zero-width and invisible characters to remove
    ZERO_WIDTH_CHARS: Set[str] = {
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\u200e',  # Left-to-right mark
        '\u200f',  # Right-to-left mark
        '\u202a',  # Left-to-right embedding
        '\u202b',  # Right-to-left embedding
        '\u202c',  # Pop directional formatting
        '\u202d',  # Left-to-right override
        '\u202e',  # Right-to-left override
        '\u2060',  # Word joiner
        '\u2061',  # Function application
        '\u2062',  # Invisible times
        '\u2063',  # Invisible separator
        '\u2064',  # Invisible plus
        '\ufeff',  # Byte order mark (BOM)
        '\ufffe',  # Invalid Unicode
        '\uffff',  # Invalid Unicode
    }
    
    # Control character categories to remove
    CONTROL_CATEGORIES: Set[str] = {'Cc', 'Cf'}
    
    # Characters to preserve (even if control)
    PRESERVE_CHARS: Set[str] = {
        '\n',  # Newline
        '\r',  # Carriage return
        '\t',  # Tab
    }
    
    def __init__(self, enabled: bool = True, apply_nfc: bool = True, remove_zero_width: bool = True, remove_control_chars: bool = True, fix_encoding: bool = True):
        """
        Initialize Unicode normalizer.
        
        Args:
            enabled: Whether normalizer is active.
            apply_nfc: Whether to apply NFC normalization.
            remove_zero_width: Whether to remove zero-width characters.
            remove_control_chars: Whether to remove control characters.
            fix_encoding: Whether to attempt encoding fixes.
        """
        super().__init__(enabled)
        self.apply_nfc = apply_nfc
        self.remove_zero_width = remove_zero_width
        self.remove_control_chars = remove_control_chars
        self.fix_encoding = fix_encoding
    
    def normalize(self, text: str) -> str:
        """
        Apply Unicode normalization.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        if not text:
            return text
        
        result = text
        
        # Fix encoding issues first
        if self.fix_encoding:
            result = self._fix_encoding(result)
        
        # Apply NFC normalization
        if self.apply_nfc:
            result = unicodedata.normalize('NFC', result)
        
        # Remove zero-width characters
        if self.remove_zero_width:
            result = self._remove_zero_width(result)
        
        # Remove control characters
        if self.remove_control_chars:
            result = self._remove_control_chars(result)
        
        return result
    
    def _fix_encoding(self, text: str) -> str:
        """
        Attempt to fix common encoding issues.
        
        Args:
            text: Text to fix.
            
        Returns:
            Fixed text.
        """
        # Common mojibake patterns and fixes
        replacements = [
            ('â€™', "'"),  # Right single quote
            ('â€"', "–"),  # En dash
            ('â€"', "—"),  # Em dash
            ('â€œ', '"'),  # Left double quote
            ('â€', '"'),   # Right double quote
            ('â€¦', '…'),  # Ellipsis
            ('Ã©', 'é'),   # e with acute
            ('Ã¨', 'è'),   # e with grave
            ('Ã ', 'à'),   # a with grave
            ('Ã¢', 'â'),   # a with circumflex
            ('Ã®', 'î'),   # i with circumflex
            ('Ã´', 'ô'),   # o with circumflex
            ('Ã»', 'û'),   # u with circumflex
            ('Ã§', 'ç'),   # c with cedilla
        ]
        
        result = text
        for wrong, right in replacements:
            result = result.replace(wrong, right)
        
        return result
    
    def _remove_zero_width(self, text: str) -> str:
        """
        Remove zero-width and invisible characters.
        
        Args:
            text: Text to clean.
            
        Returns:
            Cleaned text.
        """
        return ''.join(
            char for char in text
            if char not in self.ZERO_WIDTH_CHARS
        )
    
    def _remove_control_chars(self, text: str) -> str:
        """
        Remove control characters except preserved ones.
        
        Args:
            text: Text to clean.
            
        Returns:
            Cleaned text.
        """
        result = []
        
        for char in text:
            # Always preserve certain characters
            if char in self.PRESERVE_CHARS:
                result.append(char)
                continue
            
            # Check character category
            category = unicodedata.category(char)
            
            # Skip control and format characters
            if category in self.CONTROL_CATEGORIES:
                continue
            
            result.append(char)
        
        return ''.join(result)


class EncodingFixer(BaseNormalizer):
    """
    Specialized normalizer for fixing encoding issues.
    
    Attempts to detect and fix mojibake and other encoding problems.
    """
    
    NAME = "encoding_fixer"
    
    def __init__(self, enabled: bool = True):
        """
        Initialize encoding fixer.
        
        Args:
            enabled: Whether normalizer is active.
        """
        super().__init__(enabled)
        
        # Build replacement pattern
        self._replacements = self._build_replacements()
    
    def _build_replacements(self) -> dict:
        """Build dictionary of common encoding fixes."""
        return {
            # UTF-8 interpreted as Latin-1
            'Ã¡': 'á', 'Ã ': 'à', 'Ã¢': 'â', 'Ã£': 'ã', 'Ã¤': 'ä', 'Ã¥': 'å',
            'Ã¦': 'æ', 'Ã§': 'ç', 'Ã¨': 'è', 'Ã©': 'é', 'Ãª': 'ê', 'Ã«': 'ë',
            'Ã¬': 'ì', 'Ã­': 'í', 'Ã®': 'î', 'Ã¯': 'ï', 'Ã°': 'ð', 'Ã±': 'ñ',
            'Ã²': 'ò', 'Ã³': 'ó', 'Ã´': 'ô', 'Ãµ': 'õ', 'Ã¶': 'ö', 'Ã¸': 'ø',
            'Ã¹': 'ù', 'Ãº': 'ú', 'Ã»': 'û', 'Ã¼': 'ü', 'Ã½': 'ý', 'Ã¿': 'ÿ',
            'Ã€': 'À', 'Ã': 'Á', 'Ã‚': 'Â', 'Ãƒ': 'Ã', 'Ã„': 'Ä', 'Ã…': 'Å',
            # Smart quotes and dashes
            'â€™': "'", 'â€˜': "'", 'â€œ': '"', 'â€': '"',
            'â€"': '—', 'â€"': '–', 'â€¦': '…',
            # Other common issues
            'Â ': ' ', 'Â': '',  # Non-breaking space artifacts
        }
    
    def normalize(self, text: str) -> str:
        """
        Fix encoding issues in text.
        
        Args:
            text: Text to fix.
            
        Returns:
            Fixed text.
        """
        if not text:
            return text
        
        result = text
        
        for wrong, right in self._replacements.items():
            result = result.replace(wrong, right)
        
        return result