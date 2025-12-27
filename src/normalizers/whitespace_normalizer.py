"""
Whitespace normalizer for the Multilingual RAG Ingestion Pipeline.

Handles whitespace normalization including:
- Collapsing multiple spaces
- Normalizing line endings
- Trimming leading/trailing whitespace
- Handling special whitespace characters
"""

import re
from typing import Set

from .base import BaseNormalizer


class WhitespaceNormalizer(BaseNormalizer):
    """
    Normalizes whitespace in text.
    
    Operations:
    - Collapse multiple spaces into single space
    - Normalize line endings (CRLF -> LF)
    - Collapse multiple newlines (optional)
    - Trim leading/trailing whitespace
    - Convert special whitespace to regular space
    """
    
    NAME = "whitespace"
    
    # Special whitespace characters to normalize to regular space
    SPECIAL_WHITESPACE: Set[str] = {
        '\u00a0',  # Non-breaking space
        '\u1680',  # Ogham space mark
        '\u2000',  # En quad
        '\u2001',  # Em quad
        '\u2002',  # En space
        '\u2003',  # Em space
        '\u2004',  # Three-per-em space
        '\u2005',  # Four-per-em space
        '\u2006',  # Six-per-em space
        '\u2007',  # Figure space
        '\u2008',  # Punctuation space
        '\u2009',  # Thin space
        '\u200a',  # Hair space
        '\u202f',  # Narrow no-break space
        '\u205f',  # Medium mathematical space
        '\u3000',  # Ideographic space
    }
    
    def __init__(
        self,
        enabled: bool = True,
        collapse_spaces: bool = True,
        normalize_line_endings: bool = True,
        collapse_newlines: bool = False,
        max_consecutive_newlines: int = 2,
        trim: bool = True,
        normalize_special_whitespace: bool = True,
        preserve_paragraph_breaks: bool = True
    ):
        """
        Initialize whitespace normalizer.
        
        Args:
            enabled: Whether normalizer is active.
            collapse_spaces: Collapse multiple spaces to single space.
            normalize_line_endings: Convert CRLF to LF.
            collapse_newlines: Collapse multiple consecutive newlines.
            max_consecutive_newlines: Max newlines to keep when collapsing.
            trim: Trim leading/trailing whitespace.
            normalize_special_whitespace: Convert special spaces to regular.
            preserve_paragraph_breaks: Keep double newlines for paragraphs.
        """
        super().__init__(enabled)
        self.collapse_spaces = collapse_spaces
        self.normalize_line_endings = normalize_line_endings
        self.collapse_newlines = collapse_newlines
        self.max_consecutive_newlines = max_consecutive_newlines
        self.trim = trim
        self.normalize_special_whitespace = normalize_special_whitespace
        self.preserve_paragraph_breaks = preserve_paragraph_breaks
    
    def normalize(self, text: str) -> str:
        """
        Apply whitespace normalization.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        if not text:
            return text
        
        result = text
        
        # Normalize special whitespace characters
        if self.normalize_special_whitespace:
            result = self._normalize_special_whitespace(result)
        
        # Normalize line endings (CRLF -> LF)
        if self.normalize_line_endings:
            result = result.replace('\r\n', '\n').replace('\r', '\n')
        
        # Collapse multiple newlines
        if self.collapse_newlines:
            result = self._collapse_newlines(result)
        
        # Collapse multiple spaces (but not newlines)
        if self.collapse_spaces:
            result = self._collapse_spaces(result)
        
        # Trim leading/trailing whitespace
        if self.trim:
            result = result.strip()
        
        return result
    
    def _normalize_special_whitespace(self, text: str) -> str:
        """
        Convert special whitespace characters to regular space.
        
        Args:
            text: Text to process.
            
        Returns:
            Processed text.
        """
        result = text
        
        for ws_char in self.SPECIAL_WHITESPACE:
            result = result.replace(ws_char, ' ')
        
        return result
    
    def _collapse_spaces(self, text: str) -> str:
        """
        Collapse multiple consecutive spaces (not newlines).
        
        Args:
            text: Text to process.
            
        Returns:
            Processed text.
        """
        # Replace multiple spaces with single space
        # But preserve newlines
        result = re.sub(r'[^\S\n]+', ' ', text)
        
        # Also clean up spaces around newlines
        result = re.sub(r' *\n *', '\n', result)
        
        return result
    
    def _collapse_newlines(self, text: str) -> str:
        """
        Collapse multiple consecutive newlines.
        
        Args:
            text: Text to process.
            
        Returns:
            Processed text.
        """
        if self.preserve_paragraph_breaks:
            # Keep at most max_consecutive_newlines
            max_nl = self.max_consecutive_newlines
            pattern = r'\n{' + str(max_nl + 1) + r',}'
            replacement = '\n' * max_nl
            result = re.sub(pattern, replacement, text)
        else:
            # Collapse all multiple newlines to single
            result = re.sub(r'\n+', '\n', text)
        
        return result


class LineNormalizer(BaseNormalizer):
    """
    Normalizes text on a line-by-line basis.
    
    Useful for cleaning up text where each line should be processed
    independently.
    """
    
    NAME = "line"
    
    def __init__(self, enabled: bool = True, strip_lines: bool = True, remove_empty_lines: bool = False, collapse_empty_lines: bool = True, max_empty_lines: int = 1):
        """
        Initialize line normalizer.
        
        Args:
            enabled: Whether normalizer is active.
            strip_lines: Strip whitespace from each line.
            remove_empty_lines: Remove all empty lines.
            collapse_empty_lines: Collapse consecutive empty lines.
            max_empty_lines: Max consecutive empty lines to keep.
        """
        super().__init__(enabled)
        self.strip_lines = strip_lines
        self.remove_empty_lines = remove_empty_lines
        self.collapse_empty_lines = collapse_empty_lines
        self.max_empty_lines = max_empty_lines
    
    def normalize(self, text: str) -> str:
        """
        Apply line normalization.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        if not text:
            return text
        
        lines = text.split('\n')
        
        # Strip lines
        if self.strip_lines:
            lines = [line.strip() for line in lines]
        
        # Handle empty lines
        if self.remove_empty_lines:
            lines = [line for line in lines if line]
        elif self.collapse_empty_lines:
            lines = self._collapse_empty_lines(lines)
        
        return '\n'.join(lines)
    
    def _collapse_empty_lines(self, lines: list) -> list:
        """
        Collapse consecutive empty lines.
        
        Args:
            lines: List of lines.
            
        Returns:
            List with collapsed empty lines.
        """
        result = []
        empty_count = 0
        
        for line in lines:
            if not line.strip():
                empty_count += 1
                if empty_count <= self.max_empty_lines:
                    result.append(line)
            else:
                empty_count = 0
                result.append(line)
        
        return result


class TextCleaner(BaseNormalizer):
    """
    General text cleaner combining common operations.
    
    A convenience normalizer that combines multiple cleaning operations
    that are commonly needed together.
    """
    
    NAME = "cleaner"
    
    def __init__(
        self,
        enabled: bool = True,
        strip: bool = True,
        collapse_whitespace: bool = True,
        remove_null_bytes: bool = True,
        normalize_quotes: bool = True,
        normalize_dashes: bool = True
    ):
        """
        Initialize text cleaner.
        
        Args:
            enabled: Whether normalizer is active.
            strip: Strip leading/trailing whitespace.
            collapse_whitespace: Collapse multiple spaces.
            remove_null_bytes: Remove null bytes.
            normalize_quotes: Normalize fancy quotes to simple quotes.
            normalize_dashes: Normalize fancy dashes to simple dashes.
        """
        super().__init__(enabled)
        self.strip = strip
        self.collapse_whitespace = collapse_whitespace
        self.remove_null_bytes = remove_null_bytes
        self.normalize_quotes = normalize_quotes
        self.normalize_dashes = normalize_dashes
        
        # Quote replacements
        self._quote_map = {
            '"': '"', '"': '"',  # Double quotes
            ''': "'", ''': "'",  # Single quotes
            '‚': "'", '„': '"',  # Low quotes
            '‹': "'", '›': "'",  # Angle quotes
            '«': '"', '»': '"',  # Guillemets
        }
        
        # Dash replacements
        self._dash_map = {
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '―': '-',  # Horizontal bar
            '‐': '-',  # Hyphen
            '‑': '-',  # Non-breaking hyphen
            '‒': '-',  # Figure dash
        }
    
    def normalize(self, text: str) -> str:
        """
        Apply text cleaning.
        
        Args:
            text: Text to clean.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return text
        
        result = text
        
        # Remove null bytes
        if self.remove_null_bytes:
            result = result.replace('\x00', '')
        
        # Normalize quotes
        if self.normalize_quotes:
            for fancy, simple in self._quote_map.items():
                result = result.replace(fancy, simple)
        
        # Normalize dashes
        if self.normalize_dashes:
            for fancy, simple in self._dash_map.items():
                result = result.replace(fancy, simple)
        
        # Collapse whitespace
        if self.collapse_whitespace:
            result = re.sub(r'[ \t]+', ' ', result)
        
        # Strip
        if self.strip:
            result = result.strip()
        
        return result