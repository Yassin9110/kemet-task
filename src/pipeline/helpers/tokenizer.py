"""
Tokenization utilities for the Multilingual RAG Ingestion Pipeline.

This module provides functions for counting and managing tokens,
essential for accurate chunk sizing. Uses tiktoken for tokenization
with fallback to character-based estimation.

Note: For production with Cohere, consider using Cohere's tokenizer
directly for exact token counts. This implementation provides a
good approximation suitable for the POC.
"""


from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import tiktoken

# Default encoding for multilingual text (cl100k_base works well for multilingual)
DEFAULT_ENCODING = "cl100k_base"

@dataclass
class TokenCount:
    """Result of token counting operation."""
    
    count: int # Number of tokens.
    
    encoding: Optional[str] = None # Encoding name if tiktoken was used.
    

class Tokenizer:
    """
    Token counting and text manipulation based on tokens.
    
    Uses tiktoken when available, falls back to character-based
    estimation otherwise.
    """
    
    def __init__(self, encoding_name: str = DEFAULT_ENCODING):
        """
        Initialize the tokenizer.
        
        Args:
            encoding_name: tiktoken encoding name (default: cl100k_base).
        """
        self.encoding_name = encoding_name
        self._encoder = None
        self._encoder = tiktoken.get_encoding(encoding_name)
            
     
    def count(self, text: str) -> TokenCount:
        """
        Count tokens in text.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            TokenCount with count.
        """
        if not text:
            return TokenCount(count=0)
        try:
            tokens = self._encoder.encode(text)
            return TokenCount(
                count=len(tokens),
                encoding=self.encoding_name
            )
        except Exception as e:
            raise RuntimeError("error during token counting", e)
            
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (simple interface).
        
        Args:
            text: Text to tokenize.
            
        Returns:
            Number of tokens.
        """
        return self.count(text).count
    
    def count_batch(self, texts: List[str]) -> List[TokenCount]:
        """
        Count tokens for multiple texts.
        
        Args:
            texts: List of texts to tokenize.
            
        Returns:
            List of TokenCount results.
        """
        return [self.count(text) for text in texts]
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts (simple interface).
        
        Args:
            texts: List of texts to tokenize.
            
        Returns:
            List of token counts.
        """
        return [self.count_tokens(text) for text in texts]
    
    def fits_in_limit(self, text: str, max_tokens: int) -> bool:
        """
        Check if text fits within a token limit.
        
        Args:
            text: Text to check.
            max_tokens: Maximum allowed tokens.
            
        Returns:
            True if text fits, False otherwise.
        """
        return self.count_tokens(text) <= max_tokens
    
    def truncate(self, text: str, max_tokens: int, suffix: str = "...") -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate.
            max_tokens: Maximum tokens.
            suffix: Suffix to add if truncated (default: "...").
            
        Returns:
            Truncated text.
        """
        if not text:
            return text
        
        if self.fits_in_limit(text, max_tokens):
            return text
        
        return self._truncate_exact(text, max_tokens, suffix)
    
    def _truncate_exact(self, text: str, max_tokens: int, suffix: str) -> str:
        """Truncate using exact tokenization."""
        tokens = self._encoder.encode(text)
        suffix_tokens = self._encoder.encode(suffix) if suffix else []
        
        available_tokens = max_tokens - len(suffix_tokens)
        if available_tokens <= 0:
            return suffix
        
        truncated_tokens = tokens[:available_tokens]
        truncated_text = self._encoder.decode(truncated_tokens)
        
        return truncated_text + suffix
    
    
    def split_by_tokens(self, text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """
        Split text into chunks of approximately N tokens.
        
        Args:
            text: Text to split.
            chunk_size: Target tokens per chunk.
            overlap: Token overlap between chunks.
            
        Returns:
            List of text chunks.
        """
        if not text:
            return []
        
        if self.count_tokens(text) <= chunk_size:
            return [text]
        
        
        return self._split_exact(text, chunk_size, overlap)
    
    def _split_exact(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split using exact tokenization."""
        tokens = self._encoder.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self._encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start, accounting for overlap
            start = end - overlap
            if start >= len(tokens):
                break
            if end >= len(tokens):
                break
        
        return chunks
        
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode.
            
        Returns:
            List of token IDs.
            
        Raises:
            RuntimeError: If tiktoken is not available.
        """
        try:
            return self._encoder.encode(text)
        except Exception:
            raise RuntimeError("encode() failed")
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs.
            
        Returns:
            Decoded text.
            
        Raises:
            RuntimeError: If tiktoken is not available.
        """
        try:
            return self._encoder.decode(tokens)
        except Exception:
            raise RuntimeError("decode() failed")


# Module-level default tokenizer instance
_default_tokenizer: Optional[Tokenizer] = None


def get_tokenizer(encoding_name: str = DEFAULT_ENCODING) -> Tokenizer:
    """
    Get or create the default tokenizer instance.
    
    Args:
        encoding_name: tiktoken encoding name.
        
    Returns:
        Tokenizer instance.
    """
    global _default_tokenizer
    
    if _default_tokenizer is None or _default_tokenizer.encoding_name != encoding_name:
        _default_tokenizer = Tokenizer(encoding_name)
    
    return _default_tokenizer


# Convenience functions using default tokenizer

def count_tokens(text: str) -> int:
    """
    Count tokens in text using default tokenizer.
    
    Args:
        text: Text to tokenize.
        
    Returns:
        Number of tokens.
    """
    return get_tokenizer().count_tokens(text)


def count_tokens_batch(texts: List[str]) -> List[int]:
    """
    Count tokens for multiple texts using default tokenizer.
    
    Args:
        texts: List of texts to tokenize.
        
    Returns:
        List of token counts.
    """
    return get_tokenizer().count_tokens_batch(texts)


def truncate_to_tokens(text: str, max_tokens: int, suffix: str = "...") -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate.
        max_tokens: Maximum tokens.
        suffix: Suffix to add if truncated.
        
    Returns:
        Truncated text.
    """
    return get_tokenizer().truncate(text, max_tokens, suffix)


def split_by_tokens(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Split text into chunks of approximately N tokens.
    
    Args:
        text: Text to split.
        chunk_size: Target tokens per chunk.
        overlap: Token overlap between chunks.
        
    Returns:
        List of text chunks.
    """
    return get_tokenizer().split_by_tokens(text, chunk_size, overlap)


def fits_in_context(text: str, max_tokens: int) -> bool:
    """
    Check if text fits within a token limit.
    
    Args:
        text: Text to check.
        max_tokens: Maximum allowed tokens.
        
    Returns:
        True if text fits, False otherwise.
    """
    return get_tokenizer().fits_in_limit(text, max_tokens)


def get_token_info(text: str) -> dict:
    """
    Get detailed token information for text.
    
    Args:
        text: Text to analyze.
        
    Returns:
        Dictionary with token info.
    """
    tokenizer = get_tokenizer()
    result = tokenizer.count(text)
    
    return {
        "text_length": len(text),
        "token_count": result.count,
        "encoding": result.encoding,
        "chars_per_token": len(text) / result.count if result.count > 0 else 0,
    }