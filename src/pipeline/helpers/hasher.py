"""
Hashing utilities for the Multilingual RAG Ingestion Pipeline.

This module provides functions for computing SHA256 hashes
of files and text content for deduplication and verification.
"""

import hashlib
from pathlib import Path
from typing import Union, BinaryIO


# Default buffer size for reading files (64KB)
DEFAULT_BUFFER_SIZE = 65536


def hash_bytes(data: bytes) -> str:
    """
    Compute SHA256 hash of bytes.
    
    Args:
        data: Bytes to hash.
        
    Returns:
        Lowercase hexadecimal hash string.
    """
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str, encoding: str = "utf-8") -> str:
    """
    Compute SHA256 hash of text.
    
    Args:
        text: Text string to hash.
        encoding: Text encoding (default: utf-8).
        
    Returns:
        Lowercase hexadecimal hash string.
    """
    return hash_bytes(text.encode(encoding))


def hash_file(
    file_path: Union[str, Path],
    buffer_size: int = DEFAULT_BUFFER_SIZE
) -> str:
    """
    Compute SHA256 hash of a file.
    
    Reads the file in chunks to handle large files efficiently.
    
    Args:
        file_path: Path to the file.
        buffer_size: Size of chunks to read (default: 64KB).
        
    Returns:
        Lowercase hexadecimal hash string.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        PermissionError: If the file can't be read.
        IsADirectoryError: If the path is a directory.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {file_path}")
    
    hasher = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            hasher.update(data)
    
    return hasher.hexdigest()


def hash_file_object(
    file_obj: BinaryIO,
    buffer_size: int = DEFAULT_BUFFER_SIZE
) -> str:
    """
    Compute SHA256 hash from a file-like object.
    
    The file object should be opened in binary mode.
    The file position is not reset after hashing.
    
    Args:
        file_obj: Binary file-like object.
        buffer_size: Size of chunks to read (default: 64KB).
        
    Returns:
        Lowercase hexadecimal hash string.
    """
    hasher = hashlib.sha256()
    
    while True:
        data = file_obj.read(buffer_size)
        if not data:
            break
        hasher.update(data)
    
    return hasher.hexdigest()


def verify_file_hash(
    file_path: Union[str, Path],
    expected_hash: str,
    buffer_size: int = DEFAULT_BUFFER_SIZE
) -> bool:
    """
    Verify that a file matches an expected hash.
    
    Args:
        file_path: Path to the file.
        expected_hash: Expected SHA256 hash (hexadecimal).
        buffer_size: Size of chunks to read (default: 64KB).
        
    Returns:
        True if the file hash matches, False otherwise.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    actual_hash = hash_file(file_path, buffer_size)
    return actual_hash.lower() == expected_hash.lower()


def verify_text_hash(text: str, expected_hash: str, encoding: str = "utf-8") -> bool:
    """
    Verify that text matches an expected hash.
    
    Args:
        text: Text to verify.
        expected_hash: Expected SHA256 hash (hexadecimal).
        encoding: Text encoding (default: utf-8).
        
    Returns:
        True if the text hash matches, False otherwise.
    """
    actual_hash = hash_text(text, encoding)
    return actual_hash.lower() == expected_hash.lower()


def get_file_fingerprint(
    file_path: Union[str, Path],
    length: int = 12
) -> str:
    """
    Get a short fingerprint of a file for quick identification.
    
    This is a truncated hash, useful for display or quick comparisons
    but NOT suitable for security-critical operations.
    
    Args:
        file_path: Path to the file.
        length: Length of fingerprint (default: 12 characters).
        
    Returns:
        Short fingerprint string.
    """
    full_hash = hash_file(file_path)
    return full_hash[:length]


def get_text_fingerprint(text: str, length: int = 12, encoding: str = "utf-8") -> str:
    """
    Get a short fingerprint of text for quick identification.
    
    Args:
        text: Text to fingerprint.
        length: Length of fingerprint (default: 12 characters).
        encoding: Text encoding (default: utf-8).
        
    Returns:
        Short fingerprint string.
    """
    full_hash = hash_text(text, encoding)
    return full_hash[:length]


def format_hash(hash_value: str, prefix: str = "sha256:") -> str:
    """
    Format a hash with a prefix for storage/display.
    
    Args:
        hash_value: The hash string.
        prefix: Prefix to add (default: "sha256:").
        
    Returns:
        Formatted hash string (e.g., "sha256:abc123...").
    """
    return f"{prefix}{hash_value}"


def parse_hash(formatted_hash: str) -> tuple:
    """
    Parse a formatted hash into algorithm and value.
    
    Args:
        formatted_hash: Hash string with prefix (e.g., "sha256:abc123").
        
    Returns:
        Tuple of (algorithm, hash_value).
        If no prefix, returns ("unknown", hash_value).
    """
    if ":" in formatted_hash:
        parts = formatted_hash.split(":", 1)
        return (parts[0], parts[1])
    return ("unknown", formatted_hash)


def is_valid_sha256(hash_value: str) -> bool:
    """
    Check if a string is a valid SHA256 hash.
    
    Args:
        hash_value: String to validate.
        
    Returns:
        True if valid SHA256 hex string (64 characters), False otherwise.
    """
    if not hash_value:
        return False
    
    # Remove prefix if present
    if ":" in hash_value:
        _, hash_value = parse_hash(hash_value)
    
    # SHA256 produces 64 hex characters
    if len(hash_value) != 64:
        return False
    
    # Check if all characters are valid hex
    try:
        int(hash_value, 16)
        return True
    except ValueError:
        return False


class FileHasher:
    """
    Class-based file hasher with caching support.
    
    Caches computed hashes to avoid re-computing for the same file.
    Cache is based on file path (not content), so it should be
    cleared if files might have changed.
    """
    
    def __init__(self, buffer_size: int = DEFAULT_BUFFER_SIZE):
        """
        Initialize the file hasher.
        
        Args:
            buffer_size: Size of chunks to read (default: 64KB).
        """
        self.buffer_size = buffer_size
        self._cache: dict = {}
    
    def hash(self, file_path: Union[str, Path]) -> str:
        """
        Compute or retrieve cached hash for a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            SHA256 hash string.
        """
        file_path = Path(file_path)
        cache_key = str(file_path.absolute())
        
        if cache_key not in self._cache:
            self._cache[cache_key] = hash_file(file_path, self.buffer_size)
        
        return self._cache[cache_key]
    
    def verify(self, file_path: Union[str, Path], expected_hash: str) -> bool:
        """
        Verify a file against an expected hash.
        
        Args:
            file_path: Path to the file.
            expected_hash: Expected SHA256 hash.
            
        Returns:
            True if hash matches, False otherwise.
        """
        actual_hash = self.hash(file_path)
        return actual_hash.lower() == expected_hash.lower()
    
    def fingerprint(self, file_path: Union[str, Path], length: int = 12) -> str:
        """
        Get a short fingerprint for a file.
        
        Args:
            file_path: Path to the file.
            length: Fingerprint length.
            
        Returns:
            Short fingerprint string.
        """
        full_hash = self.hash(file_path)
        return full_hash[:length]
    
    def is_duplicate(self, file_path: Union[str, Path], known_hashes: set) -> bool:
        """
        Check if a file is a duplicate based on known hashes.
        
        Args:
            file_path: Path to the file.
            known_hashes: Set of known hash values.
            
        Returns:
            True if file hash is in known_hashes, False otherwise.
        """
        file_hash = self.hash(file_path)
        return file_hash in known_hashes
    
    @property
    def cache_size(self) -> int:
        """Get the number of cached hashes."""
        return len(self._cache)
    
    def clear_cache(self) -> None:
        """Clear the hash cache."""
        self._cache = {}
    
    def remove_from_cache(self, file_path: Union[str, Path]) -> bool:
        """
        Remove a specific file from the cache.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            True if removed, False if not in cache.
        """
        file_path = Path(file_path)
        cache_key = str(file_path.absolute())
        
        if cache_key in self._cache:
            del self._cache[cache_key]
            return True
        return False