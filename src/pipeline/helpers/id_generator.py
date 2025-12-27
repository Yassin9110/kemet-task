"""
ID generation utilities for the Multilingual RAG Ingestion Pipeline.

This module provides functions for generating unique identifiers
using UUID v4 for various pipeline entities.
"""

import uuid
import re
from typing import Optional


# Regex pattern for validating UUID v4
UUID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE
)

# Regex pattern for validating prefixed IDs
PREFIXED_ID_PATTERN = re.compile(
    r'^[a-z]+-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE
)


def generate_id() -> str:
    """
    Generate a new UUID v4 string.
    
    Returns:
        A lowercase UUID v4 string (e.g., "550e8400-e29b-41d4-a716-446655440000").
    """
    return str(uuid.uuid4())


def generate_prefixed_id(prefix: str) -> str:
    """
    Generate a UUID v4 with a prefix.
    
    Args:
        prefix: Prefix to prepend (e.g., "doc", "chunk").
        
    Returns:
        Prefixed UUID string (e.g., "doc-550e8400-e29b-41d4-a716-446655440000").
    """
    return f"{prefix}-{generate_id()}"


def generate_document_id() -> str:
    """
    Generate a unique document ID.
    
    Returns:
        Document ID with "doc-" prefix.
    """
    return generate_prefixed_id("doc")


def generate_parent_id() -> str:
    """
    Generate a unique parent chunk ID.
    
    Returns:
        Parent ID with "parent-" prefix.
    """
    return generate_prefixed_id("parent")


def generate_chunk_id() -> str:
    """
    Generate a unique child chunk ID.
    
    Returns:
        Chunk ID with "chunk-" prefix.
    """
    return generate_prefixed_id("chunk")


def generate_edge_id() -> str:
    """
    Generate a unique edge ID.
    
    Returns:
        Edge ID with "edge-" prefix.
    """
    return generate_prefixed_id("edge")


def generate_block_id() -> str:
    """
    Generate a unique block ID.
    
    Returns:
        Block ID with "block-" prefix.
    """
    return generate_prefixed_id("block")


def is_valid_uuid(value: str) -> bool:
    """
    Check if a string is a valid UUID v4.
    
    Args:
        value: String to validate.
        
    Returns:
        True if valid UUID v4, False otherwise.
    """
    if not value:
        return False
    return bool(UUID_PATTERN.match(value))


def is_valid_prefixed_id(value: str) -> bool:
    """
    Check if a string is a valid prefixed UUID.
    
    Args:
        value: String to validate (e.g., "doc-550e8400-...").
        
    Returns:
        True if valid prefixed UUID, False otherwise.
    """
    if not value:
        return False
    return bool(PREFIXED_ID_PATTERN.match(value))


def extract_prefix(prefixed_id: str) -> Optional[str]:
    """
    Extract the prefix from a prefixed ID.
    
    Args:
        prefixed_id: Prefixed UUID string.
        
    Returns:
        The prefix string, or None if invalid format.
    """
    if not prefixed_id or "-" not in prefixed_id:
        return None
    
    parts = prefixed_id.split("-", 1)
    if len(parts) != 2:
        return None
    
    prefix = parts[0]
    uuid_part = parts[1]
    
    if is_valid_uuid(uuid_part):
        return prefix
    
    return None


def extract_uuid(prefixed_id: str) -> Optional[str]:
    """
    Extract the UUID from a prefixed ID.
    
    Args:
        prefixed_id: Prefixed UUID string.
        
    Returns:
        The UUID string, or None if invalid format.
    """
    if not prefixed_id or "-" not in prefixed_id:
        return None
    
    parts = prefixed_id.split("-", 1)
    if len(parts) != 2:
        return None
    
    uuid_part = parts[1]
    
    if is_valid_uuid(uuid_part):
        return uuid_part
    
    return None


def generate_batch_ids(prefix: str, count: int) -> list:
    """
    Generate multiple prefixed IDs at once.
    
    Args:
        prefix: Prefix for all IDs.
        count: Number of IDs to generate.
        
    Returns:
        List of prefixed UUID strings.
    """
    return [generate_prefixed_id(prefix) for _ in range(count)]


class IDGenerator:
    """
    Class-based ID generator for stateful ID generation.
    
    Useful when you need to track generated IDs or 
    need consistent prefix handling.
    """
    
    def __init__(self, default_prefix: Optional[str] = None):
        """
        Initialize the ID generator.
        
        Args:
            default_prefix: Default prefix for generated IDs.
        """
        self.default_prefix = default_prefix
        self._generated_ids: list = []
    
    def generate(self, prefix: Optional[str] = None) -> str:
        """
        Generate a new ID.
        
        Args:
            prefix: Optional prefix (uses default if not provided).
            
        Returns:
            Generated ID string.
        """
        if prefix:
            new_id = generate_prefixed_id(prefix)
        elif self.default_prefix:
            new_id = generate_prefixed_id(self.default_prefix)
        else:
            new_id = generate_id()
        
        self._generated_ids.append(new_id)
        return new_id
    
    def generate_document(self) -> str:
        """Generate a document ID."""
        new_id = generate_document_id()
        self._generated_ids.append(new_id)
        return new_id
    
    def generate_parent(self) -> str:
        """Generate a parent chunk ID."""
        new_id = generate_parent_id()
        self._generated_ids.append(new_id)
        return new_id
    
    def generate_chunk(self) -> str:
        """Generate a child chunk ID."""
        new_id = generate_chunk_id()
        self._generated_ids.append(new_id)
        return new_id
    
    def generate_edge(self) -> str:
        """Generate an edge ID."""
        new_id = generate_edge_id()
        self._generated_ids.append(new_id)
        return new_id
    
    def generate_block(self) -> str:
        """Generate a block ID."""
        new_id = generate_block_id()
        self._generated_ids.append(new_id)
        return new_id
    
    @property
    def generated_count(self) -> int:
        """Get the number of IDs generated by this instance."""
        return len(self._generated_ids)
    
    @property
    def generated_ids(self) -> list:
        """Get a copy of all generated IDs."""
        return self._generated_ids.copy()
    
    def clear_history(self) -> None:
        """Clear the history of generated IDs."""
        self._generated_ids = []