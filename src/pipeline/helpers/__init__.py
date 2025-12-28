"""
Helper utilities for the Multilingual RAG Ingestion Pipeline.
"""

from .id_generator import (
    generate_id,
    generate_prefixed_id,
    generate_document_id,
    generate_parent_id,
    generate_chunk_id,
    generate_edge_id,
    generate_block_id,
    generate_batch_ids,
    is_valid_uuid,
    is_valid_prefixed_id,
    extract_prefix,
    extract_uuid,
    IDGenerator,
)

from .hasher import (
    hash_bytes,
    hash_text,
    hash_file,
    hash_file_object,
    verify_file_hash,
    verify_text_hash,
    get_file_fingerprint,
    get_text_fingerprint,
    format_hash,
    parse_hash,
    is_valid_sha256,
    FileHasher,
)