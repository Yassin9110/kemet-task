"""
Enumeration types for the Multilingual RAG Ingestion Pipeline.

This module defines all enums used throughout the pipeline:
- DocumentFormat: Supported input file formats
- Language: Detected content languages
- BlockType: Structural block classifications
- IngestionStatus: Document processing states
"""

from enum import Enum


class DocumentFormat(Enum):
    """Supported document formats for ingestion."""
    
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    TXT = "txt"


class Language(Enum):
    """Detected language classifications."""
    
    AR = "ar"          # Arabic
    EN = "en"          # English
    MIXED = "mixed"    # Both Arabic and English


class BlockType(Enum):
    """Structural classification of extracted text blocks."""
    
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    LIST = "list"
    UNKNOWN = "unknown"


class IngestionStatus(Enum):
    """Document ingestion processing states."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"