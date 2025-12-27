"""
Extractors package for the Multilingual RAG Ingestion Pipeline.

Provides format-specific document extractors for:
- Plain text (.txt)
- Markdown (.md)
- HTML (.html, .htm)
- Microsoft Word (.docx)
- PDF (.pdf)
"""

from .base import (
    BaseExtractor,
    ExtractionError,
    ExtractorRegistry,
)

from .txt_extractor import TXTExtractor
from .markdown_extractor import MarkdownExtractor
from .html_extractor import HTMLExtractor, BS4_AVAILABLE
from .docx_extractor import DOCXExtractor, DOCX_AVAILABLE
from .pdf_extractor import PDFExtractor


def create_extractor_registry() -> ExtractorRegistry:
    """
    Create a registry with all available extractors.
    
    Returns:
        ExtractorRegistry with registered extractors.
    """
    registry = ExtractorRegistry()
    
    # Always available
    registry.register(TXTExtractor())
    registry.register(MarkdownExtractor())
    
    # Conditionally available based on dependencies
    if BS4_AVAILABLE:
        registry.register(HTMLExtractor())
    
    if DOCX_AVAILABLE:
        registry.register(DOCXExtractor())
    
    
    return registry


# Default registry instance
_default_registry = None


def get_default_registry() -> ExtractorRegistry:
    """
    Get the default extractor registry.
    
    Returns:
        Default ExtractorRegistry instance.
    """
    global _default_registry
    
    if _default_registry is None:
        _default_registry = create_extractor_registry()
    
    return _default_registry


def extract_file(file_path: str) -> list:
    """
    Extract blocks from a file using appropriate extractor.
    
    Convenience function using the default registry.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        List of ExtractedBlock objects.
    """
    return get_default_registry().extract(file_path)


def get_supported_extensions() -> list:
    """
    Get list of all supported file extensions.
    
    Returns:
        List of supported extensions (e.g., ['.txt', '.md', '.pdf']).
    """
    return get_default_registry().supported_extensions


def can_extract(file_path: str) -> bool:
    """
    Check if a file can be extracted.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        True if an extractor is available for the file format.
    """
    return get_default_registry().can_extract(file_path)


def get_extractor_for_file(file_path: str) -> BaseExtractor:
    """
    Get the appropriate extractor for a file.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        Appropriate extractor instance.
        
    Raises:
        ValueError: If no extractor available for format.
    """
    extractor = get_default_registry().get_extractor(file_path)
    
    if extractor is None:
        from pathlib import Path
        ext = Path(file_path).suffix
        raise ValueError(f"No extractor available for format: {ext}")
    
    return extractor


def get_available_extractors() -> dict:
    """
    Get information about available extractors.
    
    Returns:
        Dictionary with extractor availability information.
    """
    return {
        "txt": {
            "available": True,
            "extensions": TXTExtractor.SUPPORTED_EXTENSIONS,
            "class": "TXTExtractor",
        },
        "markdown": {
            "available": True,
            "extensions": MarkdownExtractor.SUPPORTED_EXTENSIONS,
            "class": "MarkdownExtractor",
        },
        "html": {
            "available": BS4_AVAILABLE,
            "extensions": HTMLExtractor.SUPPORTED_EXTENSIONS if BS4_AVAILABLE else [],
            "class": "HTMLExtractor",
            "dependency": "beautifulsoup4",
        },
        "docx": {
            "available": DOCX_AVAILABLE,
            "extensions": DOCXExtractor.SUPPORTED_EXTENSIONS if DOCX_AVAILABLE else [],
            "class": "DOCXExtractor",
            "dependency": "python-docx",
        },
        "pdf": {
            "class": "PDFExtractor",
            "dependency": "pymupdf or pypdf",
        },
    }