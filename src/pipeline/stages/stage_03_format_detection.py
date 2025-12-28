"""
Stage 3: Format Detection

Determines the document format, preferring MIME type over extension,
with content sniffing as fallback.
"""

import time
from dataclasses import dataclass
from logging import Logger

from src.config.settings import PipelineConfig
from src.models.enums import DocumentFormat


@dataclass
class FormatDetectionInput:
    """Input for the format detection stage."""
    document_id: str
    stored_path: str
    file_extension: str
    mime_type: str


@dataclass
class FormatDetectionOutput:
    """Output from the format detection stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat


class FormatDetectionError(Exception):
    """Raised when format cannot be determined."""
    pass


class FormatDetectionStage:
    """
    Stage 3: Format Detection
    
    Responsibilities:
    - Determine document format from MIME type (preferred)
    - Fallback to extension if MIME unclear
    - Content sniffing as last resort
    - Output standardized DocumentFormat enum
    """
    
    STAGE_NAME = "Format Detection"
    STAGE_NUMBER = 3
    TOTAL_STAGES = 11
    
    # MIME type to format mapping
    MIME_TO_FORMAT = {
        'application/pdf': DocumentFormat.PDF,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentFormat.DOCX,
        'text/html': DocumentFormat.HTML,
        'application/xhtml+xml': DocumentFormat.HTML,
        'text/markdown': DocumentFormat.MARKDOWN,
        'text/x-markdown': DocumentFormat.MARKDOWN,
        'text/plain': None,  # Ambiguous - could be TXT or MD
    }
    
    # Extension to format mapping
    EXTENSION_TO_FORMAT = {
        'pdf': DocumentFormat.PDF,
        'docx': DocumentFormat.DOCX,
        'html': DocumentFormat.HTML,
        'htm': DocumentFormat.HTML,
        'md': DocumentFormat.MARKDOWN,
        'markdown': DocumentFormat.MARKDOWN,
        'txt': DocumentFormat.TXT,
    }
    
    # Magic bytes for content sniffing
    MAGIC_BYTES = {
        b'%PDF': DocumentFormat.PDF,
        b'PK\x03\x04': DocumentFormat.DOCX,  # ZIP-based (DOCX, XLSX, etc.)
        b'<!DOCTYPE html': DocumentFormat.HTML,
        b'<html': DocumentFormat.HTML,
        b'<!doctype html': DocumentFormat.HTML,
    }
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the format detection stage.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
    
    def execute(self, input_data: FormatDetectionInput, logger: Logger) -> FormatDetectionOutput:
        """
        Execute the format detection stage.
        
        Args:
            input_data: Format detection input with file info.
            logger: Logger instance for progress tracking.
            
        Returns:
            FormatDetectionOutput with determined DocumentFormat.
            
        Raises:
            FormatDetectionError: If format cannot be determined.
        """
        start_time = time.time()
        
        stored_path = input_data.stored_path
        file_extension = input_data.file_extension
        mime_type = input_data.mime_type
        
        detected_format = None
        detection_method = ""
    
        # Step 1: Try MIME type
        if mime_type in self.MIME_TO_FORMAT:
            detected_format = self.MIME_TO_FORMAT[mime_type]
            if detected_format is not None:
                detection_method = "MIME type"
        
        # Step 2: If MIME was ambiguous (text/plain), use extension
        if detected_format is None and file_extension in self.EXTENSION_TO_FORMAT:
            detected_format = self.EXTENSION_TO_FORMAT[file_extension]
            detection_method = "file extension"
        
        # Step 3: Content sniffing as fallback
        if detected_format is None:
            detected_format = self._sniff_content(stored_path)
            if detected_format is not None:
                detection_method = "content sniffing"
        
        # Final fallback based on extension
        if detected_format is None and file_extension in self.EXTENSION_TO_FORMAT:
            detected_format = self.EXTENSION_TO_FORMAT[file_extension]
            detection_method = "extension fallback"
        
        if detected_format is None:
            raise FormatDetectionError(
                f"Could not determine format for file. "
                f"MIME: {mime_type}, Extension: .{file_extension}"
            )
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log completion
        logger.info(
            f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
            f"{self.STAGE_NAME} ✓ ({duration_ms}ms)"
        )
        logger.info(f"  → Format: {detected_format.value} (via {detection_method})")
        
        return FormatDetectionOutput(
            document_id=input_data.document_id,
            stored_path=stored_path,
            format=detected_format
        )
    
    def _sniff_content(self, file_path: str) -> DocumentFormat | None:
        """
        Detect format by examining file content (magic bytes).
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Detected DocumentFormat or None if inconclusive.
        """
        try:
            with open(file_path, 'rb') as f:
                # Read first 512 bytes for analysis
                header = f.read(512)
            
            if not header:
                return None
            
            # Check magic bytes
            for magic, format_type in self.MAGIC_BYTES.items():
                if header.startswith(magic) or magic in header[:100]:
                    return format_type
            
            # Check for markdown indicators
            header_text = header.decode('utf-8', errors='ignore').lower()
            if self._looks_like_markdown(header_text):
                return DocumentFormat.MARKDOWN
            
            # Check for HTML-like content
            if '<html' in header_text or '<!doctype' in header_text:
                return DocumentFormat.HTML
            
            # Default to TXT if it's readable text
            if self._is_readable_text(header):
                return DocumentFormat.TXT
            
            return None
            
        except Exception:
            return None
    
    def _looks_like_markdown(self, text: str) -> bool:
        """
        Check if text content looks like Markdown.
        
        Args:
            text: Text content to analyze.
            
        Returns:
            True if content appears to be Markdown.
        """
        markdown_indicators = [
            '# ',      # Heading
            '## ',     # Heading level 2
            '### ',    # Heading level 3
            '* ',      # Unordered list
            '- ',      # Unordered list
            '```',     # Code block
            '[',       # Link start
            '**',      # Bold
            '__',      # Bold alternative
        ]
        
        indicator_count = sum(1 for ind in markdown_indicators if ind in text)
        return indicator_count >= 2
    
    def _is_readable_text(self, data: bytes) -> bool:
        """
        Check if binary data appears to be readable text.
        
        Args:
            data: Binary data to analyze.
            
        Returns:
            True if data appears to be text.
        """
        try:
            # Try to decode as UTF-8
            text = data.decode('utf-8')
            
            # Check for high ratio of printable characters
            printable_count = sum(
                1 for c in text
                if c.isprintable() or c in '\n\r\t'
            )
            
            return printable_count / len(text) > 0.9 if text else False
            
        except UnicodeDecodeError:
            # Try Latin-1 as fallback
            try:
                text = data.decode('latin-1')
                printable_count = sum(
                    1 for c in text
                    if c.isprintable() or c in '\n\r\t'
                )
                return printable_count / len(text) > 0.9 if text else False
            except Exception:
                return False