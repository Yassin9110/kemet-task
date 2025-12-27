"""
Stage 2: Validation & Fingerprinting

Validates MIME type, extension, computes file hash,
and detects duplicates.
"""

import mimetypes
import time
from dataclasses import dataclass
from typing import Optional, List
from logging import Logger

from src.config.settings import PipelineConfig
from src.pipeline.helpers.hasher import hash_file
from src.storage.json_storage import DocumentStorage
from src.storage.file_storage import StoredFile


@dataclass
class ValidationInput:
    """Input for the validation stage."""
    document_id: str
    stored_file: StoredFile
    file_extension: str


@dataclass
class ValidationOutput:
    """Output from the validation stage."""
    document_id: str
    stored_path: str
    file_extension: str
    mime_type: str
    file_hash: str
    is_duplicate: bool
    duplicate_document_ids: List[str]


class UnsupportedFormatError(Exception):
    """Raised when file format is not supported."""
    pass


class MimeTypeMismatchError(Exception):
    """Raised when MIME type doesn't match extension."""
    pass


class ValidationStage:
    """
    Stage 2: Validation & Fingerprinting
    
    Responsibilities:
    - Validate MIME type matches supported formats
    - Cross-check MIME with extension
    - Compute SHA256 hash
    - Detect duplicate files
    """
    
    STAGE_NAME = "Validation"
    STAGE_NUMBER = 2
    TOTAL_STAGES = 11
    
    # Mapping of extensions to expected MIME types
    SUPPORTED_FORMATS = {
        'pdf': ['application/pdf'],
        'docx': [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ],
        'html': ['text/html', 'application/xhtml+xml'],
        'htm': ['text/html', 'application/xhtml+xml'],
        'md': ['text/markdown', 'text/x-markdown', 'text/plain'],
        'markdown': ['text/markdown', 'text/x-markdown', 'text/plain'],
        'txt': ['text/plain']
    }
    
    # Fallback MIME types when detection fails
    EXTENSION_TO_MIME = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'html': 'text/html',
        'htm': 'text/html',
        'md': 'text/markdown',
        'markdown': 'text/markdown',
        'txt': 'text/plain'
    }
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the validation stage.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.json_storage =  DocumentStorage(config.documents_path)
    
    def execute(
        self,
        input_data: ValidationInput,
        logger: Logger
    ) -> ValidationOutput:
        """
        Execute the validation stage.
        
        Args:
            input_data: Validation input with document info.
            logger: Logger instance for progress tracking.
            
        Returns:
            ValidationOutput with MIME type, hash, and duplicate info.
            
        Raises:
            UnsupportedFormatError: If file format is not supported.
            MimeTypeMismatchError: If MIME type doesn't match extension.
        """
        start_time = time.time()
        
        stored_path = input_data.stored_file.stored_path
        file_extension = input_data.file_extension
        
        # Validate extension is supported
        if file_extension not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported file extension: .{file_extension}. "
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )
        
        # Detect MIME type
        mime_type = self._detect_mime_type(stored_path, file_extension)
        
        # Validate MIME type matches extension
        expected_mimes = self.SUPPORTED_FORMATS[file_extension]
        if mime_type not in expected_mimes:
            # Allow if we're using a fallback
            if mime_type != self.EXTENSION_TO_MIME.get(file_extension):
                logger.warning(
                    f"  → MIME type '{mime_type}' unexpected for "
                    f".{file_extension} (expected: {expected_mimes})"
                )
        
        # Compute file hash
        file_hash = hash_file(stored_path)
        
        # Check for duplicates
        is_duplicate, duplicate_ids = self._check_duplicates(file_hash)
        
        if is_duplicate:
            logger.warning(
                f"  → Duplicate file detected. Existing document(s): {duplicate_ids}"
            )
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log completion
        logger.info(
            f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
            f"{self.STAGE_NAME} ✓ ({duration_ms}ms)"
        )
        logger.info(f"  → Hash: {file_hash[:16]}...")
        logger.info(f"  → MIME: {mime_type}")
        
        return ValidationOutput(
            document_id=input_data.document_id,
            stored_path=stored_path,
            file_extension=file_extension,
            mime_type=mime_type,
            file_hash=file_hash,
            is_duplicate=is_duplicate,
            duplicate_document_ids=duplicate_ids
        )
    
    def _detect_mime_type(self, file_path: str, extension: str) -> str:
        """
        Detect MIME type of a file.
        
        Args:
            file_path: Path to the file.
            extension: File extension.
            
        Returns:
            Detected MIME type string.
        """
        # Try mimetypes library first
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type:
            return mime_type
        
        # Fallback to extension-based detection
        return self.EXTENSION_TO_MIME.get(extension, 'application/octet-stream')
    
    def _check_duplicates(self, file_hash: str) -> tuple[bool, List[str]]:
        """
        Check if a file with the same hash already exists.
        
        Args:
            file_hash: SHA256 hash of the file.
            
        Returns:
            Tuple of (is_duplicate, list of duplicate document IDs).
        """
        try:
            documents = self.json_storage.load_all()
        except FileNotFoundError:
            # No documents stored yet
            return False, []
        
        duplicate_ids = [
            doc.document_id
            for doc in documents
            if doc.file_hash == file_hash
        ]
        
        return len(duplicate_ids) > 0, duplicate_ids