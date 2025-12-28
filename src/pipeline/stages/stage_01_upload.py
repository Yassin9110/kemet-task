"""
Stage 1: File Upload

Accepts a file path, validates existence, assigns document ID,
and copies the file to immutable storage.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Dict
from logging import Logger

from src.config.settings import PipelineConfig
from src.pipeline.helpers.id_generator import generate_document_id
from src.storage.file_storage import FileStorage
from src.storage.file_storage import StoredFile


@dataclass
class UploadInput:
    """Input for the upload stage."""
    file_path: str
    metadata: Optional[Dict] = None


@dataclass
class UploadOutput:
    """Output from the upload stage."""
    document_id: str
    original_path: str
    stored_file: StoredFile
    file_name: str
    file_extension: str
    file_size_bytes: int
    metadata: Dict


class FileNotFoundError(Exception):
    """Raised when the input file does not exist."""
    pass


class FileTooLargeError(Exception):
    """Raised when the file exceeds maximum allowed size."""
    pass


class UploadStage:
    """
    Stage 1: File Upload
    
    Responsibilities:
    - Validate file exists
    - Check file size limits
    - Assign unique document ID
    - Copy file to immutable storage
    - Extract basic file info
    """
    
    STAGE_NAME = "File Upload"
    STAGE_NUMBER = 1
    TOTAL_STAGES = 11
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the upload stage.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.file_storage = FileStorage(config.raw_files_path)
    
    def execute(self, input_data: UploadInput, logger: Logger) -> UploadOutput:
        """
        Execute the upload stage.
        
        Args:
            input_data: Upload input containing file path and optional metadata.
            logger: Logger instance for progress tracking.
            
        Returns:
            UploadOutput with document ID, paths, and file info.
            
        Raises:
            FileNotFoundError: If file does not exist.
            FileTooLargeError: If file exceeds size limit.
        """
        start_time = time.time()
        
        file_path = input_data.file_path
        metadata = input_data.metadata or {}
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Path is not a file: {file_path}")
        
        # Get file info
        file_size_bytes = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        _, file_extension = os.path.splitext(file_name)
        file_extension = file_extension.lower().lstrip('.')
        
        # Validate file size
        max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
        if file_size_bytes > max_size_bytes:
            raise FileTooLargeError(
                f"File size ({file_size_bytes / (1024*1024):.2f} MB) "
                f"exceeds maximum allowed ({self.config.max_file_size_mb} MB)"
            )
        
        # Validate non-empty file
        if file_size_bytes == 0:
            raise ValueError("File is empty")
        
        # Generate document ID
        document_id = generate_document_id()
        
        # Copy to immutable storage
        stored_file = self.file_storage.save(
            source_path=file_path,
            document_id=document_id,
        )
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log completion
        logger.info(
            f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
            f"{self.STAGE_NAME} ✓ ({duration_ms}ms)"
        )
        logger.info(f"  → Stored at: {stored_file.stored_path}")
        
        return UploadOutput(
            document_id=document_id,
            original_path=os.path.abspath(file_path),
            stored_file=stored_file,
            file_name=file_name,
            file_extension=file_extension,
            file_size_bytes=file_size_bytes,
            metadata=metadata
        )