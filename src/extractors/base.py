"""
Base extractor for the Multilingual RAG Ingestion Pipeline.

This module defines the abstract base class that all format-specific
extractors must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union, Optional

from ..models.blocks import ExtractedBlock
from ..models.enums import DocumentFormat


class BaseExtractor(ABC):
    """
    Abstract base class for document extractors.
    
    All format-specific extractors must inherit from this class
    and implement the extract() method.
    """
    
    # Subclasses should override these
    SUPPORTED_EXTENSIONS: List[str] = []
    FORMAT: Optional[DocumentFormat] = None
    
    def __init__(self):
        """Initialize the extractor."""
        pass
    
    @abstractmethod
    def extract(self, file_path: Union[str, Path]) -> List[ExtractedBlock]:
        """
        Extract text blocks from a document.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            List of ExtractedBlock objects.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is not supported.
            ExtractionError: If extraction fails.
        """
        pass
    
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this extractor can handle the given file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            True if this extractor supports the file format.
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        return extension in self.SUPPORTED_EXTENSIONS
    
    def _validate_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that file exists and is supported.
        
        Args:
            file_path: Path to validate.
            
        Returns:
            Validated Path object.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If format not supported.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        if not self.can_extract(path):
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        
        return path
    
    def _read_file_text(self, file_path: Path, encoding: str = "utf-8") -> str:
        """
        Read file content as text.
        
        Args:
            file_path: Path to file.
            encoding: Text encoding.
            
        Returns:
            File content as string.
        """
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            # Try with different encodings
            for enc in ["utf-8-sig", "latin-1", "cp1256"]:  # cp1256 for Arabic
                try:
                    return file_path.read_text(encoding=enc)
                except UnicodeDecodeError:
                    continue
            raise
    
    def _read_file_bytes(self, file_path: Path) -> bytes:
        """
        Read file content as bytes.
        
        Args:
            file_path: Path to file.
            
        Returns:
            File content as bytes.
        """
        return file_path.read_bytes()


class ExtractionError(Exception):
    """Exception raised when extraction fails."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, cause: Optional[Exception] = None):
        """
        Initialize extraction error.
        
        Args:
            message: Error message.
            file_path: Path to the file that failed.
            cause: Original exception that caused this error.
        """
        self.file_path = file_path
        self.cause = cause
        super().__init__(message)


class ExtractorRegistry:
    """
    Registry for document extractors.
    
    Manages available extractors and routes files to appropriate extractors.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._extractors: dict = {}
    
    def register(self, extractor: BaseExtractor) -> None:
        """
        Register an extractor.
        
        Args:
            extractor: Extractor instance to register.
        """
        for ext in extractor.SUPPORTED_EXTENSIONS:
            self._extractors[ext.lower()] = extractor
    
    def get_extractor(self, file_path: Union[str, Path]) -> Optional[BaseExtractor]:
        """
        Get appropriate extractor for a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Appropriate extractor or None if not found.
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        return self._extractors.get(extension)
    
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """
        Check if any registered extractor can handle the file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            True if an extractor is available.
        """
        return self.get_extractor(file_path) is not None
    
    def extract(self, file_path: Union[str, Path]) -> List[ExtractedBlock]:
        """
        Extract blocks from a file using appropriate extractor.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            List of ExtractedBlock objects.
            
        Raises:
            ValueError: If no extractor available for format.
        """
        extractor = self.get_extractor(file_path)
        
        if extractor is None:
            path = Path(file_path)
            raise ValueError(f"No extractor available for format: {path.suffix}")
        
        return extractor.extract(file_path)
    
    @property
    def supported_extensions(self) -> List[str]:
        """Get list of all supported extensions."""
        return list(self._extractors.keys())