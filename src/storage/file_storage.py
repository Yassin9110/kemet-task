"""
File storage utilities for the Multilingual RAG Ingestion Pipeline.

This module provides functions for storing and retrieving raw document
files. Files are stored immutably with their document ID as the filename.
"""

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from ..pipeline.helpers import hash_file

@dataclass
class StoredFile:
    """Information about a stored file."""
    
    document_id: str
    """Document ID (filename without extension)."""
    
    stored_path: Path
    """Full path to stored file."""
    
    original_extension: str
    """Original file extension."""
    
    size_bytes: int
    """File size in bytes."""
    
    created_at: datetime
    """When the file was stored (from file system)."""
    
    file_hash: Optional[str] = None
    """SHA256 hash of the file."""
    
    @property
    def size_kb(self) -> float:
        """File size in kilobytes."""
        return self.size_bytes / 1024
    
    @property
    def size_mb(self) -> float:
        """File size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def filename(self) -> str:
        """Full filename with extension."""
        return self.stored_path.name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "stored_path": str(self.stored_path),
            "original_extension": self.original_extension,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "file_hash": self.file_hash,
        }


@dataclass
class StorageStats:
    """Statistics about file storage."""
    
    total_files: int
    """Total number of stored files."""
    
    total_size_bytes: int
    """Total size of all files in bytes."""
    
    files_by_extension: Dict[str, int]
    """Count of files by extension."""
    
    storage_path: Path
    """Path to storage directory."""
    
    @property
    def total_size_mb(self) -> float:
        """Total size in megabytes."""
        return self.total_size_bytes / (1024 * 1024)
    
    @property
    def average_file_size_kb(self) -> float:
        """Average file size in kilobytes."""
        if self.total_files == 0:
            return 0
        return (self.total_size_bytes / self.total_files) / 1024


class FileStorage:
    """
    Manages raw file storage for documents.
    
    Files are stored with their document ID as the base filename,
    preserving the original extension. Files are treated as immutable
    once stored.
    """
    
    def __init__(self, storage_path: Union[str, Path]):
        """
        Initialize file storage.
        
        Args:
            storage_path: Directory path for storing files.
        """
        self.storage_path = Path(storage_path)
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _get_extension(self, file_path: Union[str, Path]) -> str:
        """Extract extension from file path."""
        path = Path(file_path)
        ext = path.suffix.lower()
        return ext if ext else ""
    
    def _build_stored_filename(self, document_id: str, extension: str) -> str:
        """Build the stored filename from document ID and extension."""
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        return f"{document_id}{extension}"
    
    def _get_stored_path(self, document_id: str, extension: str) -> Path:
        """Get the full path for a stored file."""
        filename = self._build_stored_filename(document_id, extension)
        return self.storage_path / filename
    
    def save(
        self,
        source_path: Union[str, Path],
        document_id: str,
        compute_hash: bool = True
    ) -> StoredFile:
        """
        Save a file to storage.
        
        Copies the source file to storage with the document ID as filename.
        
        Args:
            source_path: Path to the source file.
            document_id: Unique document identifier.
            compute_hash: Whether to compute SHA256 hash.
            
        Returns:
            StoredFile with information about the stored file.
            
        Raises:
            FileNotFoundError: If source file doesn't exist.
            FileExistsError: If file with this document ID already exists.
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        if not source_path.is_file():
            raise ValueError(f"Source path is not a file: {source_path}")
        
        extension = self._get_extension(source_path)
        stored_path = self._get_stored_path(document_id, extension)
        
        if stored_path.exists():
            raise FileExistsError(f"File already exists for document: {document_id}")
        
        # Copy file to storage
        shutil.copy2(source_path, stored_path)
        
        # Get file info
        stat = stored_path.stat()
        
        # Compute hash if requested
        file_hash = None
        if compute_hash:
            file_hash = hash_file(stored_path)
        
        return StoredFile(
            document_id=document_id,
            stored_path=stored_path,
            original_extension=extension,
            size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            file_hash=file_hash,
        )
    
    def load(self, document_id: str, extension: str = "") -> bytes:
        """
        Load file content by document ID.
        
        Args:
            document_id: Document identifier.
            extension: File extension (with or without dot).
            
        Returns:
            File content as bytes.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        stored_path = self._find_file(document_id, extension)
        
        if stored_path is None:
            raise FileNotFoundError(f"No file found for document: {document_id}")
        
        return stored_path.read_bytes()
    
    def load_text(
        self,
        document_id: str,
        extension: str = "",
        encoding: str = "utf-8"
    ) -> str:
        """
        Load file content as text.
        
        Args:
            document_id: Document identifier.
            extension: File extension.
            encoding: Text encoding (default: utf-8).
            
        Returns:
            File content as string.
        """
        content = self.load(document_id, extension)
        return content.decode(encoding)
    
    def delete(self, document_id: str, extension: str = "") -> bool:
        """
        Delete a stored file.
        
        Args:
            document_id: Document identifier.
            extension: File extension.
            
        Returns:
            True if file was deleted, False if not found.
        """
        stored_path = self._find_file(document_id, extension)
        
        if stored_path is None:
            return False
        
        stored_path.unlink()
        return True
    
    def exists(self, document_id: str, extension: str = "") -> bool:
        """
        Check if a file exists in storage.
        
        Args:
            document_id: Document identifier.
            extension: File extension (optional, will search if not provided).
            
        Returns:
            True if file exists, False otherwise.
        """
        return self._find_file(document_id, extension) is not None
    
    def _find_file(self, document_id: str, extension: str = "") -> Optional[Path]:
        """
        Find a file by document ID.
        
        If extension is provided, looks for exact match.
        Otherwise, searches for any file starting with document ID.
        
        Args:
            document_id: Document identifier.
            extension: Optional extension to match.
            
        Returns:
            Path to file if found, None otherwise.
        """
        if extension:
            stored_path = self._get_stored_path(document_id, extension)
            if stored_path.exists():
                return stored_path
            return None
        
        # Search for any file with this document ID
        for file_path in self.storage_path.iterdir():
            if file_path.is_file() and file_path.stem == document_id:
                return file_path
        
        return None
    
    def get_path(self, document_id: str, extension: str = "") -> Optional[Path]:
        """
        Get the storage path for a document.
        
        Args:
            document_id: Document identifier.
            extension: File extension.
            
        Returns:
            Path to stored file, or None if not found.
        """
        return self._find_file(document_id, extension)
    
    def get_info(self, document_id: str, extension: str = "") -> Optional[StoredFile]:
        """
        Get information about a stored file.
        
        Args:
            document_id: Document identifier.
            extension: File extension.
            
        Returns:
            StoredFile with file information, or None if not found.
        """
        stored_path = self._find_file(document_id, extension)
        
        if stored_path is None:
            return None
        
        stat = stored_path.stat()
        
        return StoredFile(
            document_id=document_id,
            stored_path=stored_path,
            original_extension=stored_path.suffix,
            size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            file_hash=None,  # Don't compute hash unless requested
        )
    
    def get_info_with_hash(self, document_id: str, extension: str = "") -> Optional[StoredFile]:
        """
        Get information about a stored file including hash.
        
        Args:
            document_id: Document identifier.
            extension: File extension.
            
        Returns:
            StoredFile with file information and hash.
        """
        info = self.get_info(document_id, extension)
        
        if info is not None:
            info.file_hash = hash_file(info.stored_path)
        
        return info
    
    def list_files(self) -> List[StoredFile]:
        """
        List all stored files.
        
        Returns:
            List of StoredFile objects.
        """
        files = []
        
        for file_path in self.storage_path.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append(StoredFile(
                    document_id=file_path.stem,
                    stored_path=file_path,
                    original_extension=file_path.suffix,
                    size_bytes=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_ctime),
                ))
        
        return files
    
    def list_document_ids(self) -> List[str]:
        """
        List all document IDs in storage.
        
        Returns:
            List of document IDs.
        """
        return [f.document_id for f in self.list_files()]
    
    def get_stats(self) -> StorageStats:
        """
        Get storage statistics.
        
        Returns:
            StorageStats with storage information.
        """
        files = self.list_files()
        
        total_size = sum(f.size_bytes for f in files)
        
        by_extension: Dict[str, int] = {}
        for f in files:
            ext = f.original_extension or "(no extension)"
            by_extension[ext] = by_extension.get(ext, 0) + 1
        
        return StorageStats(
            total_files=len(files),
            total_size_bytes=total_size,
            files_by_extension=by_extension,
            storage_path=self.storage_path,
        )
    
    def clear(self, confirm: bool = False) -> int:
        """
        Delete all files in storage.
        
        Args:
            confirm: Must be True to actually delete files.
            
        Returns:
            Number of files deleted.
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to clear storage")
        
        count = 0
        for file_path in self.storage_path.iterdir():
            if file_path.is_file():
                file_path.unlink()
                count += 1
        
        return count
    
    def find_by_hash(self, file_hash: str) -> Optional[StoredFile]:
        """
        Find a file by its hash.
        
        Args:
            file_hash: SHA256 hash to search for.
            
        Returns:
            StoredFile if found, None otherwise.
        """
        for file_path in self.storage_path.iterdir():
            if file_path.is_file():
                current_hash = hash_file(file_path)
                if current_hash.lower() == file_hash.lower():
                    stat = file_path.stat()
                    return StoredFile(
                        document_id=file_path.stem,
                        stored_path=file_path,
                        original_extension=file_path.suffix,
                        size_bytes=stat.st_size,
                        created_at=datetime.fromtimestamp(stat.st_ctime),
                        file_hash=current_hash,
                    )
        
        return None
    
    def is_duplicate(self, source_path: Union[str, Path]) -> Optional[str]:
        """
        Check if a file is a duplicate of an existing stored file.
        
        Args:
            source_path: Path to file to check.
            
        Returns:
            Document ID of duplicate if found, None otherwise.
        """
        source_hash = hash_file(source_path)
        existing = self.find_by_hash(source_hash)
        
        if existing:
            return existing.document_id
        
        return None