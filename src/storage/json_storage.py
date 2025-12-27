"""
JSON storage utilities for the Multilingual RAG Ingestion Pipeline.

This module provides classes for persisting documents, chunks, and edges
to JSON files. Each storage class handles serialization/deserialization
of its respective data type.
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, TypeVar, Generic, Union
from filelock import FileLock

from ..models.document import Document, IngestionResult
from ..models.chunks import ParentChunk, ChildChunk
from ..models.edges import SemanticEdge


T = TypeVar('T')


class JSONStorage(ABC, Generic[T]):
    """
    Abstract base class for JSON storage operations.
    
    Provides common functionality for reading/writing JSON files
    with file locking for concurrent access safety.
    """
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize JSON storage.
        
        Args:
            file_path: Path to the JSON file.
        """
        self.file_path = Path(file_path)
        self.lock_path = Path(f"{file_path}.lock")
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Create parent directory if it doesn't exist."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _get_lock(self) -> FileLock:
        """Get a file lock for thread-safe operations."""
        return FileLock(self.lock_path)
    
    def _read_raw(self) -> Dict[str, Any]:
        """
        Read raw JSON data from file.
        
        Returns:
            Dictionary from JSON file, or empty dict if file doesn't exist.
        """
        if not self.file_path.exists():
            return self._get_empty_structure()
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _write_raw(self, data: Dict[str, Any]) -> None:
        """
        Write raw data to JSON file.
        
        Args:
            data: Dictionary to write.
        """
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @abstractmethod
    def _get_empty_structure(self) -> Dict[str, Any]:
        """Get the empty structure for this storage type."""
        pass
    
    @abstractmethod
    def _get_items_key(self) -> str:
        """Get the key for the items list in the JSON structure."""
        pass
    
    @abstractmethod
    def _item_to_dict(self, item: T) -> Dict[str, Any]:
        """Convert an item to dictionary for storage."""
        pass
    
    @abstractmethod
    def _dict_to_item(self, data: Dict[str, Any]) -> T:
        """Convert a dictionary back to an item."""
        pass
    
    @abstractmethod
    def _get_item_id(self, item: T) -> str:
        """Get the unique identifier for an item."""
        pass
    
    def save(self, item: T) -> None:
        """
        Save a single item.
        
        If item with same ID exists, it will be updated.
        
        Args:
            item: Item to save.
        """
        with self._get_lock():
            data = self._read_raw()
            items_key = self._get_items_key()
            item_id = self._get_item_id(item)
            item_dict = self._item_to_dict(item)
            
            # Find and update or append
            items = data.get(items_key, [])
            found = False
            for i, existing in enumerate(items):
                if self._get_item_id(self._dict_to_item(existing)) == item_id:
                    items[i] = item_dict
                    found = True
                    break
            
            if not found:
                items.append(item_dict)
            
            data[items_key] = items
            data["updated_at"] = datetime.now().isoformat()
            
            self._write_raw(data)
    
    def save_many(self, items: List[T]) -> None:
        """
        Save multiple items.
        
        Args:
            items: List of items to save.
        """
        with self._get_lock():
            data = self._read_raw()
            items_key = self._get_items_key()
            
            existing_items = data.get(items_key, [])
            existing_ids = {
                self._get_item_id(self._dict_to_item(e)) 
                for e in existing_items
            }
            
            # Update existing and collect new
            new_items = []
            for item in items:
                item_id = self._get_item_id(item)
                item_dict = self._item_to_dict(item)
                
                if item_id in existing_ids:
                    # Update existing
                    for i, existing in enumerate(existing_items):
                        if self._get_item_id(self._dict_to_item(existing)) == item_id:
                            existing_items[i] = item_dict
                            break
                else:
                    new_items.append(item_dict)
            
            data[items_key] = existing_items + new_items
            data["updated_at"] = datetime.now().isoformat()
            
            self._write_raw(data)
    
    def load(self, item_id: str) -> Optional[T]:
        """
        Load a single item by ID.
        
        Args:
            item_id: Unique identifier of the item.
            
        Returns:
            Item if found, None otherwise.
        """
        with self._get_lock():
            data = self._read_raw()
            items_key = self._get_items_key()
            
            for item_dict in data.get(items_key, []):
                item = self._dict_to_item(item_dict)
                if self._get_item_id(item) == item_id:
                    return item
            
            return None
    
    def load_all(self) -> List[T]:
        """
        Load all items.
        
        Returns:
            List of all items.
        """
        with self._get_lock():
            data = self._read_raw()
            items_key = self._get_items_key()
            
            return [
                self._dict_to_item(item_dict)
                for item_dict in data.get(items_key, [])
            ]
    
    def delete(self, item_id: str) -> bool:
        """
        Delete an item by ID.
        
        Args:
            item_id: Unique identifier of the item.
            
        Returns:
            True if item was deleted, False if not found.
        """
        with self._get_lock():
            data = self._read_raw()
            items_key = self._get_items_key()
            
            items = data.get(items_key, [])
            original_count = len(items)
            
            items = [
                item_dict for item_dict in items
                if self._get_item_id(self._dict_to_item(item_dict)) != item_id
            ]
            
            if len(items) < original_count:
                data[items_key] = items
                data["updated_at"] = datetime.now().isoformat()
                self._write_raw(data)
                return True
            
            return False
    
    def delete_many(self, item_ids: List[str]) -> int:
        """
        Delete multiple items by ID.
        
        Args:
            item_ids: List of item IDs to delete.
            
        Returns:
            Number of items deleted.
        """
        with self._get_lock():
            data = self._read_raw()
            items_key = self._get_items_key()
            
            items = data.get(items_key, [])
            original_count = len(items)
            
            ids_to_delete = set(item_ids)
            items = [
                item_dict for item_dict in items
                if self._get_item_id(self._dict_to_item(item_dict)) not in ids_to_delete
            ]
            
            deleted_count = original_count - len(items)
            
            if deleted_count > 0:
                data[items_key] = items
                data["updated_at"] = datetime.now().isoformat()
                self._write_raw(data)
            
            return deleted_count
    
    def exists(self, item_id: str) -> bool:
        """
        Check if an item exists.
        
        Args:
            item_id: Unique identifier of the item.
            
        Returns:
            True if item exists, False otherwise.
        """
        return self.load(item_id) is not None
    
    def count(self) -> int:
        """
        Get the total number of items.
        
        Returns:
            Number of items.
        """
        with self._get_lock():
            data = self._read_raw()
            return len(data.get(self._get_items_key(), []))
    
    def clear(self, confirm: bool = False) -> int:
        """
        Delete all items.
        
        Args:
            confirm: Must be True to actually delete.
            
        Returns:
            Number of items deleted.
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to clear storage")
        
        with self._get_lock():
            data = self._read_raw()
            items_key = self._get_items_key()
            
            count = len(data.get(items_key, []))
            
            data[items_key] = []
            data["updated_at"] = datetime.now().isoformat()
            
            self._write_raw(data)
            
            return count
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get storage metadata.
        
        Returns:
            Dictionary with metadata.
        """
        with self._get_lock():
            data = self._read_raw()
            
            return {
                "file_path": str(self.file_path),
                "item_count": len(data.get(self._get_items_key(), [])),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            }


class DocumentStorage(JSONStorage[Document]):
    """Storage for Document objects."""
    
    def _get_empty_structure(self) -> Dict[str, Any]:
        return {
            "documents": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    
    def _get_items_key(self) -> str:
        return "documents"
    
    def _item_to_dict(self, item: Document) -> Dict[str, Any]:
        return item.to_dict()
    
    def _dict_to_item(self, data: Dict[str, Any]) -> Document:
        return Document.from_dict(data)
    
    def _get_item_id(self, item: Document) -> str:
        return item.document_id
    
    def load_by_hash(self, file_hash: str) -> Optional[Document]:
        """
        Load a document by its file hash.
        
        Args:
            file_hash: SHA256 hash of the file.
            
        Returns:
            Document if found, None otherwise.
        """
        for doc in self.load_all():
            if doc.file_hash.lower() == file_hash.lower():
                return doc
        return None
    
    def load_by_status(self, status: str) -> List[Document]:
        """
        Load documents by ingestion status.
        
        Args:
            status: Status to filter by (pending, processing, completed, failed).
            
        Returns:
            List of documents with matching status.
        """
        return [
            doc for doc in self.load_all()
            if doc.ingestion_status.value == status
        ]
    
    def load_completed(self) -> List[Document]:
        """Load all completed documents."""
        return self.load_by_status("completed")
    
    def load_failed(self) -> List[Document]:
        """Load all failed documents."""
        return self.load_by_status("failed")
    
    def load_pending(self) -> List[Document]:
        """Load all pending documents."""
        return self.load_by_status("pending")


class ParentStorage(JSONStorage[ParentChunk]):
    """Storage for ParentChunk objects."""
    
    def _get_empty_structure(self) -> Dict[str, Any]:
        return {
            "parents": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    
    def _get_items_key(self) -> str:
        return "parents"
    
    def _item_to_dict(self, item: ParentChunk) -> Dict[str, Any]:
        return item.to_dict()
    
    def _dict_to_item(self, data: Dict[str, Any]) -> ParentChunk:
        return ParentChunk.from_dict(data)
    
    def _get_item_id(self, item: ParentChunk) -> str:
        return item.parent_id
    
    def load_by_document(self, document_id: str) -> List[ParentChunk]:
        """
        Load all parent chunks for a document.
        
        Args:
            document_id: Document ID to filter by.
            
        Returns:
            List of parent chunks for the document.
        """
        return [
            parent for parent in self.load_all()
            if parent.document_id == document_id
        ]
    
    def delete_by_document(self, document_id: str) -> int:
        """
        Delete all parent chunks for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Number of parents deleted.
        """
        parents = self.load_by_document(document_id)
        parent_ids = [p.parent_id for p in parents]
        return self.delete_many(parent_ids)


class ChildStorage(JSONStorage[ChildChunk]):
    """Storage for ChildChunk objects."""
    
    def _get_empty_structure(self) -> Dict[str, Any]:
        return {
            "children": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    
    def _get_items_key(self) -> str:
        return "children"
    
    def _item_to_dict(self, item: ChildChunk) -> Dict[str, Any]:
        # Don't store embeddings in JSON (they go to vector store)
        return item.to_dict(include_embedding=False)
    
    def _dict_to_item(self, data: Dict[str, Any]) -> ChildChunk:
        return ChildChunk.from_dict(data)
    
    def _get_item_id(self, item: ChildChunk) -> str:
        return item.chunk_id
    
    def load_by_document(self, document_id: str) -> List[ChildChunk]:
        """
        Load all child chunks for a document.
        
        Args:
            document_id: Document ID to filter by.
            
        Returns:
            List of child chunks for the document.
        """
        return [
            child for child in self.load_all()
            if child.document_id == document_id
        ]
    
    def load_by_parent(self, parent_id: str) -> List[ChildChunk]:
        """
        Load all child chunks for a parent.
        
        Args:
            parent_id: Parent chunk ID to filter by.
            
        Returns:
            List of child chunks for the parent.
        """
        return [
            child for child in self.load_all()
            if child.parent_id == parent_id
        ]
    
    def delete_by_document(self, document_id: str) -> int:
        """
        Delete all child chunks for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Number of children deleted.
        """
        children = self.load_by_document(document_id)
        child_ids = [c.chunk_id for c in children]
        return self.delete_many(child_ids)
    
    def delete_by_parent(self, parent_id: str) -> int:
        """
        Delete all child chunks for a parent.
        
        Args:
            parent_id: Parent chunk ID.
            
        Returns:
            Number of children deleted.
        """
        children = self.load_by_parent(parent_id)
        child_ids = [c.chunk_id for c in children]
        return self.delete_many(child_ids)


class EdgeStorage(JSONStorage[SemanticEdge]):
    """Storage for SemanticEdge objects."""
    
    def _get_empty_structure(self) -> Dict[str, Any]:
        return {
            "edges": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    
    def _get_items_key(self) -> str:
        return "edges"
    
    def _item_to_dict(self, item: SemanticEdge) -> Dict[str, Any]:
        return item.to_dict()
    
    def _dict_to_item(self, data: Dict[str, Any]) -> SemanticEdge:
        return SemanticEdge.from_dict(data)
    
    def _get_item_id(self, item: SemanticEdge) -> str:
        return item.edge_id
    
    def load_by_document(self, document_id: str) -> List[SemanticEdge]:
        """
        Load all edges for a document.
        
        Args:
            document_id: Document ID to filter by.
            
        Returns:
            List of edges for the document.
        """
        return [
            edge for edge in self.load_all()
            if edge.document_id == document_id
        ]
    
    def load_by_source(self, source_chunk_id: str) -> List[SemanticEdge]:
        """
        Load all edges from a source chunk.
        
        Args:
            source_chunk_id: Source chunk ID to filter by.
            
        Returns:
            List of edges from the source.
        """
        return [
            edge for edge in self.load_all()
            if edge.source_chunk_id == source_chunk_id
        ]
    
    def load_by_target(self, target_chunk_id: str) -> List[SemanticEdge]:
        """
        Load all edges to a target chunk.
        
        Args:
            target_chunk_id: Target chunk ID to filter by.
            
        Returns:
            List of edges to the target.
        """
        return [
            edge for edge in self.load_all()
            if edge.target_chunk_id == target_chunk_id
        ]
    
    def load_by_type(self, edge_type: str) -> List[SemanticEdge]:
        """
        Load all edges of a specific type.
        
        Args:
            edge_type: Edge type to filter by.
            
        Returns:
            List of edges of the specified type.
        """
        return [
            edge for edge in self.load_all()
            if edge.edge_type == edge_type
        ]
    
    def load_semantic_edges(self) -> List[SemanticEdge]:
        """Load all semantic similarity edges."""
        return self.load_by_type("semantic")
    
    def load_sibling_edges(self) -> List[SemanticEdge]:
        """Load all sibling edges (prev and next)."""
        return [
            edge for edge in self.load_all()
            if edge.edge_type in ("sibling_prev", "sibling_next")
        ]
    
    def delete_by_document(self, document_id: str) -> int:
        """
        Delete all edges for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Number of edges deleted.
        """
        edges = self.load_by_document(document_id)
        edge_ids = [e.edge_id for e in edges]
        return self.delete_many(edge_ids)
    
    def load_connected_chunks(self, chunk_id: str) -> List[str]:
        """
        Get all chunk IDs connected to a given chunk.
        
        Args:
            chunk_id: Chunk ID to find connections for.
            
        Returns:
            List of connected chunk IDs.
        """
        connected = set()
        
        for edge in self.load_all():
            if edge.source_chunk_id == chunk_id:
                connected.add(edge.target_chunk_id)
            elif edge.target_chunk_id == chunk_id:
                connected.add(edge.source_chunk_id)
        
        return list(connected)


class StorageManager:
    """
    Unified manager for all JSON storage types.
    
    Provides a single interface to access documents, parents,
    children, and edges storage.
    """
    
    def __init__(
        self,
        documents_path: Union[str, Path],
        parents_path: Union[str, Path],
        children_path: Union[str, Path],
        edges_path: Union[str, Path]
    ):
        """
        Initialize storage manager.
        
        Args:
            documents_path: Path to documents JSON file.
            parents_path: Path to parents JSON file.
            children_path: Path to children JSON file.
            edges_path: Path to edges JSON file.
        """
        self.documents = DocumentStorage(documents_path)
        self.parents = ParentStorage(parents_path)
        self.children = ChildStorage(children_path)
        self.edges = EdgeStorage(edges_path)
    
    @classmethod
    def from_config(cls, config: Any) -> "StorageManager":
        """
        Create storage manager from PipelineConfig.
        
        Args:
            config: PipelineConfig instance.
            
        Returns:
            Configured StorageManager.
        """
        return cls(
            documents_path=config.documents_path,
            parents_path=config.parents_path,
            children_path=config.children_path,
            edges_path=config.edges_path,
        )
    
    def delete_document_data(self, document_id: str) -> Dict[str, int]:
        """
        Delete all data associated with a document.
        
        Args:
            document_id: Document ID to delete.
            
        Returns:
            Dictionary with counts of deleted items.
        """
        deleted = {
            "edges": self.edges.delete_by_document(document_id),
            "children": self.children.delete_by_document(document_id),
            "parents": self.parents.delete_by_document(document_id),
            "documents": 1 if self.documents.delete(document_id) else 0,
        }
        
        return deleted
    
    def get_document_summary(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of all data for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Summary dictionary or None if document not found.
        """
        doc = self.documents.load(document_id)
        
        if doc is None:
            return None
        
        parents = self.parents.load_by_document(document_id)
        children = self.children.load_by_document(document_id)
        edges = self.edges.load_by_document(document_id)
        
        return {
            "document": doc.to_dict(),
            "parent_count": len(parents),
            "child_count": len(children),
            "edge_count": len(edges),
            "total_tokens": sum(c.token_count for c in children),
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all storage types.
        
        Returns:
            Dictionary with counts for each storage type.
        """
        return {
            "documents": self.documents.count(),
            "parents": self.parents.count(),
            "children": self.children.count(),
            "edges": self.edges.count(),
        }
    
    def clear_all(self, confirm: bool = False) -> Dict[str, int]:
        """
        Clear all storage.
        
        Args:
            confirm: Must be True to actually delete.
            
        Returns:
            Dictionary with counts of deleted items.
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to clear all storage")
        
        return {
            "edges": self.edges.clear(confirm=True),
            "children": self.children.clear(confirm=True),
            "parents": self.parents.clear(confirm=True),
            "documents": self.documents.clear(confirm=True),
        }