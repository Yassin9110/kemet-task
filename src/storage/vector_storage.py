"""
Vector storage utilities for the Multilingual RAG Ingestion Pipeline.

This module provides a wrapper around Qdrant for storing and searching
chunk embeddings. It handles serialization of chunk metadata and provides
filtering capabilities.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

# Try to import qdrant_client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        PointIdsList,
        FilterSelector,
        PointVectors,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from ..models.chunks import ChildChunk
from ..models.enums import Language, BlockType


@dataclass
class SearchResult:
    """Result of a vector similarity search."""
    
    chunk_id: str
    """ID of the matching chunk."""
    
    text: str
    """Text content of the chunk."""
    
    distance: float
    """Distance from query (lower = more similar)."""
    
    score: float
    """Similarity score (higher = more similar)."""
    
    metadata: Dict[str, Any]
    """Chunk metadata."""
    
    embedding: Optional[List[float]] = None
    """Embedding vector if requested."""
    
    @property
    def document_id(self) -> Optional[str]:
        """Get document ID from metadata."""
        return self.metadata.get("document_id")
    
    @property
    def parent_id(self) -> Optional[str]:
        """Get parent ID from metadata."""
        return self.metadata.get("parent_id")
    
    @property
    def language(self) -> Optional[str]:
        """Get language from metadata."""
        return self.metadata.get("language")
    
    @property
    def block_type(self) -> Optional[str]:
        """Get block type from metadata."""
        return self.metadata.get("block_type")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "distance": self.distance,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class VectorStoreStats:
    """Statistics about the vector store."""
    
    total_chunks: int
    """Total number of chunks stored."""
    
    unique_documents: int
    """Number of unique documents."""
    
    collection_name: str
    """Name of the Qdrant collection."""
    
    persist_directory: Optional[str]
    """Path to persistence directory."""
    
    chunks_by_language: Dict[str, int] = field(default_factory=dict)
    """Count of chunks by language."""
    
    chunks_by_document: Dict[str, int] = field(default_factory=dict)
    """Count of chunks by document."""


class VectorStorage:
    """
    Vector storage using Qdrant.
    
    Stores chunk embeddings with metadata for similarity search.
    Supports filtering by document, language, and other metadata.
    """
    
    DEFAULT_BATCH_SIZE = 100
    
    def __init__(
        self,
        persist_directory: Optional[Union[str, Path]] = None,
        collection_name: str = "chunks",
        embedding_dimension: int = 1536,
    ):
        """
        Initialize vector storage.
        
        Args:
            persist_directory: Directory for persistent storage (None for in-memory).
            collection_name: Name for the Qdrant collection.
            embedding_dimension: Expected embedding dimension (for validation).
            
        Raises:
            ImportError: If qdrant-client is not installed.
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required for VectorStorage. "
                "Install it with: pip install qdrant-client"
            )
        
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # Initialize Qdrant client
        self._client = self._create_client()
        self._ensure_collection_exists()
    
    def _create_client(self) -> "QdrantClient":
        """Create Qdrant client."""
        # Add traceback for debugging multiple calls
        import traceback
        
        if not hasattr(self.__class__, '_client_creation_counter'):
            self.__class__._client_creation_counter = 0
        
        self.__class__._client_creation_counter += 1
        
        print(f"[DEBUG] _create_client() called #{self.__class__._client_creation_counter}")
        print(f"  Persist Directory: {self.persist_directory}")
        
        # Print where it's being called from (last 3 stack frames)
        stack = traceback.extract_stack()[-4:-1]
        for frame in stack:
            print(f"  Called from: {frame.filename}:{frame.lineno} in {frame.name}")
        
        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=str(self.persist_directory))
        else:
            # In-memory client
            return QdrantClient(":memory:")
    
    def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists, create if not."""
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
    
    def _chunk_to_payload(self, chunk: ChildChunk) -> Dict[str, Any]:
        """
        Convert chunk to payload dictionary for storage.
        
        Includes text in payload (unlike ChromaDB which stores separately).
        """
        payload = {
            "text": chunk.text,
            "document_id": chunk.document_id,
            "language": chunk.language.value,
            "block_type": chunk.block_type.value,
            "token_count": chunk.token_count,
            "position_in_parent": chunk.position_in_parent,
            "created_at": chunk.created_at.isoformat(),
        }
        
        # Optional fields
        if chunk.parent_id:
            payload["parent_id"] = chunk.parent_id
        
        if chunk.page_number is not None:
            payload["page_number"] = chunk.page_number
        
        if chunk.prev_chunk_id:
            payload["prev_chunk_id"] = chunk.prev_chunk_id
        
        if chunk.next_chunk_id:
            payload["next_chunk_id"] = chunk.next_chunk_id
        
        if chunk.section_path:
            # Store as pipe-separated string for consistency
            payload["section_path"] = "|".join(chunk.section_path)
        
        return payload
    
    def _payload_to_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from payload (excluding text for backward compatibility)."""
        return {k: v for k, v in payload.items() if k != "text"}
    
    def _metadata_to_chunk_data(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Convert stored metadata back to chunk-like dictionary."""
        
        # Parse section_path from string
        section_path = []
        if "section_path" in metadata and metadata["section_path"]:
            section_path = metadata["section_path"].split("|")
        
        return {
            "chunk_id": chunk_id,
            "document_id": metadata.get("document_id"),
            "parent_id": metadata.get("parent_id"),
            "text": text,
            "token_count": metadata.get("token_count", 0),
            "language": metadata.get("language"),
            "block_type": metadata.get("block_type"),
            "section_path": section_path,
            "page_number": metadata.get("page_number"),
            "position_in_parent": metadata.get("position_in_parent", 0),
            "prev_chunk_id": metadata.get("prev_chunk_id"),
            "next_chunk_id": metadata.get("next_chunk_id"),
            "embedding": embedding,
            "created_at": metadata.get("created_at"),
        }
    
    def _validate_embedding(self, embedding: List[float], chunk_id: str) -> None:
        """Validate embedding dimension."""
        if len(embedding) != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch for chunk {chunk_id}: "
                f"expected {self.embedding_dimension}, got {len(embedding)}"
            )
    
    def _build_filter(self, where: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """
        Convert a simple where dict to Qdrant Filter.
        
        Supports simple key-value matching for backward compatibility
        with ChromaDB-style filters.
        
        Args:
            where: Dict like {"document_id": "doc1", "language": "en"}
            
        Returns:
            Qdrant Filter object or None if where is empty/None.
        """
        if not where:
            return None
        
        conditions = []
        for key, value in where.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )
        
        if not conditions:
            return None
        
        return Filter(must=conditions)
    
    def add(self, chunk: ChildChunk) -> None:
        """
        Add a single chunk to the vector store.
        
        Args:
            chunk: ChildChunk with embedding to store.
            
        Raises:
            ValueError: If chunk has no embedding or wrong dimension.
        """
        if chunk.embedding is None:
            raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
        
        self._validate_embedding(chunk.embedding, chunk.chunk_id)
        
        point = PointStruct(
            id=chunk.chunk_id,
            vector=chunk.embedding,
            payload=self._chunk_to_payload(chunk),
        )
        
        self._client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
    
    def add_many(self, chunks: List[ChildChunk], batch_size: Optional[int] = None) -> int:
        """
        Add multiple chunks to the vector store.
        
        Args:
            chunks: List of ChildChunks with embeddings.
            batch_size: Number of chunks per batch (default: 100).
            
        Returns:
            Number of chunks added.
            
        Raises:
            ValueError: If any chunk has no embedding or wrong dimension.
        """
        if not chunks:
            return 0
        
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE
        
        points = []
        print("\n\n in add many function\n")
        
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
            
            self._validate_embedding(chunk.embedding, chunk.chunk_id)

            point_id = chunk.chunk_id
            if point_id.startswith("chunk-"):
                point_id = point_id[6:]  # Remove "chunk-" prefix (6 characters)
            
            points.append(PointStruct(
                id=point_id,
                vector=chunk.embedding,
                payload=self._chunk_to_payload(chunk),
            ))
        
        print("\n\n Ready for adding collection \n\n")
        print(f"Points: {len(points)}")
        
        # Batch upsert
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
        
        print(f"collection added with chunks {len(chunks)}")
        
        return len(chunks)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> List[SearchResult]:
        """
        Search for similar chunks by embedding.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            where: Optional metadata filter (dict with key-value pairs).
            include_embeddings: Whether to include embeddings in results.
            
        Returns:
            List of SearchResult ordered by similarity.
        """
        query_filter = self._build_filter(where)
        
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=include_embeddings,
        )
        
        search_results = []
        
        for hit in results:
            payload = hit.payload or {}
            text = payload.get("text", "")
            metadata = self._payload_to_metadata(payload)
            
            # Qdrant returns score (higher = more similar for cosine)
            # Convert to distance for backward compatibility
            score = hit.score
            distance = 1.0 - score
            
            embedding = None
            if include_embeddings and hit.vector:
                embedding = list(hit.vector) if not isinstance(hit.vector, list) else hit.vector
            
            search_results.append(SearchResult(
                chunk_id=str(hit.id),
                text=text,
                distance=distance,
                score=score,
                metadata=metadata,
                embedding=embedding,
            ))
        
        return search_results
    
    def search_with_filter(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
        language: Optional[str] = None,
        block_type: Optional[str] = None,
        include_embeddings: bool = False,
    ) -> List[SearchResult]:
        """
        Search with convenient filter parameters.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            document_id: Filter by document ID.
            language: Filter by language (ar, en, mixed).
            block_type: Filter by block type.
            include_embeddings: Whether to include embeddings in results.
            
        Returns:
            List of SearchResult ordered by similarity.
        """
        where = {}
        
        if document_id:
            where["document_id"] = document_id
        
        if language:
            where["language"] = language
        
        if block_type:
            where["block_type"] = block_type
        
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where if where else None,
            include_embeddings=include_embeddings,
        )
    
    def get(self, chunk_id: str, include_embedding: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get a chunk by ID.
        
        Args:
            chunk_id: Chunk ID to retrieve.
            include_embedding: Whether to include embedding.
            
        Returns:
            Chunk data dictionary or None if not found.
        """
        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[chunk_id],
            with_payload=True,
            with_vectors=include_embedding,
        )
        
        if results:
            point = results[0]
            payload = point.payload or {}
            text = payload.get("text", "")
            metadata = self._payload_to_metadata(payload)
            
            embedding = None
            if include_embedding and point.vector:
                embedding = list(point.vector) if not isinstance(point.vector, list) else point.vector
            
            return self._metadata_to_chunk_data(
                chunk_id=str(point.id),
                text=text,
                metadata=metadata,
                embedding=embedding,
            )
        
        return None
    
    def get_many(
        self,
        chunk_ids: List[str],
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get multiple chunks by IDs.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve.
            include_embeddings: Whether to include embeddings.
            
        Returns:
            List of chunk data dictionaries.
        """
        if not chunk_ids:
            return []
        
        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=chunk_ids,
            with_payload=True,
            with_vectors=include_embeddings,
        )
        
        chunks = []
        
        for point in results:
            payload = point.payload or {}
            text = payload.get("text", "")
            metadata = self._payload_to_metadata(payload)
            
            embedding = None
            if include_embeddings and point.vector:
                embedding = list(point.vector) if not isinstance(point.vector, list) else point.vector
            
            chunks.append(self._metadata_to_chunk_data(
                chunk_id=str(point.id),
                text=text,
                metadata=metadata,
                embedding=embedding,
            ))
        
        return chunks
    
    def get_by_document(
        self,
        document_id: str,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID to filter by.
            include_embeddings: Whether to include embeddings.
            
        Returns:
            List of chunk data dictionaries.
        """
        query_filter = self._build_filter({"document_id": document_id})
        
        chunks = []
        offset = None
        
        while True:
            results, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                with_payload=True,
                with_vectors=include_embeddings,
                limit=100,
                offset=offset,
            )
            
            for point in results:
                payload = point.payload or {}
                text = payload.get("text", "")
                metadata = self._payload_to_metadata(payload)
                
                embedding = None
                if include_embeddings and point.vector:
                    embedding = list(point.vector) if not isinstance(point.vector, list) else point.vector
                
                chunks.append(self._metadata_to_chunk_data(
                    chunk_id=str(point.id),
                    text=text,
                    metadata=metadata,
                    embedding=embedding,
                ))
            
            if next_offset is None:
                break
            offset = next_offset
        
        return chunks
    
    def delete(self, chunk_id: str) -> bool:
        """
        Delete a chunk by ID.
        
        Args:
            chunk_id: Chunk ID to delete.
            
        Returns:
            True if deleted successfully.
        """
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[chunk_id]),
            )
            return True
        except Exception:
            return False
    
    def delete_many(self, chunk_ids: List[str]) -> int:
        """
        Delete multiple chunks by ID.
        
        Args:
            chunk_ids: List of chunk IDs to delete.
            
        Returns:
            Number of chunks deleted.
        """
        if not chunk_ids:
            return 0
        
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=chunk_ids),
            )
            return len(chunk_ids)
        except Exception:
            return 0
    
    def delete_by_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Number of chunks deleted.
        """
        # First count chunks to return accurate number
        chunks = self.get_by_document(document_id)
        count = len(chunks)
        
        if count == 0:
            return 0
        
        query_filter = self._build_filter({"document_id": document_id})
        
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=query_filter),
            )
            return count
        except Exception:
            return 0
    
    def update_embedding(self, chunk_id: str, embedding: List[float]) -> bool:
        """
        Update the embedding for an existing chunk.
        
        Args:
            chunk_id: Chunk ID to update.
            embedding: New embedding vector.
            
        Returns:
            True if updated successfully.
        """
        self._validate_embedding(embedding, chunk_id)
        
        try:
            self._client.update_vectors(
                collection_name=self.collection_name,
                points=[
                    PointVectors(
                        id=chunk_id,
                        vector=embedding,
                    )
                ],
            )
            return True
        except Exception:
            return False
    
    def update_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing chunk.
        
        Args:
            chunk_id: Chunk ID to update.
            metadata: New metadata (will be merged with existing).
            
        Returns:
            True if updated successfully.
        """
        try:
            self._client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[chunk_id],
            )
            return True
        except Exception:
            return False
    
    def exists(self, chunk_id: str) -> bool:
        """
        Check if a chunk exists in the store.
        
        Args:
            chunk_id: Chunk ID to check.
            
        Returns:
            True if chunk exists.
        """
        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[chunk_id],
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0
    
    def count(self) -> int:
        """
        Get total number of chunks in the store.
        
        Returns:
            Number of chunks.
        """
        collection_info = self._client.get_collection(self.collection_name)
        return collection_info.points_count
    
    def count_by_document(self, document_id: str) -> int:
        """
        Get number of chunks for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Number of chunks for the document.
        """
        query_filter = self._build_filter({"document_id": document_id})
        
        result = self._client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
        )
        
        return result.count
    
    def list_documents(self) -> List[str]:
        """
        List all unique document IDs in the store.
        
        Returns:
            List of unique document IDs.
        """
        document_ids = set()
        offset = None
        
        while True:
            results, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                with_payload=["document_id"],
                with_vectors=False,
                limit=100,
                offset=offset,
            )
            
            for point in results:
                if point.payload and "document_id" in point.payload:
                    document_ids.add(point.payload["document_id"])
            
            if next_offset is None:
                break
            offset = next_offset
        
        return list(document_ids)
    
    def get_stats(self) -> VectorStoreStats:
        """
        Get statistics about the vector store.
        
        Returns:
            VectorStoreStats with store information.
        """
        chunks_by_language: Dict[str, int] = {}
        chunks_by_document: Dict[str, int] = {}
        total_chunks = 0
        
        offset = None
        
        while True:
            results, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                with_payload=["language", "document_id"],
                with_vectors=False,
                limit=100,
                offset=offset,
            )
            
            for point in results:
                total_chunks += 1
                payload = point.payload or {}
                
                # Count by language
                lang = payload.get("language", "unknown")
                chunks_by_language[lang] = chunks_by_language.get(lang, 0) + 1
                
                # Count by document
                doc_id = payload.get("document_id", "unknown")
                chunks_by_document[doc_id] = chunks_by_document.get(doc_id, 0) + 1
            
            if next_offset is None:
                break
            offset = next_offset
        
        return VectorStoreStats(
            total_chunks=total_chunks,
            unique_documents=len(chunks_by_document),
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory) if self.persist_directory else None,
            chunks_by_language=chunks_by_language,
            chunks_by_document=chunks_by_document,
        )
    
    def clear(self, confirm: bool = False) -> int:
        """
        Delete all chunks from the store.
        
        Args:
            confirm: Must be True to actually delete.
            
        Returns:
            Number of chunks deleted.
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to clear vector store")
        
        count = self.count()
        
        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._ensure_collection_exists()
        
        return count
    
    def persist(self) -> None:
        """
        Persist the vector store to disk.
        
        Note: Qdrant automatically persists data when using persistent storage,
        so this method is a no-op. Kept for backward compatibility.
        """
        # Qdrant auto-persists, no action needed
        pass
    
    def reset(self) -> None:
        """
        Reset the vector store (delete and recreate collection).
        """
        self._client.delete_collection(self.collection_name)
        self._ensure_collection_exists()


def create_vector_storage_from_config(config: Any) -> VectorStorage:
    """
    Create VectorStorage from a PipelineConfig.
    
    Args:
        config: PipelineConfig instance.
        
    Returns:
        Configured VectorStorage instance.
    """
    # Consider renaming to config.vector_store_path in the future
    return VectorStorage(
        persist_directory=config.qdrant_path,
        collection_name="chunks",
        embedding_dimension=1536,  # Cohere multilingual dimension
    )