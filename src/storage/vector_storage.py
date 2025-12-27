"""
Vector storage utilities for the Multilingual RAG Ingestion Pipeline.

This module provides a wrapper around ChromaDB for storing and searching
chunk embeddings. It handles serialization of chunk metadata and provides
filtering capabilities.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

# Try to import chromadb
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

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
    """Name of the Chroma collection."""
    
    persist_directory: Optional[str]
    """Path to persistence directory."""
    
    chunks_by_language: Dict[str, int] = field(default_factory=dict)
    """Count of chunks by language."""
    
    chunks_by_document: Dict[str, int] = field(default_factory=dict)
    """Count of chunks by document."""


class VectorStorage:
    """
    Vector storage using ChromaDB.
    
    Stores chunk embeddings with metadata for similarity search.
    Supports filtering by document, language, and other metadata.
    """
    
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
            collection_name: Name for the Chroma collection.
            embedding_dimension: Expected embedding dimension (for validation).
            
        Raises:
            ImportError: If chromadb is not installed.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required for VectorStorage. "
                "Install it with: pip install chromadb"
            )
        
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # Initialize ChromaDB client
        self._client = self._create_client()
        self._collection = self._get_or_create_collection()
    
    def _create_client(self) -> "chromadb.Client":
        """Create ChromaDB client."""
        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            return chromadb.PersistentClient(path=str(self.persist_directory))
        else:
            # In-memory client
            return chromadb.EphemeralClient()
    
    def _get_or_create_collection(self) -> "chromadb.Collection":
        """Get or create the chunks collection."""
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def _chunk_to_metadata(self, chunk: ChildChunk) -> Dict[str, Any]:
        """
        Convert chunk to metadata dictionary for storage.
        
        Chroma only supports str, int, float, bool in metadata.
        """
        metadata = {
            "document_id": chunk.document_id,
            "language": chunk.language.value,
            "block_type": chunk.block_type.value,
            "token_count": chunk.token_count,
            "position_in_parent": chunk.position_in_parent,
            "created_at": chunk.created_at.isoformat(),
        }
        
        # Optional fields
        if chunk.parent_id:
            metadata["parent_id"] = chunk.parent_id
        
        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number
        
        if chunk.prev_chunk_id:
            metadata["prev_chunk_id"] = chunk.prev_chunk_id
        
        if chunk.next_chunk_id:
            metadata["next_chunk_id"] = chunk.next_chunk_id
        
        if chunk.section_path:
            # Store as JSON string since Chroma doesn't support lists
            metadata["section_path"] = "|".join(chunk.section_path)
        
        return metadata
    
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
        
        self._collection.add(
            ids=[chunk.chunk_id],
            embeddings=[chunk.embedding],
            documents=[chunk.text],
            metadatas=[self._chunk_to_metadata(chunk)],
        )
    
    def add_many(self, chunks: List[ChildChunk]) -> int:
        """
        Add multiple chunks to the vector store.
        
        Args:
            chunks: List of ChildChunks with embeddings.
            
        Returns:
            Number of chunks added.
            
        Raises:
            ValueError: If any chunk has no embedding or wrong dimension.
        """
        if not chunks:
            return 0
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
            
            self._validate_embedding(chunk.embedding, chunk.chunk_id)
            
            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append(self._chunk_to_metadata(chunk))
        
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        
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
            where: Optional metadata filter (Chroma where clause).
            include_embeddings: Whether to include embeddings in results.
            
        Returns:
            List of SearchResult ordered by similarity.
        """
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=include,
        )
        
        search_results = []
        
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [None] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
            embeddings = results.get("embeddings", [[None] * len(ids)])[0] if include_embeddings else [None] * len(ids)
            
            for i, chunk_id in enumerate(ids):
                # Convert distance to similarity score (for cosine: similarity = 1 - distance)
                distance = distances[i] if distances else 0.0
                score = 1.0 - distance
                
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    text=documents[i] or "",
                    distance=distance,
                    score=score,
                    metadata=metadatas[i] or {},
                    embedding=embeddings[i] if include_embeddings else None,
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
        include = ["documents", "metadatas"]
        if include_embedding:
            include.append("embeddings")
        
        results = self._collection.get(
            ids=[chunk_id],
            include=include,
        )
        
        if results and results["ids"]:
            idx = 0
            embedding = None
            if include_embedding and results.get("embeddings"):
                embedding = results["embeddings"][idx]
            
            return self._metadata_to_chunk_data(
                chunk_id=results["ids"][idx],
                text=results["documents"][idx] if results["documents"] else "",
                metadata=results["metadatas"][idx] if results["metadatas"] else {},
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
        
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")
        
        results = self._collection.get(
            ids=chunk_ids,
            include=include,
        )
        
        chunks = []
        
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                embedding = None
                if include_embeddings and results.get("embeddings"):
                    embedding = results["embeddings"][i]
                
                chunks.append(self._metadata_to_chunk_data(
                    chunk_id=chunk_id,
                    text=results["documents"][i] if results["documents"] else "",
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
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
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")
        
        results = self._collection.get(
            where={"document_id": document_id},
            include=include,
        )
        
        chunks = []
        
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                embedding = None
                if include_embeddings and results.get("embeddings"):
                    embedding = results["embeddings"][i]
                
                chunks.append(self._metadata_to_chunk_data(
                    chunk_id=chunk_id,
                    text=results["documents"][i] if results["documents"] else "",
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                    embedding=embedding,
                ))
        
        return chunks
    
    def delete(self, chunk_id: str) -> bool:
        """
        Delete a chunk by ID.
        
        Args:
            chunk_id: Chunk ID to delete.
            
        Returns:
            True if deleted (always returns True for Chroma).
        """
        try:
            self._collection.delete(ids=[chunk_id])
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
            self._collection.delete(ids=chunk_ids)
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
        # First get all chunk IDs for this document
        chunks = self.get_by_document(document_id)
        chunk_ids = [c["chunk_id"] for c in chunks]
        
        if not chunk_ids:
            return 0
        
        return self.delete_many(chunk_ids)
    
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
            self._collection.update(
                ids=[chunk_id],
                embeddings=[embedding],
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
            self._collection.update(
                ids=[chunk_id],
                metadatas=[metadata],
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
        result = self._collection.get(ids=[chunk_id])
        return bool(result and result["ids"])
    
    def count(self) -> int:
        """
        Get total number of chunks in the store.
        
        Returns:
            Number of chunks.
        """
        return self._collection.count()
    
    def count_by_document(self, document_id: str) -> int:
        """
        Get number of chunks for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Number of chunks for the document.
        """
        results = self._collection.get(
            where={"document_id": document_id},
            include=[],  # Don't need data, just count
        )
        return len(results["ids"]) if results and results["ids"] else 0
    
    def list_documents(self) -> List[str]:
        """
        List all unique document IDs in the store.
        
        Returns:
            List of unique document IDs.
        """
        results = self._collection.get(include=["metadatas"])
        
        if not results or not results["metadatas"]:
            return []
        
        document_ids = set()
        for metadata in results["metadatas"]:
            if metadata and "document_id" in metadata:
                document_ids.add(metadata["document_id"])
        
        return list(document_ids)
    
    def get_stats(self) -> VectorStoreStats:
        """
        Get statistics about the vector store.
        
        Returns:
            VectorStoreStats with store information.
        """
        results = self._collection.get(include=["metadatas"])
        
        total_chunks = 0
        chunks_by_language: Dict[str, int] = {}
        chunks_by_document: Dict[str, int] = {}
        
        if results and results["metadatas"]:
            total_chunks = len(results["metadatas"])
            
            for metadata in results["metadatas"]:
                if metadata:
                    # Count by language
                    lang = metadata.get("language", "unknown")
                    chunks_by_language[lang] = chunks_by_language.get(lang, 0) + 1
                    
                    # Count by document
                    doc_id = metadata.get("document_id", "unknown")
                    chunks_by_document[doc_id] = chunks_by_document.get(doc_id, 0) + 1
        
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
        self._collection = self._get_or_create_collection()
        
        return count
    
    def persist(self) -> None:
        """
        Persist the vector store to disk.
        
        Only needed for persistent storage mode.
        """
        if self.persist_directory:
            self._client.persist()
    
    def reset(self) -> None:
        """
        Reset the vector store (delete and recreate collection).
        """
        self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create_collection()


def create_vector_storage_from_config(config: Any) -> VectorStorage:
    """
    Create VectorStorage from a PipelineConfig.
    
    Args:
        config: PipelineConfig instance.
        
    Returns:
        Configured VectorStorage instance.
    """
    return VectorStorage(
        persist_directory=config.chroma_path,
        collection_name="chunks",
        embedding_dimension=1536,  # Cohere multilingual dimension
    )