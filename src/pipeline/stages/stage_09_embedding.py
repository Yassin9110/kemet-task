"""
Stage 9: Embedding

Generates vector embeddings for child chunks using Cohere
embed-multilingual-v3.0 model.
"""

import time
from dataclasses import dataclass
from typing import List, Optional
from logging import Logger

from src.config.settings import PipelineConfig
from src.models.enums import DocumentFormat
from src.models.chunks import ParentChunk, ChildChunk
from src.embedding.base import BaseEmbedder, EmbeddingError
from src.embedding.cohere_embedder import CohereEmbedder


@dataclass
class EmbeddingInput:
    """Input for the embedding stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    parent_chunks: List[ParentChunk]
    child_chunks: List[ChildChunk]
    hierarchy_depth: int
    total_tokens: int


@dataclass
class EmbeddingOutput:
    """Output from the embedding stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    parent_chunks: List[ParentChunk]
    child_chunks: List[ChildChunk]
    hierarchy_depth: int
    total_tokens: int
    embedding_stats: dict


class EmbeddingStage:
    """
    Stage 9: Embedding
    
    Responsibilities:
    - Generate embeddings for child chunks only
    - Batch processing for efficiency
    - Handle API errors with retries
    - Track embedding statistics
    
    Note: Parent chunks are stored as text only (not embedded).
    """
    
    STAGE_NAME = "Embedding"
    STAGE_NUMBER = 9
    TOTAL_STAGES = 11
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the embedding stage.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self._embedder: Optional[BaseEmbedder] = None
    
    @property
    def embedder(self) -> BaseEmbedder:
        """
        Lazy initialization of embedder.
        
        Returns:
            BaseEmbedder instance.
        """
        if self._embedder is None:
            self._embedder = CohereEmbedder(self.config)
        return self._embedder
    
    def execute(
        self,
        input_data: EmbeddingInput,
        logger: Logger
    ) -> EmbeddingOutput:
        """
        Execute the embedding stage.
        
        Args:
            input_data: Embedding input with chunks.
            logger: Logger instance for progress tracking.
            
        Returns:
            EmbeddingOutput with embedded child chunks.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        start_time = time.time()
        
        child_chunks = input_data.child_chunks
        
        # Statistics
        stats = {
            'total_chunks': len(child_chunks),
            'chunks_embedded': 0,
            'chunks_skipped': 0,
            'batches_processed': 0,
            'embedding_dimension': self.embedder.dimension,
            'model': self.embedder.model_name,
        }
        
        if not child_chunks:
            logger.info(
                f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
                f"{self.STAGE_NAME} ✓ (0ms)"
            )
            logger.info("  → No chunks to embed")
            
            return EmbeddingOutput(
                document_id=input_data.document_id,
                stored_path=input_data.stored_path,
                format=input_data.format,
                parent_chunks=input_data.parent_chunks,
                child_chunks=child_chunks,
                hierarchy_depth=input_data.hierarchy_depth,
                total_tokens=input_data.total_tokens,
                embedding_stats=stats
            )
        
        # Extract texts for embedding
        texts_to_embed: List[str] = []
        chunk_indices: List[int] = []
        
        for i, chunk in enumerate(child_chunks):
            if chunk.text and chunk.text.strip():
                texts_to_embed.append(chunk.text)
                chunk_indices.append(i)
            else:
                stats['chunks_skipped'] += 1
        
        # Generate embeddings in batches
        if texts_to_embed:
            try:
                embeddings = self._embed_with_progress(
                    texts=texts_to_embed,
                    logger=logger,
                    stats=stats
                )
                
                # Assign embeddings to chunks
                for idx, embedding in zip(chunk_indices, embeddings):
                    child_chunks[idx].embedding = embedding
                    stats['chunks_embedded'] += 1
                    
            except EmbeddingError as e:
                logger.error(f"  → Embedding failed: {str(e)}")
                raise
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log completion
        logger.info(
            f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
            f"{self.STAGE_NAME} ✓ ({duration_ms}ms)"
        )
        logger.info(
            f"  → Embedded {stats['chunks_embedded']} chunks "
            f"({stats['batches_processed']} batches)"
        )
        
        if stats['chunks_skipped'] > 0:
            logger.warning(
                f"  → Skipped {stats['chunks_skipped']} empty chunks"
            )
        
        return EmbeddingOutput(
            document_id=input_data.document_id,
            stored_path=input_data.stored_path,
            format=input_data.format,
            parent_chunks=input_data.parent_chunks,
            child_chunks=child_chunks,
            hierarchy_depth=input_data.hierarchy_depth,
            total_tokens=input_data.total_tokens,
            embedding_stats=stats
        )
    
    def _embed_with_progress(
        self,
        texts: List[str],
        logger: Logger,
        stats: dict
    ) -> List[List[float]]:
        """
        Generate embeddings with progress tracking.
        
        Args:
            texts: List of texts to embed.
            logger: Logger for progress updates.
            stats: Statistics dictionary to update.
            
        Returns:
            List of embedding vectors.
        """
        batch_size = self.config.embedding_batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        all_embeddings: List[List[float]] = []
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Generate embeddings for batch
            batch_embeddings = self.embedder.embed(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            stats['batches_processed'] += 1
        
        return all_embeddings