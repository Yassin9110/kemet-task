"""
Cohere embedding implementation using embed-multilingual-v3.0.
"""

import time
from typing import List, Optional

import cohere

from src.embedding.base import BaseEmbedder, EmbeddingError
from src.config.settings import PipelineConfig


class CohereEmbedder(BaseEmbedder):
    """
    Embedding provider using Cohere's embed-multilingual-v3.0 model.
    
    This embedder supports both Arabic and English text and is optimized
    for multilingual retrieval tasks.
    """

    # Cohere embed-multilingual-v3.0 produces 1536-dimensional vectors
    EMBEDDING_DIMENSION = 1536
    
    # Maximum texts per API call (Cohere limit is 96)
    MAX_BATCH_SIZE = 96
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the Cohere embedder.
        
        Args:
            config: Pipeline configuration containing API key and settings.
            
        Raises:
            EmbeddingError: If API key is missing or invalid.
        """
        self.config = config
        self._model_name = "embed-v4.0"
        
        if not config.cohere_api_key:
            raise EmbeddingError("Cohere API key is required but not provided in config")
        
        try:
            self.client = cohere.Client(api_key=config.cohere_api_key)
        except Exception as e:
            raise EmbeddingError("Failed to initialize Cohere client", e)
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension (1024 for embed-multilingual-v3.0)."""
        return self.EMBEDDING_DIMENSION
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Handles batching automatically if input exceeds maximum batch size.
        Implements retry logic for transient API failures.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors, one per input text.
            
        Raises:
            EmbeddingError: If embedding generation fails after retries.
        """
        if not texts:
            return []
        
        # Filter out empty strings and track their positions
        non_empty_texts = []
        non_empty_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(i)
        
        if not non_empty_texts:
            # All texts were empty, return zero vectors
            return [[0.0] * self.EMBEDDING_DIMENSION for _ in texts]
        
        # Process in batches
        all_embeddings = []
        batch_size = min(self.config.embedding_batch_size, self.MAX_BATCH_SIZE)
        
        for i in range(0, len(non_empty_texts), batch_size):
            batch = non_empty_texts[i:i + batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Reconstruct full result with zero vectors for empty texts
        result = [[0.0] * self.EMBEDDING_DIMENSION for _ in texts]
        for idx, embedding in zip(non_empty_indices, all_embeddings):
            result[idx] = embedding
        
        return result
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed.
            
        Returns:
            Embedding vector as a list of floats.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not text or not text.strip():
            return [0.0] * self.EMBEDDING_DIMENSION
        
        embeddings = self.embed([text])
        return embeddings[0]
    
    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with retry logic.
        
        Args:
            texts: List of non-empty text strings to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            EmbeddingError: If all retry attempts fail.
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return self._call_cohere_api(texts)
            except cohere.errors.TooManyRequestsError as e:
                # Rate limiting - wait longer
                last_error = e
                wait_time = self.RETRY_DELAY_SECONDS * attempt * 2
                time.sleep(wait_time)
            except (cohere.errors.ServiceUnavailableError, 
                    cohere.errors.InternalServerError) as e:
                # Transient server errors - retry
                last_error = e
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_SECONDS * attempt)
            except cohere.errors.BadRequestError as e:
                # Client error - don't retry
                raise EmbeddingError(f"Invalid request to Cohere API: {str(e)}", e)
            except cohere.errors.UnauthorizedError as e:
                # Auth error - don't retry
                raise EmbeddingError("Cohere API authentication failed. Check your API key.", e)
            except Exception as e:
                # Unknown error - retry once then fail
                last_error = e
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_SECONDS)
        
        raise EmbeddingError(
            f"Failed to generate embeddings after {self.MAX_RETRIES} attempts",
            last_error
        )
    
    def _call_cohere_api(self, texts: List[str]) -> List[List[float]]:
        """
        Make the actual API call to Cohere.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        response = self.client.embed(
            texts=texts,
            model=self._model_name,
            input_type="search_document",  # For documents being indexed
            truncate="END"  # Truncate from end if text exceeds model limit
        )
        
        return [list(embedding) for embedding in response.embeddings]
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Uses input_type="search_query" for better retrieval performance.
        This should be used when embedding queries for search, not documents.
        
        Args:
            query: Query text string to embed.
            
        Returns:
            Embedding vector as a list of floats.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not query or not query.strip():
            return [0.0] * self.EMBEDDING_DIMENSION
        
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.embed(
                    texts=[query],
                    model=self._model_name,
                    input_type="search_query",  # Optimized for queries
                    truncate="END"
                )
                return list(response.embeddings[0])
            except (cohere.errors.TooManyRequestsError,
                    cohere.errors.ServiceUnavailableError,
                    cohere.errors.InternalServerError) as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    wait_time = self.RETRY_DELAY_SECONDS * attempt
                    time.sleep(wait_time)
            except cohere.errors.BadRequestError as e:
                raise EmbeddingError(f"Invalid query request to Cohere API: {str(e)}", e)
            except cohere.errors.UnauthorizedError as e:
                raise EmbeddingError("Cohere API authentication failed. Check your API key.", e)
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_SECONDS)
        
        raise EmbeddingError(
            f"Failed to generate query embedding after {self.MAX_RETRIES} attempts",
            last_error
        )