"""
Abstract base class for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding providers.
    
    All embedding implementations must inherit from this class
    and implement the required methods.
    """
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors, one per input text.
            Each embedding is a list of floats.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        pass
    
    @abstractmethod
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
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.
        
        Returns:
            Integer representing embedding dimension.
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the name of the embedding model.
        
        Returns:
            String identifier for the model.
        """
        pass


class EmbeddingError(Exception):
    """
    Exception raised when embedding generation fails.
    """
    
    def __init__(self, message: str, original_error: Exception = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)
    
    def __str__(self):
        if self.original_error:
            return f"{self.message} | Original error: {self.original_error}"
        return self.message