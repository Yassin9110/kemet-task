"""
Embedding module for generating vector representations of text.
"""

from src.embedding.base import BaseEmbedder, EmbeddingError
from src.embedding.cohere_embedder import CohereEmbedder

__all__ = [
    "BaseEmbedder",
    "EmbeddingError", 
    "CohereEmbedder",
]