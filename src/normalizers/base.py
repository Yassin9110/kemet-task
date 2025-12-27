"""
Base normalizer for the Multilingual RAG Ingestion Pipeline.

This module defines the abstract base class for text normalizers
and provides a chain mechanism for applying multiple normalizers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union


class BaseNormalizer(ABC):
    """
    Abstract base class for text normalizers.
    
    All normalizers must inherit from this class and implement
    the normalize() method.
    """
    
    # Subclasses should set this
    NAME: str = "base"
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the normalizer.
        
        Args:
            enabled: Whether this normalizer is active.
        """
        self.enabled = enabled
    
    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Normalize the input text.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        pass
    
    def __call__(self, text: str) -> str:
        """
        Allow normalizer to be called as a function.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text if enabled, original text otherwise.
        """
        if not self.enabled:
            return text
        return self.normalize(text)
    
    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"{self.__class__.__name__}({status})"


class NormalizerChain:
    """
    Chain multiple normalizers together.
    
    Applies normalizers in sequence, passing the output of each
    to the next normalizer in the chain.
    """
    
    def __init__(self, normalizers: Optional[List[BaseNormalizer]] = None):
        """
        Initialize the normalizer chain.
        
        Args:
            normalizers: List of normalizers to apply in order.
        """
        self._normalizers: List[BaseNormalizer] = normalizers or []
    
    def add(self, normalizer: BaseNormalizer) -> "NormalizerChain":
        """
        Add a normalizer to the chain.
        
        Args:
            normalizer: Normalizer to add.
            
        Returns:
            Self for method chaining.
        """
        self._normalizers.append(normalizer)
        return self
    
    def insert(self, index: int, normalizer: BaseNormalizer) -> "NormalizerChain":
        """
        Insert a normalizer at a specific position.
        
        Args:
            index: Position to insert at.
            normalizer: Normalizer to insert.
            
        Returns:
            Self for method chaining.
        """
        self._normalizers.insert(index, normalizer)
        return self
    
    def remove(self, normalizer_name: str) -> bool:
        """
        Remove a normalizer by name.
        
        Args:
            normalizer_name: Name of normalizer to remove.
            
        Returns:
            True if removed, False if not found.
        """
        for i, norm in enumerate(self._normalizers):
            if norm.NAME == normalizer_name:
                self._normalizers.pop(i)
                return True
        return False
    
    def get(self, normalizer_name: str) -> Optional[BaseNormalizer]:
        """
        Get a normalizer by name.
        
        Args:
            normalizer_name: Name of normalizer to get.
            
        Returns:
            Normalizer if found, None otherwise.
        """
        for norm in self._normalizers:
            if norm.NAME == normalizer_name:
                return norm
        return None
    
    def enable(self, normalizer_name: str) -> bool:
        """
        Enable a normalizer by name.
        
        Args:
            normalizer_name: Name of normalizer to enable.
            
        Returns:
            True if found and enabled, False otherwise.
        """
        norm = self.get(normalizer_name)
        if norm:
            norm.enabled = True
            return True
        return False
    
    def disable(self, normalizer_name: str) -> bool:
        """
        Disable a normalizer by name.
        
        Args:
            normalizer_name: Name of normalizer to disable.
            
        Returns:
            True if found and disabled, False otherwise.
        """
        norm = self.get(normalizer_name)
        if norm:
            norm.enabled = False
            return True
        return False
    
    def normalize(self, text: str) -> str:
        """
        Apply all enabled normalizers in sequence.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        result = text
        for normalizer in self._normalizers:
            result = normalizer(result)
        return result
    
    def __call__(self, text: str) -> str:
        """
        Allow chain to be called as a function.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        return self.normalize(text)
    
    def __len__(self) -> int:
        """Get number of normalizers in chain."""
        return len(self._normalizers)
    
    def __iter__(self):
        """Iterate over normalizers."""
        return iter(self._normalizers)
    
    @property
    def normalizer_names(self) -> List[str]:
        """Get list of normalizer names in order."""
        return [n.NAME for n in self._normalizers]
    
    @property
    def enabled_normalizers(self) -> List[str]:
        """Get list of enabled normalizer names."""
        return [n.NAME for n in self._normalizers if n.enabled]
    
    def __repr__(self) -> str:
        names = ", ".join(self.normalizer_names)
        return f"NormalizerChain([{names}])"


class NormalizationResult:
    """
    Result of a normalization operation with metadata.
    """
    
    def __init__(self, original: str, normalized: str, normalizers_applied: List[str]):
        """
        Initialize normalization result.
        
        Args:
            original: Original text before normalization.
            normalized: Text after normalization.
            normalizers_applied: List of normalizer names applied.
        """
        self.original = original
        self.normalized = normalized
        self.normalizers_applied = normalizers_applied
    
    @property
    def was_modified(self) -> bool:
        """Check if text was modified."""
        return self.original != self.normalized
    
    @property
    def original_length(self) -> int:
        """Length of original text."""
        return len(self.original)
    
    @property
    def normalized_length(self) -> int:
        """Length of normalized text."""
        return len(self.normalized)
    
    @property
    def length_difference(self) -> int:
        """Difference in length (negative means shorter)."""
        return self.normalized_length - self.original_length
    
    def __str__(self) -> str:
        return self.normalized
    
    def __repr__(self) -> str:
        modified = "modified" if self.was_modified else "unchanged"
        return f"NormalizationResult({modified}, {len(self.normalizers_applied)} normalizers)"