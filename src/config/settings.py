"""
Configuration settings for the Multilingual RAG Ingestion Pipeline.

This module defines the PipelineConfig dataclass containing all
configurable parameters for the pipeline, with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    """
    Central configuration for the entire ingestion pipeline.
    
    All parameters have defaults and can be overridden at instantiation.
    Paths are computed automatically from base directories.
    """

    # Token Limits
    max_tokens: int = 512
    """Maximum tokens per parent chunk."""
    
    min_tokens: int = 80
    """Minimum tokens — below this, merge or skip parent-child split."""
    
    child_target_tokens: int = 128
    """Target size for child chunks."""
    
    # Overlap Settings
    overlap_tokens: int = 50
    """Token overlap between consecutive chunks (English)."""
    
    arabic_overlap_multiplier: float = 1.2
    """Multiplier for Arabic text overlap (default: 50 × 1.2 = 60)."""
    
    # Hierarchy Thresholds
    small_doc_threshold: int = 1500
    """Documents below this token count: 1 level (chunks only)."""
    
    large_doc_threshold: int = 10000
    """Documents above this token count: 3 levels (doc → section → chunk)."""
    
    # Embedding Settings
    embedding_model: str = "embed-multilingual-v3.0"
    """Cohere embedding model name."""
    
    embedding_batch_size: int = 32
    """Number of chunks per embedding API call."""
    
    cohere_api_key: str = ""
    """Cohere API key. If empty, will attempt to load from COHERE_API_KEY env var."""
    
    # Semantic Edges
    compute_semantic_edges: bool = True
    """Whether to compute similarity edges between chunks."""
    
    semantic_similarity_threshold: float = 0.85
    """Minimum cosine similarity for creating a semantic edge."""
    
    # Storage Paths
    base_data_path: str = "./data"
    """Root directory for all data storage."""
    
    raw_files_dir: str = "raw"
    """Subdirectory name for raw file storage."""
    
    chroma_dir: str = "chroma"
    """Subdirectory name for Chroma vector database."""
    
    documents_filename: str = "documents.json"
    """Filename for documents registry."""
    
    parents_filename: str = "parents.json"
    """Filename for parent chunks storage."""
    
    children_filename: str = "children.json"
    """Filename for child chunks backup."""
    
    edges_filename: str = "edges.json"
    """Filename for semantic edges storage."""
    
    # Processing Settings
    supported_formats: List[str] = field( default_factory=lambda: ["pdf", "docx", "html", "md", "txt"])
    """List of allowed file extensions."""
    
    max_file_size_mb: int = 20
    """Maximum allowed file size in megabytes."""
    
    # Logging Settings
    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""
    
    log_to_file: bool = True
    """Whether to write logs to file in addition to console."""
    
    log_dir: str = "./logs"
    """Directory for log files."""
    
    log_filename: str = "pipeline.log"
    """Log file name."""
    
    # Computed Properties
    @property
    def raw_files_path(self) -> str:
        """Full path to raw files directory."""
        return os.path.join(self.base_data_path, self.raw_files_dir)
    
    @property
    def chroma_path(self) -> str:
        """Full path to Chroma database directory."""
        return os.path.join(self.base_data_path, self.chroma_dir)
    
    @property
    def documents_path(self) -> str:
        """Full path to documents JSON file."""
        return os.path.join(self.base_data_path, self.documents_filename)
    
    @property
    def parents_path(self) -> str:
        """Full path to parents JSON file."""
        return os.path.join(self.base_data_path, self.parents_filename)
    
    @property
    def children_path(self) -> str:
        """Full path to children JSON file."""
        return os.path.join(self.base_data_path, self.children_filename)
    
    @property
    def edges_path(self) -> str:
        """Full path to edges JSON file."""
        return os.path.join(self.base_data_path, self.edges_filename)
    
    @property
    def log_file_path(self) -> str:
        """Full path to log file."""
        return os.path.join(self.log_dir, self.log_filename)
    
    @property
    def arabic_overlap_tokens(self) -> int:
        """Computed overlap for Arabic text."""
        return int(self.overlap_tokens * self.arabic_overlap_multiplier)
    
    @property
    def max_file_size_bytes(self) -> int:
        """Maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    # Methods
    def get_api_key(self) -> str:
        """
        Get the Cohere API key.
        
        Returns the configured key if set, otherwise attempts to load
        from the COHERE_API_KEY environment variable.
        
        Returns:
            str: The API key.
            
        Raises:
            ValueError: If no API key is configured or found in environment.
        """
        if self.cohere_api_key:
            return self.cohere_api_key
        
        env_key = os.environ.get("COHERE_API_KEY", "")
        if env_key:
            return env_key
        
        raise ValueError(
            "No Cohere API key configured. "
            "Set 'cohere_api_key' in config or COHERE_API_KEY environment variable."
        )
    
    def get_overlap_for_language(self, language: str) -> int:
        """
        Get the appropriate overlap tokens for a given language.
        
        Args:
            language: Language code ('ar', 'en', or 'mixed').
            
        Returns:
            int: Number of overlap tokens to use.
        """
        if language in ("ar", "mixed"):
            return self.arabic_overlap_tokens
        return self.overlap_tokens
    
    
    def get_hierarchy_depth(self, total_tokens: int) -> int:
        """
        Determine hierarchy depth based on document size.
        
        Args:
            total_tokens: Total token count of the document.
            
        Returns:
            int: Hierarchy depth (1, 2, or 3).
        """
        if total_tokens < self.small_doc_threshold:
            return 1
        elif total_tokens < self.large_doc_threshold:
            return 2
        else:
            return 3
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid.
        """
        errors = []
        
        # Token limits validation
        if self.max_tokens <= 0:
            errors.append("max_tokens must be positive")
        if self.min_tokens <= 0:
            errors.append("min_tokens must be positive")
        if self.child_target_tokens <= 0:
            errors.append("child_target_tokens must be positive")
        if self.min_tokens >= self.max_tokens:
            errors.append("min_tokens must be less than max_tokens")
        if self.child_target_tokens >= self.max_tokens:
            errors.append("child_target_tokens must be less than max_tokens")
        
        # Overlap validation
        if self.overlap_tokens < 0:
            errors.append("overlap_tokens cannot be negative")
        if self.arabic_overlap_multiplier <= 0:
            errors.append("arabic_overlap_multiplier must be positive")
        
        # Threshold validation
        if self.small_doc_threshold <= 0:
            errors.append("small_doc_threshold must be positive")
        if self.large_doc_threshold <= 0:
            errors.append("large_doc_threshold must be positive")
        if self.small_doc_threshold >= self.large_doc_threshold:
            errors.append("small_doc_threshold must be less than large_doc_threshold")
        
        # Embedding validation
        if self.embedding_batch_size <= 0:
            errors.append("embedding_batch_size must be positive")
        
        # Semantic edges validation
        if not (0 <= self.semantic_similarity_threshold <= 1):
            errors.append("semantic_similarity_threshold must be between 0 and 1")
        
        # File size validation
        if self.max_file_size_mb <= 0:
            errors.append("max_file_size_mb must be positive")
        
        # Log level validation
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"log_level must be one of: {valid_log_levels}")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))