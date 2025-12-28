"""
Context Expander

Provides functionality to expand retrieved chunks
with parent context and sibling chunks.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from logging import Logger

from src.config.settings import PipelineConfig
from src.models.chunks import ParentChunk, ChildChunk
from src.models.edges import SemanticEdge
from src.storage.json_storage import JSONStorage
from src.storage.vector_storage import VectorStorage
from src.logging.pipeline_logger import create_logger_from_config


@dataclass
class ExpandedContext:
    """Expanded context for a chunk."""
    chunk: ChildChunk
    parent: Optional[ParentChunk]
    prev_sibling: Optional[ChildChunk]
    next_sibling: Optional[ChildChunk]
    semantic_neighbors: List[ChildChunk]
    section_path: List[str]
    context_text: str  # Combined text for LLM


@dataclass
class ExpandedSearchResult:
    """Search result with expanded context."""
    chunk: ChildChunk
    score: float
    expanded_context: ExpandedContext


class ContextExpander:
    """
    Expands retrieved chunks with surrounding context.
    
    Provides functionality to:
    - Get parent chunk content
    - Get sibling chunks (prev/next)
    - Get semantically similar chunks
    - Build combined context for LLM
    
    Usage:
        expander = ContextExpander(config)
        context = expander.expand(chunk_id)
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the context expander.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.logger = create_logger_from_config(config)
        self.json_storage = JSONStorage(config)
        self.vector_storage = VectorStorage(config)
        
        # Cache for frequently accessed data
        self._chunks_cache: Optional[Dict[str, ChildChunk]] = None
        self._parents_cache: Optional[Dict[str, ParentChunk]] = None
        self._edges_cache: Optional[List[SemanticEdge]] = None
    
    def expand(
        self,
        chunk_id: str,
        include_siblings: bool = True,
        include_semantic: bool = True,
        max_semantic_neighbors: int = 3
    ) -> Optional[ExpandedContext]:
        """
        Expand a chunk with surrounding context.
        
        Args:
            chunk_id: Chunk ID to expand.
            include_siblings: Whether to include sibling chunks.
            include_semantic: Whether to include semantic neighbors.
            max_semantic_neighbors: Maximum semantic neighbors to include.
            
        Returns:
            ExpandedContext or None if chunk not found.
        """
        # Get the chunk
        chunk = self._get_chunk(chunk_id)
        if not chunk:
            return None
        
        # Get parent
        parent = None
        if chunk.parent_id:
            parent = self._get_parent(chunk.parent_id)
        
        # Get siblings
        prev_sibling = None
        next_sibling = None
        if include_siblings:
            if chunk.prev_chunk_id:
                prev_sibling = self._get_chunk(chunk.prev_chunk_id)
            if chunk.next_chunk_id:
                next_sibling = self._get_chunk(chunk.next_chunk_id)
        
        # Get semantic neighbors
        semantic_neighbors: List[ChildChunk] = []
        if include_semantic:
            semantic_neighbors = self._get_semantic_neighbors(
                chunk_id=chunk_id,
                document_id=chunk.document_id,
                max_neighbors=max_semantic_neighbors
            )
        
        # Build combined context text
        context_text = self._build_context_text(
            chunk=chunk,
            parent=parent,
            prev_sibling=prev_sibling,
            next_sibling=next_sibling,
            semantic_neighbors=semantic_neighbors
        )
        
        return ExpandedContext(
            chunk=chunk,
            parent=parent,
            prev_sibling=prev_sibling,
            next_sibling=next_sibling,
            semantic_neighbors=semantic_neighbors,
            section_path=chunk.section_path,
            context_text=context_text
        )
    
    def expand_multiple(
        self,
        chunk_ids: List[str],
        include_siblings: bool = True,
        include_semantic: bool = False,
        deduplicate: bool = True
    ) -> List[ExpandedContext]:
        """
        Expand multiple chunks with context.
        
        Args:
            chunk_ids: List of chunk IDs to expand.
            include_siblings: Whether to include sibling chunks.
            include_semantic: Whether to include semantic neighbors.
            deduplicate: Whether to remove duplicate context chunks.
            
        Returns:
            List of expanded contexts.
        """
        # Load caches for efficiency
        self._load_caches()
        
        results: List[ExpandedContext] = []
        seen_chunk_ids: set = set()
        
        for chunk_id in chunk_ids:
            if deduplicate and chunk_id in seen_chunk_ids:
                continue
            
            context = self.expand(
                chunk_id=chunk_id,
                include_siblings=include_siblings,
                include_semantic=include_semantic
            )
            
            if context:
                results.append(context)
                seen_chunk_ids.add(chunk_id)
                
                # Track siblings to avoid duplicates
                if context.prev_sibling:
                    seen_chunk_ids.add(context.prev_sibling.chunk_id)
                if context.next_sibling:
                    seen_chunk_ids.add(context.next_sibling.chunk_id)
        
        return results
    
    def get_full_section_context(
        self,
        chunk_id: str
    ) -> Optional[str]:
        """
        Get the full text of the section containing a chunk.
        
        This returns the parent chunk text if available,
        otherwise returns the chunk with its siblings.
        
        Args:
            chunk_id: Chunk ID.
            
        Returns:
            Full section text or None if not found.
        """
        chunk = self._get_chunk(chunk_id)
        if not chunk:
            return None
        
        # If parent exists, return parent text
        if chunk.parent_id:
            parent = self._get_parent(chunk.parent_id)
            if parent:
                return parent.text
        
        # Otherwise, combine chunk with siblings
        texts: List[str] = []
        
        # Get all previous siblings
        prev_chunk = chunk
        while prev_chunk.prev_chunk_id:
            prev_chunk = self._get_chunk(prev_chunk.prev_chunk_id)
            if prev_chunk:
                texts.insert(0, prev_chunk.text)
            else:
                break
        
        # Add current chunk
        texts.append(chunk.text)
        
        # Get all next siblings
        next_chunk = chunk
        while next_chunk.next_chunk_id:
            next_chunk = self._get_chunk(next_chunk.next_chunk_id)
            if next_chunk:
                texts.append(next_chunk.text)
            else:
                break
        
        return "\n\n".join(texts)
    
    def get_document_outline(
        self,
        document_id: str
    ) -> List[Dict]:
        """
        Get document outline based on section structure.
        
        Args:
            document_id: Document ID.
            
        Returns:
            List of section dictionaries with title and path.
        """
        try:
            parents = self.json_storage.load_parents(document_id=document_id)
        except FileNotFoundError:
            return []
        
        outline: List[Dict] = []
        seen_paths: set = set()
        
        for parent in parents:
            path_key = " > ".join(parent.section_path)
            
            if path_key not in seen_paths:
                outline.append({
                    'path': parent.section_path,
                    'title': parent.section_path[-1] if parent.section_path else "Root",
                    'depth': len(parent.section_path),
                    'token_count': parent.token_count,
                    'language': parent.language.value
                })
                seen_paths.add(path_key)
        
        return outline
    
    def build_retrieval_context(
        self,
        chunks: List[ChildChunk],
        max_tokens: Optional[int] = None,
        include_section_headers: bool = True
    ) -> str:
        """
        Build a combined context string from multiple chunks.
        
        Suitable for passing to an LLM as context.
        
        Args:
            chunks: List of chunks to combine.
            max_tokens: Maximum tokens in output (approximate).
            include_section_headers: Whether to include section path headers.
            
        Returns:
            Combined context string.
        """
        if not chunks:
            return ""
        
        context_parts: List[str] = []
        current_section: List[str] = []
        
        for chunk in chunks:
            # Add section header if changed
            if include_section_headers and chunk.section_path != current_section:
                if chunk.section_path:
                    header = " > ".join(chunk.section_path)
                    context_parts.append(f"\n[{header}]\n")
                current_section = chunk.section_path
            
            context_parts.append(chunk.text)
        
        combined = "\n\n".join(context_parts)
        
        # Truncate if needed (rough approximation)
        if max_tokens:
            # Approximate 4 chars per token
            max_chars = max_tokens * 4
            if len(combined) > max_chars:
                combined = combined[:max_chars] + "..."
        
        return combined
    
    def _get_chunk(self, chunk_id: str) -> Optional[ChildChunk]:
        """
        Get a chunk by ID, using cache if available.
        
        Args:
            chunk_id: Chunk ID.
            
        Returns:
            ChildChunk or None.
        """
        if self._chunks_cache is not None:
            return self._chunks_cache.get(chunk_id)
        
        try:
            children = self.json_storage.load_children()
            for chunk in children:
                if chunk.chunk_id == chunk_id:
                    return chunk
            return None
        except FileNotFoundError:
            return None
    
    def _get_parent(self, parent_id: str) -> Optional[ParentChunk]:
        """
        Get a parent by ID, using cache if available.
        
        Args:
            parent_id: Parent ID.
            
        Returns:
            ParentChunk or None.
        """
        if self._parents_cache is not None:
            return self._parents_cache.get(parent_id)
        
        try:
            parents = self.json_storage.load_parents()
            for parent in parents:
                if parent.parent_id == parent_id:
                    return parent
            return None
        except FileNotFoundError:
            return None
    
    def _get_semantic_neighbors(
        self,
        chunk_id: str,
        document_id: str,
        max_neighbors: int
    ) -> List[ChildChunk]:
        """
        Get semantically similar chunks from stored edges.
        
        Args:
            chunk_id: Source chunk ID.
            document_id: Document ID for filtering.
            max_neighbors: Maximum neighbors to return.
            
        Returns:
            List of semantically similar chunks.
        """
        try:
            edges = self._edges_cache or self.json_storage.load_edges()
        except FileNotFoundError:
            return []
        
        # Find edges involving this chunk
        neighbor_ids: List[tuple[str, float]] = []
        
        for edge in edges:
            if edge.source_chunk_id == chunk_id:
                neighbor_ids.append((edge.target_chunk_id, edge.similarity_score))
            elif edge.target_chunk_id == chunk_id:
                neighbor_ids.append((edge.source_chunk_id, edge.similarity_score))
        
        # Sort by similarity score
        neighbor_ids.sort(key=lambda x: x[1], reverse=True)
        
        # Get chunk objects
        neighbors: List[ChildChunk] = []
        for neighbor_id, _ in neighbor_ids[:max_neighbors]:
            chunk = self._get_chunk(neighbor_id)
            if chunk and chunk.document_id == document_id:
                neighbors.append(chunk)
        
        return neighbors
    
    def _build_context_text(
        self,
        chunk: ChildChunk,
        parent: Optional[ParentChunk],
        prev_sibling: Optional[ChildChunk],
        next_sibling: Optional[ChildChunk],
        semantic_neighbors: List[ChildChunk]
    ) -> str:
        """
        Build combined context text for LLM.
        
        Args:
            chunk: Main chunk.
            parent: Parent chunk.
            prev_sibling: Previous sibling chunk.
            next_sibling: Next sibling chunk.
            semantic_neighbors: Semantically similar chunks.
            
        Returns:
            Combined context string.
        """
        parts: List[str] = []
        
        # Add section path
        if chunk.section_path:
            path_str = " > ".join(chunk.section_path)
            parts.append(f"[Section: {path_str}]")
        
        # Option 1: Use parent text if available (more complete context)
        if parent:
            parts.append(f"\n{parent.text}")
        else:
            # Option 2: Combine siblings
            if prev_sibling:
                parts.append(f"\n[Previous context]\n{prev_sibling.text}")
            
            parts.append(f"\n[Main content]\n{chunk.text}")
            
            if next_sibling:
                parts.append(f"\n[Following context]\n{next_sibling.text}")
        
        # Add semantic neighbors
        if semantic_neighbors:
            parts.append("\n[Related content]")
            for neighbor in semantic_neighbors:
                parts.append(f"\n{neighbor.text}")
        
        return "\n".join(parts)
    
    def _load_caches(self) -> None:
        """Load data into caches for batch operations."""
        try:
            children = self.json_storage.load_children()
            self._chunks_cache = {c.chunk_id: c for c in children}
        except FileNotFoundError:
            self._chunks_cache = {}
        
        try:
            parents = self.json_storage.load_parents()
            self._parents_cache = {p.parent_id: p for p in parents}
        except FileNotFoundError:
            self._parents_cache = {}
        
        try:
            self._edges_cache = self.json_storage.load_edges()
        except FileNotFoundError:
            self._edges_cache = []
    
    def clear_caches(self) -> None:
        """Clear internal caches."""
        self._chunks_cache = None
        self._parents_cache = None
        self._edges_cache = None