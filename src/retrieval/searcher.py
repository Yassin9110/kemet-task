"""
Vector Search

Provides vector similarity search functionality
for retrieving relevant chunks.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from logging import Logger

from src.config.settings import PipelineConfig
from src.models.chunks import ParentChunk, ChildChunk
from src.models.enums import Language
from src.storage.json_storage import JSONStorage
from src.storage.vector_storage import VectorStorage
from src.embedding.cohere_embedder import CohereEmbedder
from src.logging.pipeline_logger import create_logger_from_config


@dataclass
class SearchResult:
    """Single search result with chunk and score."""
    chunk: ChildChunk
    score: float
    parent: Optional[ParentChunk] = None


@dataclass
class SearchResponse:
    """Complete search response."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: int


class Searcher:
    """
    Vector similarity search for RAG retrieval.
    
    Provides functionality to search for relevant chunks
    based on query similarity.
    
    Usage:
        searcher = Searcher(config)
        results = searcher.search("What is machine learning?", top_k=5)
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the searcher.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.logger = create_logger_from_config(config)
        self.json_storage = JSONStorage(config)
        self.vector_storage = VectorStorage(config)
        self._embedder: Optional[CohereEmbedder] = None
    
    @property
    def embedder(self) -> CohereEmbedder:
        """
        Lazy initialization of embedder.
        
        Returns:
            CohereEmbedder instance.
        """
        if self._embedder is None:
            self._embedder = CohereEmbedder(self.config)
        return self._embedder
    
    def search(self, query: str, top_k: int = 5, document_id: Optional[str] = None, language: Optional[Language] = None, include_parents: bool = True) -> SearchResponse:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            document_id: Optional filter by document ID.
            language: Optional filter by language.
            include_parents: Whether to include parent chunks in results.
            
        Returns:
            SearchResponse with results and metadata.
        """
        import time
        start_time = time.time()
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Build filters
        filters: Dict = {}
        if document_id:
            filters['document_id'] = document_id
        if language:
            filters['language'] = language.value
        
        # Search vector storage
        chunks, scores = self.vector_storage.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters if filters else None
        )
        
        # Build results
        results: List[SearchResult] = []
        
        # Load parents if needed
        parents_map: Dict[str, ParentChunk] = {}
        if include_parents:
            parents_map = self._load_parents_map(chunks)
        
        for chunk, score in zip(chunks, scores):
            parent = None
            if include_parents and chunk.parent_id:
                parent = parents_map.get(chunk.parent_id)
            
            results.append(SearchResult(
                chunk=chunk,
                score=score,
                parent=parent
            ))
        
        # Calculate search time
        search_time_ms = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms
        )
    
    def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
        exclude_chunk_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search using a pre-computed embedding.
        
        Args:
            embedding: Query embedding vector.
            top_k: Number of results to return.
            document_id: Optional filter by document ID.
            exclude_chunk_ids: Chunk IDs to exclude from results.
            
        Returns:
            List of search results.
        """
        # Build filters
        filters: Dict = {}
        if document_id:
            filters['document_id'] = document_id
        
        # Search vector storage
        chunks, scores = self.vector_storage.search(
            query_embedding=embedding,
            top_k=top_k + len(exclude_chunk_ids or []),  # Get extra to account for exclusions
            filters=filters if filters else None
        )
        
        # Filter excluded chunks
        results: List[SearchResult] = []
        exclude_set = set(exclude_chunk_ids or [])
        
        for chunk, score in zip(chunks, scores):
            if chunk.chunk_id not in exclude_set:
                results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    parent=None
                ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_similar_chunks(
        self,
        chunk_id: str,
        top_k: int = 5,
        same_document_only: bool = True
    ) -> List[SearchResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: Source chunk ID.
            top_k: Number of results to return.
            same_document_only: Whether to limit to same document.
            
        Returns:
            List of similar chunks.
        """
        # Get the source chunk
        source_chunk = self._get_chunk_by_id(chunk_id)
        if not source_chunk or not source_chunk.embedding:
            return []
        
        # Search using chunk's embedding
        document_filter = source_chunk.document_id if same_document_only else None
        
        return self.search_by_embedding(
            embedding=source_chunk.embedding,
            top_k=top_k,
            document_id=document_filter,
            exclude_chunk_ids=[chunk_id]
        )
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        keyword_weight: float = 0.3
    ) -> SearchResponse:
        """
        Hybrid search combining vector and keyword matching.
        
        This is a simple implementation that boosts results
        containing query keywords.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            document_id: Optional filter by document ID.
            keyword_weight: Weight for keyword matching (0-1).
            
        Returns:
            SearchResponse with reranked results.
        """
        import time
        start_time = time.time()
        
        # Get more results than needed for reranking
        vector_results = self.search(
            query=query,
            top_k=top_k * 2,
            document_id=document_id,
            include_parents=True
        )
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        # Rerank based on keyword presence
        reranked_results: List[SearchResult] = []
        
        for result in vector_results.results:
            keyword_score = self._compute_keyword_score(
                text=result.chunk.text,
                keywords=keywords
            )
            
            # Combine scores
            combined_score = (
                (1 - keyword_weight) * result.score +
                keyword_weight * keyword_score
            )
            
            reranked_results.append(SearchResult(
                chunk=result.chunk,
                score=combined_score,
                parent=result.parent
            ))
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Take top_k
        final_results = reranked_results[:top_k]
        
        # Calculate search time
        search_time_ms = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            query=query,
            results=final_results,
            total_results=len(final_results),
            search_time_ms=search_time_ms
        )
    
    def _load_parents_map(
        self,
        chunks: List[ChildChunk]
    ) -> Dict[str, ParentChunk]:
        """
        Load parent chunks for a list of child chunks.
        
        Args:
            chunks: List of child chunks.
            
        Returns:
            Dictionary mapping parent_id to ParentChunk.
        """
        parent_ids = {
            chunk.parent_id
            for chunk in chunks
            if chunk.parent_id is not None
        }
        
        if not parent_ids:
            return {}
        
        try:
            all_parents = self.json_storage.load_parents()
            return {
                parent.parent_id: parent
                for parent in all_parents
                if parent.parent_id in parent_ids
            }
        except FileNotFoundError:
            return {}
    
    def _get_chunk_by_id(self, chunk_id: str) -> Optional[ChildChunk]:
        """
        Get a chunk by its ID.
        
        Args:
            chunk_id: Chunk ID.
            
        Returns:
            ChildChunk or None if not found.
        """
        try:
            children = self.json_storage.load_children()
            for chunk in children:
                if chunk.chunk_id == chunk_id:
                    return chunk
            return None
        except FileNotFoundError:
            return None
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from a query.
        
        Simple implementation that removes common stop words.
        
        Args:
            query: Query text.
            
        Returns:
            List of keywords.
        """
        # Simple stop words list
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'it', 'its', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'they',
            'them', 'their', 'theirs', 'themselves',
            # Arabic stop words
            'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه',
            'التي', 'الذي', 'التى', 'الذى', 'ان', 'أن', 'كان',
            'كانت', 'لا', 'ما', 'هو', 'هي', 'و', 'أو', 'ثم',
        }
        
        # Tokenize and filter
        words = query.lower().split()
        keywords = [
            word.strip('.,!?;:()[]{}"\'-')
            for word in words
            if word.lower() not in stop_words and len(word) > 2
        ]
        
        return keywords
    
    def _compute_keyword_score(
        self,
        text: str,
        keywords: List[str]
    ) -> float:
        """
        Compute keyword matching score.
        
        Args:
            text: Text to search in.
            keywords: Keywords to match.
            
        Returns:
            Score between 0 and 1.
        """
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        
        return matches / len(keywords)