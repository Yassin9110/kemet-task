"""
Parent chunker for the Multilingual RAG Ingestion Pipeline.

Creates parent chunks from normalized blocks using structure-aware
splitting based on headings and size constraints.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..models.blocks import NormalizedBlock
from ..models.chunks import ParentChunk
from ..models.enums import Language, BlockType
from ..pipeline.helpers.id_generator import generate_parent_id
from ..pipeline.helpers.tokenizer import count_tokens
from ..pipeline.stages.stage_07_structure_parsing import StructuredBlock


@dataclass
class ParentChunkCandidate:
    """Intermediate representation during parent chunk creation."""
    
    blocks: List[StructuredBlock] = field(default_factory=list)
    section_path: List[str] = field(default_factory=list)
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    
    def add_block(self, block: StructuredBlock) -> None:
        """Add a block to this candidate."""
        self.blocks.append(block)
        
        # Track page range
        if block.page_number is not None:
            if self.start_page is None:
                self.start_page = block.page_number
            self.end_page = block.page_number
    
    @property
    def text(self) -> str:
        """Combined text of all blocks."""
        return "\n\n".join(b.normalized_text for b in self.blocks)
    
    @property
    def is_empty(self) -> bool:
        """Check if candidate has no blocks."""
        return len(self.blocks) == 0
    
    @property
    def page_range(self) -> Optional[Tuple[int, int]]:
        """Get page range as tuple."""
        if self.start_page is not None and self.end_page is not None:
            return (self.start_page, self.end_page)
        return None


class ParentChunker:
    """
    Creates parent chunks from normalized blocks.
    
    Strategy:
    - If headings available: Use headings as split points (section-aware)
    - If section too large: Split at sentence boundaries near max_tokens
    - If section too small: Keep as single chunk (no parent-child split)
    - Tables are kept as atomic units
    """
    
    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 80,
        overlap_tokens: int = 0
    ):
        """
        Initialize parent chunker.
        
        Args:
            max_tokens: Maximum tokens per parent chunk.
            min_tokens: Minimum tokens for parent-child split.
            overlap_tokens: Token overlap between parents (usually 0).
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
    
    def create_parents(
        self,
        blocks: List[StructuredBlock],
        document_id: str,
        hierarchy_depth: int
    ) -> List[ParentChunk]:
        """
        Create parent chunks from normalized blocks.
        
        Args:
            blocks: List of normalized blocks.
            document_id: Document ID for reference.
            hierarchy_depth: Hierarchy depth (1, 2, or 3).
            
        Returns:
            List of ParentChunk objects.
        """
        if not blocks:
            return []
        
        # For 1-level hierarchy, no parents needed
        if hierarchy_depth == 1:
            return []
        
        # Check if document has heading structure
        has_headings = any(b.structural_hint == BlockType.HEADING for b in blocks)
        
        if has_headings:
            return self._create_structure_aware_parents(blocks, document_id)
        else:
            return self._create_fixed_window_parents(blocks, document_id)
    
    def _create_structure_aware_parents(
        self,
        blocks: List[StructuredBlock],
        document_id: str
    ) -> List[ParentChunk]:
        """
        Create parents using heading structure.
        
        Args:
            blocks: List of normalized blocks.
            document_id: Document ID.
            
        Returns:
            List of ParentChunk objects.
        """
        parents = []
        current_candidate = ParentChunkCandidate()
        current_section_path: List[str] = []
        
        for block in blocks:
            if block.structural_hint == BlockType.HEADING:
                # Finalize current candidate if not empty
                if not current_candidate.is_empty:
                    parent = self._finalize_candidate(
                        current_candidate, document_id, parents
                    )
                    if parent:
                        parents.extend(parent if isinstance(parent, list) else [parent])
                
                # Update section path based on heading level
                current_section_path = self._update_section_path(
                    current_section_path,
                    block.normalized_text,
                    block.heading_level or 1
                )
                
                # Start new candidate
                current_candidate = ParentChunkCandidate(
                    section_path=current_section_path.copy()
                )
                current_candidate.add_block(block)
            
            else:
                # Add to current candidate
                current_candidate.add_block(block)
        
        # Finalize last candidate
        if not current_candidate.is_empty:
            parent = self._finalize_candidate(
                current_candidate, document_id, parents
            )
            if parent:
                parents.extend(parent if isinstance(parent, list) else [parent])
        
        return parents
    
    def _create_fixed_window_parents(
        self,
        blocks: List[StructuredBlock],
        document_id: str
    ) -> List[ParentChunk]:
        """
        Create parents using fixed-size windows.
        
        Args:
            blocks: List of normalized blocks.
            document_id: Document ID.
            
        Returns:
            List of ParentChunk objects.
        """
        parents = []
        current_candidate = ParentChunkCandidate()
        current_tokens = 0
        
        for block in blocks:
            block_tokens = count_tokens(block.normalized_text)
            
            # Check if adding this block exceeds limit
            if current_tokens + block_tokens > self.max_tokens and not current_candidate.is_empty:
                # Finalize current candidate
                parent = self._finalize_candidate(
                    current_candidate, document_id, parents
                )
                if parent:
                    parents.extend(parent if isinstance(parent, list) else [parent])
                
                # Start new candidate
                current_candidate = ParentChunkCandidate()
                current_tokens = 0
            
            current_candidate.add_block(block)
            current_tokens += block_tokens
        
        # Finalize last candidate
        if not current_candidate.is_empty:
            parent = self._finalize_candidate(
                current_candidate, document_id, parents
            )
            if parent:
                parents.extend(parent if isinstance(parent, list) else [parent])
        
        return parents
    
    def _finalize_candidate(
        self,
        candidate: ParentChunkCandidate,
        document_id: str,
        existing_parents: List[ParentChunk]
    ) -> Optional[List[ParentChunk]]:
        """
        Finalize a parent chunk candidate.
        
        May split if too large or return None if too small.
        
        Args:
            candidate: Candidate to finalize.
            document_id: Document ID.
            existing_parents: Existing parents (for context).
            
        Returns:
            List of ParentChunk objects or None.
        """
        if candidate.is_empty:
            return None
        
        text = candidate.text
        token_count = count_tokens(text)
        
        # If too large, split into multiple parents
        if token_count > self.max_tokens:
            return self._split_oversized_candidate(candidate, document_id)
        
        # Create single parent
        language = self._determine_language(candidate.blocks)
        
        parent = ParentChunk(
            parent_id=generate_parent_id(),
            document_id=document_id,
            text=text,
            token_count=token_count,
            section_path=candidate.section_path,
            page_range=candidate.page_range,
            language=language,
            child_ids=[],
            metadata={
                "block_count": len(candidate.blocks),
            }
        )
        
        return [parent]
    
    def _split_oversized_candidate(
        self,
        candidate: ParentChunkCandidate,
        document_id: str
    ) -> List[ParentChunk]:
        """
        Split an oversized candidate into multiple parents.
        
        Args:
            candidate: Oversized candidate.
            document_id: Document ID.
            
        Returns:
            List of ParentChunk objects.
        """
        parents = []
        current_blocks: List[NormalizedBlock] = []
        current_tokens = 0
        
        for block in candidate.blocks:
            block_tokens = count_tokens(block.normalized_text)
            
            # If single block exceeds max, it becomes its own parent
            if block_tokens > self.max_tokens:
                # Finalize current if not empty
                if current_blocks:
                    parent = self._create_parent_from_blocks(
                        current_blocks,
                        document_id,
                        candidate.section_path
                    )
                    parents.append(parent)
                    current_blocks = []
                    current_tokens = 0
                
                # Create parent from oversized block
                parent = self._create_parent_from_blocks(
                    [block],
                    document_id,
                    candidate.section_path
                )
                parents.append(parent)
                continue
            
            # Check if adding block exceeds limit
            if current_tokens + block_tokens > self.max_tokens and current_blocks:
                parent = self._create_parent_from_blocks(
                    current_blocks,
                    document_id,
                    candidate.section_path
                )
                parents.append(parent)
                current_blocks = []
                current_tokens = 0
            
            current_blocks.append(block)
            current_tokens += block_tokens
        
        # Finalize remaining
        if current_blocks:
            parent = self._create_parent_from_blocks(
                current_blocks,
                document_id,
                candidate.section_path
            )
            parents.append(parent)
        
        return parents
    
    def _create_parent_from_blocks(
        self,
        blocks: List[StructuredBlock],
        document_id: str,
        section_path: List[str]
    ) -> ParentChunk:
        """
        Create a parent chunk from a list of blocks.
        
        Args:
            blocks: List of blocks.
            document_id: Document ID.
            section_path: Section path.
            
        Returns:
            ParentChunk object.
        """
        text = "\n\n".join(b.normalized_text for b in blocks)
        token_count = count_tokens(text)
        language = self._determine_language(blocks)
        
        # Determine page range
        pages = [b.page_number for b in blocks if b.page_number is not None]
        page_range = (min(pages), max(pages)) if pages else None
        
        return ParentChunk(
            parent_id=generate_parent_id(),
            document_id=document_id,
            text=text,
            token_count=token_count,
            section_path=section_path,
            page_range=page_range,
            language=language,
            child_ids=[],
            metadata={
                "block_count": len(blocks),
            }
        )
    
    def _update_section_path(
        self,
        current_path: List[str],
        heading_text: str,
        heading_level: int
    ) -> List[str]:
        """
        Update section path based on new heading.
        
        Args:
            current_path: Current section path.
            heading_text: New heading text.
            heading_level: Level of new heading (1-6).
            
        Returns:
            Updated section path.
        """
        # Truncate path to heading level - 1
        new_path = current_path[:heading_level - 1]
        
        # Add new heading
        new_path.append(heading_text)
        
        return new_path
    
    def _determine_language(self, blocks: List[StructuredBlock]) -> Language:
        """
        Determine predominant language from blocks.
        
        Args:
            blocks: List of blocks.
            
        Returns:
            Predominant Language.
        """
        if not blocks:
            return Language.EN
        
        lang_counts = {Language.AR: 0, Language.EN: 0, Language.MIXED: 0}
        
        for block in blocks:
            lang_counts[block.language] = lang_counts.get(block.language, 0) + 1
        
        # If any mixed, return mixed
        if lang_counts[Language.MIXED] > 0:
            return Language.MIXED
        
        # Return majority
        if lang_counts[Language.AR] > lang_counts[Language.EN]:
            return Language.AR
        elif lang_counts[Language.EN] > lang_counts[Language.AR]:
            return Language.EN
        else:
            return Language.MIXED