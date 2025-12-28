"""
Stage 8: Chunking

Creates parent and child chunks based on document hierarchy,
respecting atomic units and applying language-aware overlap.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from logging import Logger

from src.config.settings import PipelineConfig
from src.models.enums import DocumentFormat, Language, BlockType
from src.models.chunks import ParentChunk, ChildChunk
from src.pipeline.stages.stage_07_structure_parsing import StructuredBlock
from src.pipeline.helpers.tokenizer import count_tokens
from src.pipeline.helpers.id_generator import generate_parent_id, generate_chunk_id
from src.chunking.hierarchy_resolver import HierarchyResolver
from src.chunking.parent_chunker import ParentChunker
from src.chunking.child_chunker import ChildChunker
from src.chunking.atomic_detector import AtomicDetector


@dataclass
class ChunkingInput:
    """Input for the chunking stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    blocks: List[StructuredBlock]
    primary_language: Language


@dataclass
class ChunkingOutput:
    """Output from the chunking stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    parent_chunks: List[ParentChunk]
    child_chunks: List[ChildChunk]
    hierarchy_depth: int
    total_tokens: int
    chunking_stats: dict


class ChunkingStage:
    """
    Stage 8: Chunking
    
    Responsibilities:
    - Determine hierarchy level (1, 2, or 3) based on document size
    - Create parent chunks (structure-aware)
    - Create child chunks with overlap
    - Respect atomic units (tables, lists)
    - Maintain sibling links (prev/next)
    - Apply language-aware overlap
    """
    
    STAGE_NAME = "Chunking"
    STAGE_NUMBER = 8
    TOTAL_STAGES = 11
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the chunking stage.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.hierarchy_resolver = HierarchyResolver.from_config(config)
        self.parent_chunker = ParentChunker(max_tokens=config.max_tokens, min_tokens=config.min_tokens, overlap_tokens= config.overlap_tokens)
        self.child_chunker = ChildChunker(config)
        self.atomic_detector = AtomicDetector(config)
    
    def execute(
        self,
        input_data: ChunkingInput,
        logger: Logger
    ) -> ChunkingOutput:
        """
        Execute the chunking stage.
        
        Args:
            input_data: Chunking input with structured blocks.
            logger: Logger instance for progress tracking.
            
        Returns:
            ChunkingOutput with parent and child chunks.
        """
        start_time = time.time()
        
        blocks = input_data.blocks
        document_id = input_data.document_id
        primary_language = input_data.primary_language
        
        # Calculate total tokens
        total_tokens = self._calculate_total_tokens(blocks)
        
        # Determine hierarchy depth
        hierarchy_decision = self.hierarchy_resolver.resolve(blocks=blocks, total_tokens= total_tokens)
        hierarchy_depth = hierarchy_decision.depth

        # Initialize outputs
        parent_chunks: List[ParentChunk] = []
        child_chunks: List[ChildChunk] = []
        
        # Statistics
        stats = {
            'total_tokens': total_tokens,
            'hierarchy_depth': hierarchy_depth,
            'parent_chunks': 0,
            'child_chunks': 0,
            'atomic_units_preserved': 0,
            'avg_child_tokens': 0,
        }
        
        if hierarchy_depth == 1:
            # Small document: chunks only, no parents
            child_chunks = self._create_flat_chunks(
                blocks=blocks,
                document_id=document_id,
                primary_language=primary_language
            )
        else:
            # Medium/Large document: create parents and children
            parent_chunks, child_chunks = self._create_hierarchical_chunks(
                blocks=blocks,
                document_id=document_id,
                primary_language=primary_language,
                hierarchy_depth=hierarchy_depth
            )
        
        # Link siblings
        child_chunks = self._link_siblings(child_chunks)
        
        # Update statistics
        stats['parent_chunks'] = len(parent_chunks)
        stats['child_chunks'] = len(child_chunks)
        
        if child_chunks:
            total_child_tokens = sum(c.token_count for c in child_chunks)
            stats['avg_child_tokens'] = round(total_child_tokens / len(child_chunks), 1)
        
        stats['atomic_units_preserved'] = self._count_atomic_units(child_chunks)
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log completion
        logger.info(
            f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
            f"{self.STAGE_NAME} ✓ ({duration_ms}ms)"
        )
        logger.info(
            f"  → Hierarchy: {hierarchy_depth} level(s) ({total_tokens:,} tokens)"
        )
        logger.info(
            f"  → Created {len(parent_chunks)} parents, {len(child_chunks)} children"
        )
                
        return ChunkingOutput(
            document_id=document_id,
            stored_path=input_data.stored_path,
            format=input_data.format,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
            hierarchy_depth=hierarchy_depth,
            total_tokens=total_tokens,
            chunking_stats=stats
        )
    
    def _calculate_total_tokens(self, blocks: List[StructuredBlock]) -> int:
        """
        Calculate total token count for all blocks.
        
        Args:
            blocks: List of structured blocks.
            
        Returns:
            Total token count.
        """
        total_text = " ".join(block.normalized_text for block in blocks)
        return count_tokens(total_text)
    









    def _create_flat_chunks(
        self,
        blocks: List[StructuredBlock],
        document_id: str,
        primary_language: Language
    ) -> List[ChildChunk]:

        child_chunks: List[ChildChunk] = []

        # Group blocks by section
        section_blocks = self._group_blocks_by_section(blocks)

        for _, section_block_list in section_blocks.items():

            combined_text = self._combine_block_texts(section_block_list)
            if not combined_text.strip():
                continue

            # Delegate flat chunking to ChildChunker
            children = self.child_chunker.create_children_from_blocks(
                blocks=section_block_list,
                document_id=document_id
            )

            child_chunks.extend(children)

        return child_chunks


    def _create_hierarchical_chunks(
        self,
        blocks: List[StructuredBlock],
        document_id: str,
        primary_language: Language,
        hierarchy_depth: int
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Create hierarchical parent and child chunks.
        
        Args:
            blocks: List of structured blocks.
            document_id: Document ID.
            primary_language: Primary language.
            hierarchy_depth: Hierarchy depth (2 or 3).
            
        Returns:
            Tuple of (parent_chunks, child_chunks).
        """

        parent_chunks: List[ParentChunk] = []
        child_chunks: List[ChildChunk] = []

        # Group blocks by section
        section_blocks = self._group_blocks_by_section(blocks)

        for _, section_block_list in section_blocks.items():

            combined_text = self._combine_block_texts(section_block_list)
            if not combined_text.strip():
                continue

            # Small section → no parents, single-level children
            if count_tokens(combined_text) < self.config.min_tokens:
                children = self.child_chunker.create_children_from_blocks(
                    blocks=section_block_list,
                    document_id=document_id
                )
                child_chunks.extend(children)
                continue

            # Create parents
            parents = self.parent_chunker.create_parents(
                blocks=section_block_list,
                document_id=document_id,
                hierarchy_depth=hierarchy_depth
            )

            if not parents:
                continue

            # Create children FROM parents (correct API)
            children = self.child_chunker.create_children_from_parents(
                parents=parents,
                document_id=document_id
            )

            parent_chunks.extend(parents)
            child_chunks.extend(children)

        return parent_chunks, child_chunks


    def _create_child_chunk(
        self,
        text: str,
        document_id: str,
        parent_id: Optional[str],
        section_path: List[str],
        page_number: Optional[int],
        language: Language,
        block_type: BlockType,
        position: int
    ) -> ChildChunk:
        """
        Create a child chunk.
        
        Args:
            text: Chunk text.
            document_id: Document ID.
            parent_id: Parent chunk ID (or None).
            section_path: Section path list.
            page_number: Page number (if known).
            language: Chunk language.
            block_type: Block type.
            position: Position in document.
            
        Returns:
            ChildChunk instance.
        """
        return ChildChunk(
            chunk_id=generate_chunk_id(),
            parent_id=parent_id,
            document_id=document_id,
            text=text,
            token_count=count_tokens(text),
            language=language,
            block_type=block_type,
            section_path=section_path,
            page_number=page_number,
            position_in_parent=position,
            prev_chunk_id=None,  # Set in _link_siblings
            next_chunk_id=None,  # Set in _link_siblings
            embedding=None  # Set in embedding stage
        )
    
    def _link_siblings(self, chunks: List[ChildChunk]) -> List[ChildChunk]:
        """
        Link chunks to their siblings (prev/next).
        
        Args:
            chunks: List of child chunks.
            
        Returns:
            List of chunks with sibling links set.
        """
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.prev_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i + 1].chunk_id
        
        return chunks
    
    def _group_blocks_by_section(
        self,
        blocks: List[StructuredBlock]
    ) -> dict[str, List[StructuredBlock]]:
        """
        Group blocks by their section ID.
        
        Args:
            blocks: List of structured blocks.
            
        Returns:
            Dictionary mapping section_id to list of blocks.
        """
        sections: dict[str, List[StructuredBlock]] = {}
        
        for block in blocks:
            section_id = block.section_id
            if section_id not in sections:
                sections[section_id] = []
            sections[section_id].append(block)
        
        return sections
    
    def _combine_block_texts(self, blocks: List[StructuredBlock]) -> str:
        """
        Combine text from multiple blocks.
        
        Args:
            blocks: List of blocks.
            
        Returns:
            Combined text string.
        """
        texts = [block.normalized_text for block in blocks]
        return "\n\n".join(texts)
    
    def _determine_chunk_language(
        self,
        blocks: List[StructuredBlock],
        primary_language: Language
    ) -> Language:
        """
        Determine language for a group of blocks.
        
        Args:
            blocks: List of blocks.
            primary_language: Document's primary language.
            
        Returns:
            Language for the chunk.
        """
        if not blocks:
            return primary_language
        
        # Count language occurrences
        lang_counts = {Language.AR: 0, Language.EN: 0, Language.MIXED: 0}
        
        for block in blocks:
            lang_counts[block.language] += 1
        
        # If any blocks are mixed, the chunk is mixed
        if lang_counts[Language.MIXED] > 0:
            return Language.MIXED
        
        # Return dominant language
        if lang_counts[Language.AR] > lang_counts[Language.EN]:
            return Language.AR
        elif lang_counts[Language.EN] > lang_counts[Language.AR]:
            return Language.EN
        else:
            return primary_language
    
    def _get_page_range(
        self,
        blocks: List[StructuredBlock]
    ) -> Optional[Tuple[int, int]]:
        """
        Get page range for a group of blocks.
        
        Args:
            blocks: List of blocks.
            
        Returns:
            Tuple of (start_page, end_page) or None.
        """
        page_numbers = [
            b.page_number for b in blocks
            if b.page_number is not None
        ]
        
        if not page_numbers:
            return None
        
        return (min(page_numbers), max(page_numbers))
    
    def _is_atomic_section(self, blocks: List[StructuredBlock]) -> bool:
        """
        Check if a section contains atomic units that shouldn't be split.
        
        Args:
            blocks: List of blocks in section.
            
        Returns:
            True if section is atomic.
        """
        for block in blocks:
            if block.structural_hint in [BlockType.TABLE, BlockType.LIST]:
                return self.atomic_detector.is_atomic(
                    block.normalized_text,
                    block.structural_hint
                )
        return False
    
    def _get_dominant_block_type(self, blocks: List[StructuredBlock]) -> BlockType:
        """
        Get the dominant block type from a list of blocks.
        
        Args:
            blocks: List of blocks.
            
        Returns:
            Most common BlockType.
        """
        if not blocks:
            return BlockType.PARAGRAPH
        
        type_counts: dict[BlockType, int] = {}
        
        for block in blocks:
            block_type = block.structural_hint
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
        
        return max(type_counts, key=type_counts.get)
    
    def _count_atomic_units(self, chunks: List[ChildChunk]) -> int:
        """
        Count atomic units in chunks.
        
        Args:
            chunks: List of child chunks.
            
        Returns:
            Count of atomic units.
        """
        atomic_types = {BlockType.TABLE, BlockType.LIST}
        return sum(1 for c in chunks if c.block_type in atomic_types)