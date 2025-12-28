"""
Stage 7: Structural Parsing

Builds a document tree based on heading hierarchy
and tracks section paths for each block.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from logging import Logger

from src.config.settings import PipelineConfig
from src.models.enums import DocumentFormat, Language, BlockType
from src.pipeline.stages.stage_06_language_detection import LanguageBlock


@dataclass
class SectionNode:
    """Represents a section in the document hierarchy."""
    section_id: str
    title: str
    level: int  # 1-6 for headings, 0 for root
    parent_id: Optional[str]
    children_ids: List[str] = field(default_factory=list)
    block_ids: List[str] = field(default_factory=list)


@dataclass
class StructuredBlock:
    """Block with section path information attached."""
    block_id: str
    normalized_text: str
    original_text: str
    page_number: int | None
    structural_hint: BlockType
    heading_level: int | None
    source_offset: int
    language: Language
    section_path: List[str]  # ["Chapter 1", "Section 1.1", ...]
    section_id: str
    depth: int  # Depth in section hierarchy


@dataclass
class StructureParsingInput:
    """Input for the structure parsing stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    blocks: List[LanguageBlock]
    primary_language: Language


@dataclass
class StructureParsingOutput:
    """Output from the structure parsing stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    blocks: List[StructuredBlock]
    primary_language: Language
    section_tree: Dict[str, SectionNode]
    max_depth: int
    total_sections: int


class StructureParsingStage:
    """
    Stage 7: Structural Parsing
    
    Responsibilities:
    - Build document tree based on heading hierarchy
    - Track section paths (full depth ancestry)
    - Attach blocks to sections
    - Calculate document structure metrics
    """
    
    STAGE_NAME = "Structure Parsing"
    STAGE_NUMBER = 7
    TOTAL_STAGES = 11
    
    ROOT_SECTION_ID = "root"
    ROOT_SECTION_TITLE = "Document Root"
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the structure parsing stage.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
    
    def execute(
        self,
        input_data: StructureParsingInput,
        logger: Logger
    ) -> StructureParsingOutput:
        """
        Execute the structure parsing stage.
        
        Args:
            input_data: Structure parsing input with language blocks.
            logger: Logger instance for progress tracking.
            
        Returns:
            StructureParsingOutput with structured blocks and section tree.
        """
        start_time = time.time()
        
        blocks = input_data.blocks
        
        # Build section tree
        section_tree, max_depth = self._build_section_tree(blocks)
        
        # Attach blocks to sections
        structured_blocks = self._attach_blocks_to_sections(blocks, section_tree)
        
        # Count actual sections (excluding root)
        total_sections = len(section_tree) - 1  # Exclude root
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log completion
        logger.info(
            f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
            f"{self.STAGE_NAME} ✓ ({duration_ms}ms)"
        )
        logger.info(f"  → Found {total_sections} sections, max depth: {max_depth}")
        
        return StructureParsingOutput(
            document_id=input_data.document_id,
            stored_path=input_data.stored_path,
            format=input_data.format,
            blocks=structured_blocks,
            primary_language=input_data.primary_language,
            section_tree=section_tree,
            max_depth=max_depth,
            total_sections=total_sections
        )
    
    def _build_section_tree(
        self,
        blocks: List[LanguageBlock]
    ) -> tuple[Dict[str, SectionNode], int]:
        """
        Build a hierarchical section tree from blocks.
        
        Args:
            blocks: List of language blocks.
            
        Returns:
            Tuple of (section_tree dict, max_depth).
        """
        # Initialize root section
        section_tree: Dict[str, SectionNode] = {
            self.ROOT_SECTION_ID: SectionNode(
                section_id=self.ROOT_SECTION_ID,
                title=self.ROOT_SECTION_TITLE,
                level=0,
                parent_id=None,
                children_ids=[],
                block_ids=[]
            )
        }
        
        # Track current section at each level
        # level_stack[i] = section_id at level i
        level_stack: Dict[int, str] = {0: self.ROOT_SECTION_ID}
        max_depth = 0
        section_counter = 0
        
        for block in blocks:
            # Check if this block is a heading
            if self._is_heading(block):
                heading_level = block.heading_level or 1
                max_depth = max(max_depth, heading_level)
                
                # Create new section
                section_counter += 1
                section_id = f"section_{section_counter}"
                
                # Find parent section (closest level above this one)
                parent_level = heading_level - 1
                while parent_level >= 0 and parent_level not in level_stack:
                    parent_level -= 1
                
                parent_id = level_stack.get(parent_level, self.ROOT_SECTION_ID)
                
                # Create section node
                section_node = SectionNode(
                    section_id=section_id,
                    title=self._extract_heading_title(block.normalized_text),
                    level=heading_level,
                    parent_id=parent_id,
                    children_ids=[],
                    block_ids=[block.block_id]
                )
                
                section_tree[section_id] = section_node
                
                # Update parent's children
                section_tree[parent_id].children_ids.append(section_id)
                
                # Update level stack - clear all levels >= current level
                levels_to_remove = [l for l in level_stack if l >= heading_level]
                for l in levels_to_remove:
                    del level_stack[l]
                
                level_stack[heading_level] = section_id
            
            else:
                # Non-heading block - attach to current deepest section
                current_section_id = self._get_current_section(level_stack)
                section_tree[current_section_id].block_ids.append(block.block_id)
        
        return section_tree, max_depth
    
    def _attach_blocks_to_sections(
        self,
        blocks: List[LanguageBlock],
        section_tree: Dict[str, SectionNode]
    ) -> List[StructuredBlock]:
        """
        Create structured blocks with section path information.
        
        Args:
            blocks: List of language blocks.
            section_tree: Section tree dictionary.
            
        Returns:
            List of structured blocks with section paths.
        """
        # Build block_id to section_id mapping
        block_to_section: Dict[str, str] = {}
        for section_id, section in section_tree.items():
            for block_id in section.block_ids:
                block_to_section[block_id] = section_id
        
        # Create structured blocks
        structured_blocks: List[StructuredBlock] = []
        
        for block in blocks:
            section_id = block_to_section.get(block.block_id, self.ROOT_SECTION_ID)
            section_path = self._get_section_path(section_id, section_tree)
            depth = len(section_path)
            
            # Convert structural_hint to BlockType if it's a string
            structural_hint = block.block_type
            if isinstance(structural_hint, str):
                try:
                    structural_hint = BlockType(structural_hint)
                except ValueError:
                    structural_hint = BlockType.PARAGRAPH
            
            structured_block = StructuredBlock(
                block_id=block.block_id,
                normalized_text=block.normalized_text,
                original_text=block.original_text,
                page_number=block.page_number,
                structural_hint=structural_hint,
                heading_level=block.heading_level,
                source_offset=block.source_offset,
                language=block.language,
                section_path=section_path,
                section_id=section_id,
                depth=depth
            )
            
            structured_blocks.append(structured_block)
        
        return structured_blocks
    
    def _get_section_path(
        self,
        section_id: str,
        section_tree: Dict[str, SectionNode]
    ) -> List[str]:
        """
        Get the full section path (ancestry) for a section.
        
        Args:
            section_id: Section ID.
            section_tree: Section tree dictionary.
            
        Returns:
            List of section titles from root to current section.
        """
        path: List[str] = []
        current_id = section_id
        
        while current_id and current_id != self.ROOT_SECTION_ID:
            section = section_tree.get(current_id)
            if not section:
                break
            
            path.insert(0, section.title)
            current_id = section.parent_id
        
        return path
    
    def _get_current_section(self, level_stack: Dict[int, str]) -> str:
        """
        Get the current deepest section from the level stack.
        
        Args:
            level_stack: Dictionary mapping levels to section IDs.
            
        Returns:
            Section ID of the deepest current section.
        """
        if not level_stack:
            return self.ROOT_SECTION_ID
        
        max_level = max(level_stack.keys())
        return level_stack[max_level]
    
    def _is_heading(self, block: LanguageBlock) -> bool:
        """
        Check if a block is a heading.
        
        Args:
            block: Language block to check.
            
        Returns:
            True if block is a heading.
        """
        # Check structural hint
        hint = block.block_type
        if isinstance(hint, str):
            return hint.lower() == 'heading'
        elif isinstance(hint, BlockType):
            return hint == BlockType.HEADING
        
        # Check heading level
        if block.heading_level is not None and block.heading_level > 0:
            return True
        
        return False
    
    def _extract_heading_title(self, text: str) -> str:
        """
        Extract a clean heading title from text.
        
        Args:
            text: Raw heading text.
            
        Returns:
            Cleaned heading title.
        """
        # Remove leading/trailing whitespace
        title = text.strip()
        
        # Remove markdown heading markers if present
        while title.startswith('#'):
            title = title[1:]
        
        title = title.strip()
        
        # Truncate if too long
        max_length = 100
        if len(title) > max_length:
            title = title[:max_length] + "..."
        
        return title or "Untitled Section"