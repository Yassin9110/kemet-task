"""
Child chunker for the Multilingual RAG Ingestion Pipeline.

Creates child chunks from parent chunks or directly from blocks,
with proper overlap and sibling linking.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..models.blocks import NormalizedBlock
from ..models.chunks import ParentChunk, ChildChunk
from ..models.enums import Language, BlockType
from ..pipeline.helpers.id_generator import generate_chunk_id
from ..pipeline.helpers.tokenizer import count_tokens, Tokenizer, get_tokenizer
from ..pipeline.stages.stage_07_structure_parsing import StructuredBlock


@dataclass
class ChunkingConfig:
    """Configuration for child chunking."""
    
    target_tokens: int = 128
    """Target tokens per child chunk."""
    
    overlap_tokens: int = 50
    """Token overlap between consecutive chunks."""
    
    arabic_overlap_multiplier: float = 1.2
    """Multiplier for Arabic text overlap."""
    
    min_tokens: int = 20
    """Minimum tokens for a chunk."""
    
    def get_overlap_for_language(self, language: Language) -> int:
        """Get overlap tokens for a specific language."""
        if language in (Language.AR, Language.MIXED):
            return int(self.overlap_tokens * self.arabic_overlap_multiplier)
        return self.overlap_tokens


class SentenceSplitter:
    """
    Splits text into sentences, respecting Arabic and English punctuation.
    """
    
    # Sentence-ending punctuation
    SENTENCE_ENDINGS = r'[.!?؟।。]'
    
    # Pattern for splitting sentences
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?؟।。])\s+(?=[A-Z\u0600-\u06FF])|'  # After punctuation, before capital/Arabic
        r'(?<=[.!?؟।。])\s*\n',  # After punctuation at line end
        re.UNICODE
    )
    
    # Arabic sentence endings
    ARABIC_ENDINGS = re.compile(r'[.؟!،؛]')
    
    @classmethod
    def split(cls, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split.
            
        Returns:
            List of sentences.
        """
        if not text:
            return []
        
        # First split by explicit sentence boundaries
        parts = cls.SENTENCE_PATTERN.split(text)
        
        sentences = []
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
        
        # If no splits found, return as single sentence
        if not sentences:
            return [text.strip()]
        
        return sentences
    
    @classmethod
    def split_at_boundaries(cls, text: str, max_length: int) -> List[str]:
        """
        Split text at sentence boundaries, respecting max length.
        
        Args:
            text: Text to split.
            max_length: Maximum character length per chunk.
            
        Returns:
            List of text chunks.
        """
        sentences = cls.split(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If single sentence exceeds max, split it further
            if sentence_length > max_length:
                # Finalize current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence by clauses or words
                sub_chunks = cls._split_long_sentence(sentence, max_length)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding sentence exceeds limit
            if current_length + sentence_length + 1 > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        
        # Finalize remaining
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    @classmethod
    def _split_long_sentence(cls, sentence: str, max_length: int) -> List[str]:
        """
        Split a long sentence into smaller parts.
        
        Args:
            sentence: Long sentence to split.
            max_length: Maximum length per part.
            
        Returns:
            List of sentence parts.
        """
        # Try splitting by clause markers
        clause_markers = re.compile(r'[,;:،؛]\s*')
        clauses = clause_markers.split(sentence)
        
        if len(clauses) > 1:
            # Rebuild with markers
            chunks = []
            current = []
            current_length = 0
            
            for clause in clauses:
                if current_length + len(clause) + 2 > max_length and current:
                    chunks.append(', '.join(current))
                    current = []
                    current_length = 0
                
                current.append(clause.strip())
                current_length += len(clause) + 2
            
            if current:
                chunks.append(', '.join(current))
            
            return chunks
        
        # Last resort: split by words
        words = sentence.split()
        chunks = []
        current = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length and current:
                chunks.append(' '.join(current))
                current = []
                current_length = 0
            
            current.append(word)
            current_length += len(word) + 1
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks


class ChildChunker:
    """
    Creates child chunks from parent chunks or blocks.
    
    Features:
    - Token-based chunking with configurable target size
    - Language-aware overlap (more overlap for Arabic)
    - Sentence boundary respect
    - Sibling linking (prev/next)
    - Atomic block handling (tables kept whole)
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize child chunker.
        
        Args:
            config: Chunking configuration.
        """
        self.config = config or ChunkingConfig()
        self.tokenizer = get_tokenizer()
        self.sentence_splitter = SentenceSplitter()
    
    def create_children_from_parents(
        self,
        parents: List[ParentChunk],
        document_id: str
    ) -> List[ChildChunk]:
        """
        Create child chunks from parent chunks.
        
        Args:
            parents: List of parent chunks.
            document_id: Document ID.
            
        Returns:
            List of ChildChunk objects with sibling links.
        """
        all_children = []
        
        for parent in parents:
            children = self._chunk_parent(parent, document_id)
            
            # Update parent's child_ids
            parent.child_ids = [c.chunk_id for c in children]
            
            all_children.extend(children)
        
        # Link siblings across all children
        self._link_siblings(all_children)
        
        return all_children
    
    def create_children_from_blocks(
        self,
        blocks: List[NormalizedBlock],
        document_id: str
    ) -> List[ChildChunk]:
        """
        Create child chunks directly from blocks (for 1-level hierarchy).
        
        Args:
            blocks: List of normalized blocks.
            document_id: Document ID.
            
        Returns:
            List of ChildChunk objects with sibling links.
        """
        children = []
        
        for block in blocks:
            block_children = self._chunk_block(block, document_id, parent_id=None)
            children.extend(block_children)
        
        # Link siblings
        self._link_siblings(children)
        
        return children
    
    def _chunk_parent(
        self,
        parent: ParentChunk,
        document_id: str
    ) -> List[ChildChunk]:
        """
        Create child chunks from a single parent.
        
        Args:
            parent: Parent chunk to split.
            document_id: Document ID.
            
        Returns:
            List of ChildChunk objects.
        """
        text = parent.text
        token_count = parent.token_count
        
        # If parent is small enough, create single child
        if token_count <= self.config.target_tokens:
            child = ChildChunk(
                chunk_id=generate_chunk_id(),
                document_id=document_id,
                parent_id=parent.parent_id,
                text=text,
                token_count=token_count,
                language=parent.language,
                block_type=BlockType.PARAGRAPH,
                section_path=parent.section_path,
                page_number=parent.page_range[0] if parent.page_range else None,
                position_in_parent=0,
                metadata={
                    "source": "parent",
                }
            )
            return [child]
        
        # Split into multiple children
        return self._split_text_into_children(
            text=text,
            document_id=document_id,
            parent_id=parent.parent_id,
            language=parent.language,
            section_path=parent.section_path,
            page_number=parent.page_range[0] if parent.page_range else None,
            block_type=BlockType.PARAGRAPH
        )
    
    def _chunk_block(
        self,
        block: StructuredBlock,
        document_id: str,
        parent_id: Optional[str]
    ) -> List[ChildChunk]:
        """
        Create child chunks from a single block.
        
        Args:
            block: Block to chunk.
            document_id: Document ID.
            parent_id: Parent ID (if any).
            
        Returns:
            List of ChildChunk objects.
        """
        text = block.normalized_text
        token_count = count_tokens(text)
        
        # If block is atomic (table), keep as single chunk
        if block.structural_hint == BlockType.TABLE:
            child = ChildChunk(
                chunk_id=generate_chunk_id(),
                document_id=document_id,
                parent_id=parent_id,
                text=text,
                token_count=token_count,
                language=block.language,
                block_type=block.structural_hint,
                section_path=[],
                page_number=block.page_number,
                position_in_parent=0,
                metadata={
                    "source": "block",
                    "atomic": True,
                }
            )
            return [child]
        
        # If small enough, create single child
        if token_count <= self.config.target_tokens:
            child = ChildChunk(
                chunk_id=generate_chunk_id(),
                document_id=document_id,
                parent_id=parent_id,
                text=text,
                token_count=token_count,
                language=block.language,
                block_type=block.structural_hint,
                section_path=[],
                page_number=block.page_number,
                position_in_parent=0,
                metadata={
                    "source": "block",
                }
            )
            return [child]
        
        # Split into multiple children
        return self._split_text_into_children(
            text=text,
            document_id=document_id,
            parent_id=parent_id,
            language=block.language,
            section_path=[],
            page_number=block.page_number,
            block_type=block.structural_hint
        )
    
    def _split_text_into_children(
        self,
        text: str,
        document_id: str,
        parent_id: Optional[str],
        language: Language,
        section_path: List[str],
        page_number: Optional[int],
        block_type: BlockType
    ) -> List[ChildChunk]:
        """
        Split text into child chunks with overlap.
        
        Args:
            text: Text to split.
            document_id: Document ID.
            parent_id: Parent ID.
            language: Language of text.
            section_path: Section path.
            page_number: Page number.
            block_type: Block type.
            
        Returns:
            List of ChildChunk objects.
        """
        children = []
        
        # Get overlap for this language
        overlap = self.config.get_overlap_for_language(language)
        target = self.config.target_tokens
        
        # Split into sentences first
        sentences = self.sentence_splitter.split(text)
        
        # Build chunks from sentences
        current_sentences: List[str] = []
        current_tokens = 0
        position = 0
        
        # Track for overlap
        previous_chunk_sentences: List[str] = []
        
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)
            
            # If single sentence exceeds target, handle separately
            if sentence_tokens > target:
                # Finalize current chunk
                if current_sentences:
                    chunk = self._create_child_chunk(
                        sentences=current_sentences,
                        document_id=document_id,
                        parent_id=parent_id,
                        language=language,
                        section_path=section_path,
                        page_number=page_number,
                        block_type=block_type,
                        position=position
                    )
                    children.append(chunk)
                    previous_chunk_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                    current_sentences = []
                    current_tokens = 0
                    position += 1
                
                # Split long sentence and create chunks
                long_chunks = self._split_long_sentence_to_chunks(
                    sentence=sentence,
                    document_id=document_id,
                    parent_id=parent_id,
                    language=language,
                    section_path=section_path,
                    page_number=page_number,
                    block_type=block_type,
                    start_position=position
                )
                children.extend(long_chunks)
                position += len(long_chunks)
                previous_chunk_sentences = []
                continue
            
            # Check if adding sentence exceeds target
            if current_tokens + sentence_tokens > target and current_sentences:
                # Create chunk
                chunk = self._create_child_chunk(
                    sentences=current_sentences,
                    document_id=document_id,
                    parent_id=parent_id,
                    language=language,
                    section_path=section_path,
                    page_number=page_number,
                    block_type=block_type,
                    position=position
                )
                children.append(chunk)
                position += 1
                
                # Prepare overlap: keep last sentences up to overlap tokens
                overlap_sentences = []
                overlap_tokens_count = 0
                for s in reversed(current_sentences):
                    s_tokens = count_tokens(s)
                    if overlap_tokens_count + s_tokens <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens_count += s_tokens
                    else:
                        break
                
                current_sentences = overlap_sentences
                current_tokens = overlap_tokens_count
            
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
        
        # Finalize remaining
        if current_sentences:
            # Check if too small and should merge with previous
            if current_tokens < self.config.min_tokens and children:
                # Merge with previous chunk
                prev_chunk = children[-1]
                merged_text = prev_chunk.text + " " + " ".join(current_sentences)
                prev_chunk.text = merged_text  # Note: This modifies the chunk directly
                prev_chunk.token_count = count_tokens(merged_text)
            else:
                chunk = self._create_child_chunk(
                    sentences=current_sentences,
                    document_id=document_id,
                    parent_id=parent_id,
                    language=language,
                    section_path=section_path,
                    page_number=page_number,
                    block_type=block_type,
                    position=position
                )
                children.append(chunk)
        
        return children
    
    def _create_child_chunk(
        self,
        sentences: List[str],
        document_id: str,
        parent_id: Optional[str],
        language: Language,
        section_path: List[str],
        page_number: Optional[int],
        block_type: BlockType,
        position: int
    ) -> ChildChunk:
        """
        Create a child chunk from sentences.
        
        Args:
            sentences: List of sentences.
            document_id: Document ID.
            parent_id: Parent ID.
            language: Language.
            section_path: Section path.
            page_number: Page number.
            block_type: Block type.
            position: Position in parent.
            
        Returns:
            ChildChunk object.
        """
        text = " ".join(sentences)
        token_count = count_tokens(text)
        
        return ChildChunk(
            chunk_id=generate_chunk_id(),
            document_id=document_id,
            parent_id=parent_id,
            text=text,
            token_count=token_count,
            language=language,
            block_type=block_type,
            section_path=section_path,
            page_number=page_number,
            position_in_parent=position,
            metadata={
                "sentence_count": len(sentences),
            }
        )
    
    def _split_long_sentence_to_chunks(
        self,
        sentence: str,
        document_id: str,
        parent_id: Optional[str],
        language: Language,
        section_path: List[str],
        page_number: Optional[int],
        block_type: BlockType,
        start_position: int
    ) -> List[ChildChunk]:
        """
        Split a long sentence into multiple chunks.
        
        Args:
            sentence: Long sentence to split.
            document_id: Document ID.
            parent_id: Parent ID.
            language: Language.
            section_path: Section path.
            page_number: Page number.
            block_type: Block type.
            start_position: Starting position.
            
        Returns:
            List of ChildChunk objects.
        """
        # Estimate characters per token for this language
        chars_per_token = 4.0 if language == Language.EN else 2.5
        max_chars = int(self.config.target_tokens * chars_per_token)
        
        # Split by clause or word boundaries
        parts = self.sentence_splitter._split_long_sentence(sentence, max_chars)
        
        chunks = []
        for i, part in enumerate(parts):
            chunk = ChildChunk(
                chunk_id=generate_chunk_id(),
                document_id=document_id,
                parent_id=parent_id,
                text=part,
                token_count=count_tokens(part),
                language=language,
                block_type=block_type,
                section_path=section_path,
                page_number=page_number,
                position_in_parent=start_position + i,
                metadata={
                    "from_long_sentence": True,
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _link_siblings(self, children: List[ChildChunk]) -> None:
        """
        Link children as siblings (prev/next).
        
        Args:
            children: List of children to link.
        """
        for i, child in enumerate(children):
            if i > 0:
                child.prev_chunk_id = children[i - 1].chunk_id
            
            if i < len(children) - 1:
                child.next_chunk_id = children[i + 1].chunk_id