"""
Plain text extractor for the Multilingual RAG Ingestion Pipeline.

Extracts text from .txt files, preserving paragraph structure.
"""

from pathlib import Path
from typing import List, Union
import re

from .base import BaseExtractor, ExtractionError
from ..models.blocks import ExtractedBlock
from ..models.enums import DocumentFormat, BlockType
from ..pipeline.helpers.id_generator import generate_block_id


class TXTExtractor(BaseExtractor):
    """
    Extractor for plain text files.
    
    Splits text into blocks based on paragraph breaks (double newlines).
    Single newlines within paragraphs are preserved.
    """
    
    SUPPORTED_EXTENSIONS = [".txt", ".text"]
    FORMAT = DocumentFormat.TXT
    
    def __init__(self, min_block_length: int = 1):
        """
        Initialize TXT extractor.
        
        Args:
            min_block_length: Minimum characters for a block to be included.
        """
        super().__init__()
        self.min_block_length = min_block_length
    
    def extract(self, file_path: Union[str, Path]) -> List[ExtractedBlock]:
        """
        Extract text blocks from a plain text file.
        
        Args:
            file_path: Path to the .txt file.
            
        Returns:
            List of ExtractedBlock objects.
        """
        path = self._validate_file(file_path)
        
        try:
            content = self._read_file_text(path)
            return self._parse_content(content)
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract text from {path}: {str(e)}",
                file_path=str(path),
                cause=e
            )
    
    def _parse_content(self, content: str) -> List[ExtractedBlock]:
        """
        Parse text content into blocks.
        
        Args:
            content: Raw text content.
            
        Returns:
            List of ExtractedBlock objects.
        """
        blocks = []
        
        # Split on double newlines (paragraph breaks)
        # Keep single newlines within paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_offset = 0
        
        for para in paragraphs:
            # Clean up the paragraph
            para_text = para.strip()
            
            if len(para_text) < self.min_block_length:
                # Track offset even for skipped paragraphs
                current_offset = content.find(para, current_offset) + len(para)
                continue
            
            # Find actual offset in original content
            para_start = content.find(para, current_offset)
            if para_start == -1:
                para_start = current_offset
            
            # Detect if this looks like a heading (short, possibly caps/title case)
            block_type = self._detect_block_type(para_text)
            heading_level = None
            
            if block_type == BlockType.HEADING:
                heading_level = self._estimate_heading_level(para_text, len(blocks))
            
            block = ExtractedBlock(
                block_id=generate_block_id(),
                raw_text=para_text,
                block_type=block_type,
                source_offset=para_start,
                heading_level=heading_level,
                metadata={
                    "source_format": "txt",
                    "paragraph_index": len(blocks),
                }
            )
            
            blocks.append(block)
            current_offset = para_start + len(para)
        
        return blocks
    
    def _detect_block_type(self, text: str) -> BlockType:
        """
        Detect the type of block based on content.
        
        Args:
            text: Block text.
            
        Returns:
            Detected BlockType.
        """
        lines = text.strip().split('\n')
        first_line = lines[0].strip()
        
        # Check for list patterns
        if self._is_list(text):
            return BlockType.LIST
        
        # Check for heading patterns (short, possibly formatted)
        if self._is_heading(first_line, len(lines)):
            return BlockType.HEADING
        
        return BlockType.PARAGRAPH
    
    def _is_heading(self, first_line: str, num_lines: int) -> bool:
        """Check if text looks like a heading."""
        # Single line
        if num_lines > 1:
            return False
        
        # Not too long
        if len(first_line) > 100:
            return False
        
        # Doesn't end with typical sentence punctuation
        if first_line.rstrip().endswith(('.', ',', ';', ':', '؟', '،', '؛')):
            return False
        
        # Check for common heading patterns
        # Numbered heading: "1. Introduction" or "Chapter 1"
        if re.match(r'^(\d+\.?\s+|chapter\s+\d+|section\s+\d+)', first_line, re.IGNORECASE):
            return True
        
        # All caps (but not too short)
        if first_line.isupper() and len(first_line) > 3:
            return True
        
        # Title case and short
        if len(first_line) < 60 and first_line.istitle():
            return True
        
        return False
    
    def _is_list(self, text: str) -> bool:
        """Check if text looks like a list."""
        lines = text.strip().split('\n')
        
        if len(lines) < 2:
            return False
        
        # Check if most lines start with list markers
        list_patterns = [
            r'^\s*[-*•]\s+',  # Bullet points
            r'^\s*\d+[.)]\s+',  # Numbered list
            r'^\s*[a-zA-Z][.)]\s+',  # Lettered list
            r'^\s*[٠-٩]+[.)]\s+',  # Arabic numerals
        ]
        
        list_lines = 0
        for line in lines:
            for pattern in list_patterns:
                if re.match(pattern, line):
                    list_lines += 1
                    break
        
        return list_lines >= len(lines) * 0.6
    
    
    def _estimate_heading_level(self, text: str, position: int) -> int:
        """
        Estimate heading level based on formatting and position.
        
        Args:
            text: Heading text.
            position: Position in document (block index).
            
        Returns:
            Estimated heading level (1-6).
        """
        # First heading is usually level 1
        if position == 0:
            return 1
        
        # All caps suggests higher level
        if text.isupper():
            return 1
        
        # Numbered chapters
        if re.match(r'^chapter\s+\d+', text, re.IGNORECASE):
            return 1
        
        # Numbered sections
        if re.match(r'^\d+\.\s+', text):
            return 2
        
        if re.match(r'^\d+\.\d+\s+', text):
            return 3
        
        if re.match(r'^\d+\.\d+\.\d+\s+', text):
            return 4
        
        # Default to level 2 for other headings
        return 2