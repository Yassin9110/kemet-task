"""
Markdown extractor for the Multilingual RAG Ingestion Pipeline.

Extracts text from Markdown files, preserving heading hierarchy,
lists, and other structural elements.
"""

from pathlib import Path
from typing import List, Union, Optional, Tuple
import re

from .base import BaseExtractor, ExtractionError
from ..models.blocks import ExtractedBlock
from ..models.enums import DocumentFormat, BlockType
from ..pipeline.helpers.id_generator import generate_block_id


class MarkdownExtractor(BaseExtractor):
    """
    Extractor for Markdown files.
    
    Parses Markdown structure including:
    - Headings (# through ######)
    - Lists (ordered and unordered)
    - Tables
    - Paragraphs
    """
    
    SUPPORTED_EXTENSIONS = [".md", ".markdown", ".mdown", ".mkd"]
    FORMAT = DocumentFormat.MARKDOWN
    
    # Regex patterns for Markdown elements
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    TABLE_PATTERN = re.compile(r'^\|.+\|$\n^\|[-:| ]+\|$\n(?:^\|.+\|$\n?)+', re.MULTILINE)
    UNORDERED_LIST_PATTERN = re.compile(r'((?:^[ \t]*[-*+][ \t]+.+\n?)+)', re.MULTILINE)
    ORDERED_LIST_PATTERN = re.compile(r'((?:^[ \t]*\d+\.[ \t]+.+\n?)+)', re.MULTILINE)
    
    def __init__(self, preserve_code_language: bool = True):
        """
        Initialize Markdown extractor.
        
        Args:
            preserve_code_language: Whether to store code language in metadata.
        """
        super().__init__()
        self.preserve_code_language = preserve_code_language
    
    def extract(self, file_path: Union[str, Path]) -> List[ExtractedBlock]:
        """
        Extract blocks from a Markdown file.
        
        Args:
            file_path: Path to the Markdown file.
            
        Returns:
            List of ExtractedBlock objects.
        """
        path = self._validate_file(file_path)
        
        try:
            content = self._read_file_text(path)
            return self._parse_content(content)
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract Markdown from {path}: {str(e)}",
                file_path=str(path),
                cause=e
            )
    
    def _parse_content(self, content: str) -> List[ExtractedBlock]:
        """
        Parse Markdown content into blocks.
        
        Args:
            content: Raw Markdown content.
            
        Returns:
            List of ExtractedBlock objects.
        """
        blocks = []
        
        # Track positions of special elements to avoid double-processing
        processed_ranges: List[Tuple[int, int]] = []
        
        # Extract tables
        for match in self.TABLE_PATTERN.finditer(content):
            if self._is_in_processed_range(match.start(), processed_ranges):
                continue
            
            block = ExtractedBlock(
                block_id=generate_block_id(),
                raw_text=match.group(0).strip(),
                block_type=BlockType.TABLE,
                source_offset=match.start(),
                metadata={
                    "source_format": "markdown",
                }
            )
            blocks.append(block)
            processed_ranges.append((match.start(), match.end()))
        
        # Process line by line for remaining elements
        lines = content.split('\n')
        current_offset = 0
        i = 0
        
        while i < len(lines):
            line = lines[i]
            line_start = current_offset
            
            # Skip if in processed range
            if self._is_in_processed_range(line_start, processed_ranges):
                current_offset += len(line) + 1
                i += 1
                continue
            
            # Check for heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()
                
                block = ExtractedBlock(
                    block_id=generate_block_id(),
                    raw_text=text,
                    block_type=BlockType.HEADING,
                    source_offset=line_start,
                    heading_level=level,
                    metadata={
                        "source_format": "markdown",
                    }
                )
                blocks.append(block)
                processed_ranges.append((line_start, line_start + len(line)))
                current_offset += len(line) + 1
                i += 1
                continue
            
            # Check for unordered list
            if re.match(r'^[ \t]*[-*+][ \t]+', line):
                list_lines, end_i = self._collect_list_lines(lines, i, r'^[ \t]*[-*+][ \t]+')
                list_text = '\n'.join(list_lines)
                
                block = ExtractedBlock(
                    block_id=generate_block_id(),
                    raw_text=list_text,
                    block_type=BlockType.LIST,
                    source_offset=line_start,
                    metadata={
                        "source_format": "markdown",
                        "list_type": "unordered",
                    }
                )
                blocks.append(block)
                
                # Update offset and index
                for j in range(i, end_i):
                    current_offset += len(lines[j]) + 1
                i = end_i
                continue
            
            # Check for ordered list
            if re.match(r'^[ \t]*\d+\.[ \t]+', line):
                list_lines, end_i = self._collect_list_lines(lines, i, r'^[ \t]*\d+\.[ \t]+')
                list_text = '\n'.join(list_lines)
                
                block = ExtractedBlock(
                    block_id=generate_block_id(),
                    raw_text=list_text,
                    block_type=BlockType.LIST,
                    source_offset=line_start,
                    metadata={
                        "source_format": "markdown",
                        "list_type": "ordered",
                    }
                )
                blocks.append(block)
                
                for j in range(i, end_i):
                    current_offset += len(lines[j]) + 1
                i = end_i
                continue
            
            # Regular paragraph - collect until empty line or special element
            if line.strip():
                para_lines = []
                while i < len(lines) and lines[i].strip():
                    # Stop if we hit a special element
                    if (re.match(r'^#{1,6}\s+', lines[i]) or
                        re.match(r'^[ \t]*[-*+][ \t]+', lines[i]) or
                        re.match(r'^[ \t]*\d+\.[ \t]+', lines[i]) or
                        re.match(r'^```', lines[i]) or
                        re.match(r'^\|', lines[i])):
                        break
                    para_lines.append(lines[i])
                    i += 1
                
                if para_lines:
                    para_text = '\n'.join(para_lines).strip()
                    
                    block = ExtractedBlock(
                        block_id=generate_block_id(),
                        raw_text=para_text,
                        block_type=BlockType.PARAGRAPH,
                        source_offset=line_start,
                        metadata={
                            "source_format": "markdown",
                        }
                    )
                    blocks.append(block)
                    
                    for para_line in para_lines:
                        current_offset += len(para_line) + 1
                    continue
            
            # Empty line or unhandled - just advance
            current_offset += len(line) + 1
            i += 1
        
        # Sort blocks by source offset
        blocks.sort(key=lambda b: b.source_offset)
        
        return blocks
    
    def _is_in_processed_range(self, offset: int, ranges: List[Tuple[int, int]]) -> bool:
        """Check if offset is within any processed range."""
        for start, end in ranges:
            if start <= offset < end:
                return True
        return False
    
    def _collect_list_lines(
        self,
        lines: List[str],
        start_i: int,
        pattern: str
    ) -> Tuple[List[str], int]:
        """
        Collect consecutive list lines.
        
        Args:
            lines: All lines.
            start_i: Starting index.
            pattern: Regex pattern for list items.
            
        Returns:
            Tuple of (list_lines, end_index).
        """
        list_lines = []
        i = start_i
        
        while i < len(lines):
            line = lines[i]
            
            # List item
            if re.match(pattern, line):
                list_lines.append(line)
                i += 1
            # Continuation (indented)
            elif line.startswith('  ') and list_lines:
                list_lines.append(line)
                i += 1
            # Empty line might continue list
            elif line.strip() == '' and i + 1 < len(lines) and re.match(pattern, lines[i + 1]):
                list_lines.append(line)
                i += 1
            else:
                break
        
        return list_lines, i