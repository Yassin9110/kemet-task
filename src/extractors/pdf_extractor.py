"""
PDF extractor for the Multilingual RAG Ingestion Pipeline.

Extracts text from PDF files (digital text only, no OCR).
Preserves page information and attempts to detect structure.
"""

from pathlib import Path
from typing import List, Union, Optional, Tuple
import re
import fitz  # PyMuPDF
from pypdf import PdfReader
from .base import BaseExtractor, ExtractionError
from ..models.blocks import ExtractedBlock
from ..models.enums import DocumentFormat, BlockType
from ..pipeline.helpers.id_generator import generate_block_id

PYMUPDF_AVAILABLE = True

class PDFExtractor(BaseExtractor):
    """
    Extractor for PDF files.
    
    Extracts digital text from PDFs (no OCR).
    Uses PyMuPDF (fitz) as primary library with pypdf as fallback.
    """
    
    SUPPORTED_EXTENSIONS = [".pdf"]
    FORMAT = DocumentFormat.PDF
    
    def __init__(self, min_text_length: int = 10, extract_by_page: bool = True, detect_headings: bool = True):
        """
        Initialize PDF extractor.
        
        Args:
            min_text_length: Minimum text length for a block.
            extract_by_page: Whether to track page numbers.
            detect_headings: Whether to attempt heading detection.
        """
        super().__init__()        
        self.min_text_length = min_text_length
        self.extract_by_page = extract_by_page
        self.detect_headings = detect_headings
        self.use_pymupdf = PYMUPDF_AVAILABLE
    
    def extract(self, file_path: Union[str, Path]) -> List[ExtractedBlock]:
        """
        Extract blocks from a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of ExtractedBlock objects.
        """
        path = self._validate_file(file_path)
        try:
            if self.use_pymupdf:
                return self._extract_with_pymupdf(path)
            else:
                return self._extract_with_pypdf(path)
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract PDF from {path}: {str(e)}",
                file_path=str(path),
                cause=e
            )
    
    def _extract_with_pymupdf(self, file_path: Path) -> List[ExtractedBlock]:
        """
        Extract using PyMuPDF (fitz).
        
        Args:
            file_path: Path to PDF.
            
        Returns:
            List of ExtractedBlock objects.
        """
        blocks = []
        offset = 0
        
        doc = fitz.open(file_path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_number = page_num + 1  # 1-indexed
                
                # Get text blocks with position information
                text_blocks = page.get_text("blocks")
                
                for block_data in text_blocks:
                    # block_data: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(block_data) < 5:
                        continue
                    
                    text = block_data[4]
                    if isinstance(text, bytes):
                        text = text.decode('utf-8', errors='ignore')
                    
                    text = text.strip()
                    if len(text) < self.min_text_length:
                        continue
                    
                    # Detect block type
                    block_type = self._detect_block_type(text, block_data)
                    heading_level = None
                    
                    if block_type == BlockType.HEADING and self.detect_headings:
                        heading_level = self._estimate_heading_level(text)
                    
                    block = ExtractedBlock(
                        block_id=generate_block_id(),
                        raw_text=text,
                        block_type=block_type,
                        source_offset=offset,
                        page_number=page_number if self.extract_by_page else None,
                        heading_level=heading_level,
                        metadata={
                            "source_format": "pdf",
                            "extractor": "pymupdf",
                            "bbox": block_data[:4] if len(block_data) >= 4 else None,
                        }
                    )
                    
                    blocks.append(block)
                    offset += len(text)
        finally:
            doc.close()
        
        return blocks
    
    def _extract_with_pypdf(self, file_path: Path) -> List[ExtractedBlock]:
        """
        Extract using pypdf (fallback).
        
        Args:
            file_path: Path to PDF.
            
        Returns:
            List of ExtractedBlock objects.
        """
        blocks = []
        offset = 0
        
        reader = PdfReader(file_path)
        
        for page_num, page in enumerate(reader.pages):
            page_number = page_num + 1  # 1-indexed
            
            text = page.extract_text() or ""
            
            # Split into paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            
            for para in paragraphs:
                para = para.strip()
                
                if len(para) < self.min_text_length:
                    continue
                
                # Detect block type
                block_type = self._detect_block_type_simple(para)
                heading_level = None
                
                if block_type == BlockType.HEADING and self.detect_headings:
                    heading_level = self._estimate_heading_level(para)
                
                block = ExtractedBlock(
                    block_id=generate_block_id(),
                    raw_text=para,
                    block_type=block_type,
                    source_offset=offset,
                    page_number=page_number if self.extract_by_page else None,
                    heading_level=heading_level,
                    metadata={
                        "source_format": "pdf",
                        "extractor": "pypdf",
                    }
                )
                
                blocks.append(block)
                offset += len(para)
        
        return blocks
    
    def _detect_block_type(
        self,
        text: str,
        block_data: tuple
    ) -> BlockType:
        """
        Detect block type using PyMuPDF block data.
        
        Args:
            text: Block text.
            block_data: PyMuPDF block information.
            
        Returns:
            Detected BlockType.
        """
        # PyMuPDF block_type: 0 = text, 1 = image
        if len(block_data) > 6 and block_data[6] == 1:
            return BlockType.UNKNOWN  # Image block
        
        return self._detect_block_type_simple(text)
    
    def _detect_block_type_simple(self, text: str) -> BlockType:
        """
        Simple block type detection based on text patterns.
        
        Args:
            text: Block text.
            
        Returns:
            Detected BlockType.
        """
        lines = text.strip().split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # Check for table patterns (multiple | or consistent spacing)
        if self._looks_like_table(text):
            return BlockType.TABLE
        
        # Check for list patterns
        if self._looks_like_list(text):
            return BlockType.LIST
        
        
        # Check for heading patterns
        if self._looks_like_heading(first_line, len(lines)):
            return BlockType.HEADING
        
        return BlockType.PARAGRAPH
    
    def _looks_like_table(self, text: str) -> bool:
        """Check if text looks like a table."""
        lines = text.strip().split('\n')
        
        if len(lines) < 2:
            return False
        
        # Check for pipe separators
        pipe_lines = sum(1 for line in lines if '|' in line)
        if pipe_lines >= len(lines) * 0.5:
            return True
        
        # Check for tab-separated content
        tab_lines = sum(1 for line in lines if '\t' in line)
        if tab_lines >= len(lines) * 0.7:
            return True
        
        return False
    
    def _looks_like_list(self, text: str) -> bool:
        """Check if text looks like a list."""
        lines = text.strip().split('\n')
        
        if len(lines) < 2:
            return False
        
        list_patterns = [
            r'^\s*[-•●○◦▪▸]\s+',  # Bullets
            r'^\s*\d+[.)]\s+',  # Numbers
            r'^\s*[a-zA-Z][.)]\s+',  # Letters
            r'^\s*[٠-٩]+[.)]\s+',  # Arabic numbers
        ]
        
        list_lines = 0
        for line in lines:
            for pattern in list_patterns:
                if re.match(pattern, line):
                    list_lines += 1
                    break
        
        return list_lines >= len(lines) * 0.5
       
    def _looks_like_heading(self, first_line: str, num_lines: int) -> bool:
        """Check if text looks like a heading."""
        # Multiple lines unlikely to be heading
        if num_lines > 2:
            return False
        
        # Too long
        if len(first_line) > 150:
            return False
        
        # Ends with sentence punctuation
        if first_line.rstrip().endswith(('.', ',', ';', '؟', '،')):
            return False
        
        # All caps (but reasonable length)
        if first_line.isupper() and 3 < len(first_line) < 100:
            return True
        
        # Numbered heading
        if re.match(r'^(\d+\.?\s+|chapter\s+\d+|section\s+\d+)', first_line, re.IGNORECASE):
            return True
        
        # Short title-case line
        if len(first_line) < 80 and first_line.istitle():
            return True
        
        return False
    
    def _estimate_heading_level(self, text: str) -> int:
        """
        Estimate heading level.
        
        Args:
            text: Heading text.
            
        Returns:
            Heading level (1-6).
        """
        text = text.strip()
        
        # Chapter -> level 1
        if re.match(r'^chapter\s+', text, re.IGNORECASE):
            return 1
        
        # Part -> level 1
        if re.match(r'^part\s+', text, re.IGNORECASE):
            return 1
        
        # All caps -> level 1 or 2
        if text.isupper():
            return 1
        
        # Numbered sections
        if re.match(r'^\d+\s+', text):
            return 2
        
        if re.match(r'^\d+\.\d+\s+', text):
            return 3
        
        if re.match(r'^\d+\.\d+\.\d+\s+', text):
            return 4
        
        # Default
        return 2