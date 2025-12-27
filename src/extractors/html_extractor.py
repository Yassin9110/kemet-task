"""
HTML extractor for the Multilingual RAG Ingestion Pipeline.

Extracts text from HTML files, removing boilerplate and preserving
semantic structure (headings, paragraphs, lists, tables).
"""

from pathlib import Path
from typing import List, Union, Optional, Set
import re
from bs4 import BeautifulSoup, NavigableString, Tag
BS4_AVAILABLE = True
from .base import BaseExtractor, ExtractionError
from ..models.blocks import ExtractedBlock
from ..models.enums import DocumentFormat, BlockType
from ..pipeline.helpers.id_generator import generate_block_id

class HTMLExtractor(BaseExtractor):
    """
    Extractor for HTML files.
    
    Uses BeautifulSoup to parse HTML and extract semantic content,
    removing navigation, footers, scripts, and other boilerplate.
    """
    
    SUPPORTED_EXTENSIONS = [".html", ".htm", ".xhtml"]
    FORMAT = DocumentFormat.HTML
    
    # Tags to completely remove
    REMOVE_TAGS: Set[str] = {
        'script', 'style', 'noscript', 'iframe', 'svg', 'canvas',
        'nav', 'footer', 'header', 'aside', 'form', 'button',
        'input', 'select', 'textarea', 'meta', 'link', 'head',
        'pre', 'code'
    }
    
    # Tags that indicate boilerplate by class/id
    BOILERPLATE_PATTERNS: List[str] = [
        'nav', 'menu', 'sidebar', 'footer', 'header', 'advertisement',
        'ad-', 'ads-', 'social', 'share', 'comment', 'related',
        'breadcrumb', 'pagination', 'cookie', 'popup', 'modal',
    ]
    
    # Heading tags
    HEADING_TAGS: Set[str] = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
    
    # Block-level tags that typically contain content
    CONTENT_TAGS: Set[str] = {
        'p', 'div', 'article', 'section', 'main', 'blockquote',
        'table', 'ul', 'ol', 'dl', 'figure',
    }
    
    def __init__(self, remove_boilerplate: bool = True, min_text_length: int = 10):
        """
        Initialize HTML extractor.
        
        Args:
            remove_boilerplate: Whether to remove nav, footer, etc.
            min_text_length: Minimum text length for a block.
        """
        super().__init__()
        
        if not BS4_AVAILABLE:
            raise ImportError(
                "BeautifulSoup4 is required for HTML extraction. "
                "Install it with: pip install beautifulsoup4 lxml"
            )
        
        self.remove_boilerplate = remove_boilerplate
        self.min_text_length = min_text_length
    
    def extract(self, file_path: Union[str, Path]) -> List[ExtractedBlock]:
        """
        Extract blocks from an HTML file.
        
        Args:
            file_path: Path to the HTML file.
            
        Returns:
            List of ExtractedBlock objects.
        """
        path = self._validate_file(file_path)
        
        try:
            content = self._read_file_text(path)
            return self._parse_content(content)
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract HTML from {path}: {str(e)}",
                file_path=str(path),
                cause=e
            )
    
    def _parse_content(self, content: str) -> List[ExtractedBlock]:
        """
        Parse HTML content into blocks.
        
        Args:
            content: Raw HTML content.
            
        Returns:
            List of ExtractedBlock objects.
        """
        soup = BeautifulSoup(content, 'lxml')
        
        # Remove unwanted tags
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Remove boilerplate by class/id
        if self.remove_boilerplate:
            self._remove_boilerplate(soup)
        
        blocks = []
        offset_counter = [0]  # Mutable counter for tracking offset
        
        # Find main content area if exists
        main_content = soup.find('main') or soup.find('article') or soup.find('body') or soup
        
        # Process the content
        self._process_element(main_content, blocks, offset_counter)
        
        return blocks
    
    def _remove_boilerplate(self, soup: BeautifulSoup) -> None:
        """Remove boilerplate elements based on class/id patterns."""
        for element in soup.find_all(True):  # All tags
            if not hasattr(element, 'attrs'):
                continue
            
            # Check class and id attributes
            classes = element.get('class', [])
            if isinstance(classes, str):
                classes = [classes]
            
            element_id = element.get('id', '')
            
            # Combine for checking
            identifiers = ' '.join(classes + [element_id]).lower()
            
            for pattern in self.BOILERPLATE_PATTERNS:
                if pattern in identifiers:
                    element.decompose()
                    break
    
    def _process_element(self, element, blocks: List[ExtractedBlock], offset_counter: List[int]) -> None:
        """
        Recursively process an HTML element.
        
        Args:
            element: BeautifulSoup element.
            blocks: List to append blocks to.
            offset_counter: Mutable offset counter.
        """
        if isinstance(element, NavigableString):
            return
        
        if not isinstance(element, Tag):
            return
        
        tag_name = element.name.lower() if element.name else ''
        
        # Handle headings
        if tag_name in self.HEADING_TAGS:
            text = self._get_text(element)
            if text and len(text) >= self.min_text_length:
                level = int(tag_name[1])  # h1 -> 1, h2 -> 2, etc.
                
                block = ExtractedBlock(
                    block_id=generate_block_id(),
                    raw_text=text,
                    block_type=BlockType.HEADING,
                    source_offset=offset_counter[0],
                    heading_level=level,
                    metadata={
                        "source_format": "html",
                        "html_tag": tag_name,
                    }
                )
                blocks.append(block)
                offset_counter[0] += len(text)
            return
        
        # Handle tables
        if tag_name == 'table':
            text = self._extract_table_text(element)
            if text and len(text) >= self.min_text_length:
                block = ExtractedBlock(
                    block_id=generate_block_id(),
                    raw_text=text,
                    block_type=BlockType.TABLE,
                    source_offset=offset_counter[0],
                    metadata={
                        "source_format": "html",
                        "html_tag": tag_name,
                    }
                )
                blocks.append(block)
                offset_counter[0] += len(text)
            return
        
        # Handle lists
        if tag_name in ('ul', 'ol', 'dl'):
            text = self._extract_list_text(element)
            if text and len(text) >= self.min_text_length:
                block = ExtractedBlock(
                    block_id=generate_block_id(),
                    raw_text=text,
                    block_type=BlockType.LIST,
                    source_offset=offset_counter[0],
                    metadata={
                        "source_format": "html",
                        "html_tag": tag_name,
                        "list_type": "ordered" if tag_name == 'ol' else "unordered",
                    }
                )
                blocks.append(block)
                offset_counter[0] += len(text)
            return
        
        # Handle paragraphs
        if tag_name == 'p':
            text = self._get_text(element)
            if text and len(text) >= self.min_text_length:
                block = ExtractedBlock(
                    block_id=generate_block_id(),
                    raw_text=text,
                    block_type=BlockType.PARAGRAPH,
                    source_offset=offset_counter[0],
                    metadata={
                        "source_format": "html",
                        "html_tag": tag_name,
                    }
                )
                blocks.append(block)
                offset_counter[0] += len(text)
            return
        
        # Handle blockquotes
        if tag_name == 'blockquote':
            text = self._get_text(element)
            if text and len(text) >= self.min_text_length:
                block = ExtractedBlock(
                    block_id=generate_block_id(),
                    raw_text=text,
                    block_type=BlockType.PARAGRAPH,
                    source_offset=offset_counter[0],
                    metadata={
                        "source_format": "html",
                        "html_tag": tag_name,
                        "is_quote": True,
                    }
                )
                blocks.append(block)
                offset_counter[0] += len(text)
            return
        
        # For container elements, process children
        for child in element.children:
            self._process_element(child, blocks, offset_counter)
    
    def _get_text(self, element, preserve_whitespace: bool = False) -> str:
        """
        Extract text from an element.
        
        Args:
            element: BeautifulSoup element.
            preserve_whitespace: Whether to preserve whitespace.
            
        Returns:
            Extracted text.
        """
        if preserve_whitespace:
            text = element.get_text()
        else:
            text = element.get_text(separator=' ')
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_table_text(self, table) -> str:
        """
        Extract text from a table element.
        
        Args:
            table: Table element.
            
        Returns:
            Table content as text with | separators.
        """
        rows = []
        
        for tr in table.find_all('tr'):
            cells = []
            for cell in tr.find_all(['td', 'th']):
                cell_text = self._get_text(cell)
                cells.append(cell_text)
            
            if cells:
                rows.append(' | '.join(cells))
        
        return '\n'.join(rows)
    
    def _extract_list_text(self, list_element) -> str:
        """
        Extract text from a list element.
        
        Args:
            list_element: UL, OL, or DL element.
            
        Returns:
            List content as text.
        """
        items = []
        tag_name = list_element.name.lower()
        
        if tag_name == 'dl':
            for child in list_element.children:
                if isinstance(child, Tag):
                    if child.name == 'dt':
                        items.append(f"**{self._get_text(child)}**")
                    elif child.name == 'dd':
                        items.append(f"  {self._get_text(child)}")
        else:
            for i, li in enumerate(list_element.find_all('li', recursive=False)):
                text = self._get_text(li)
                if tag_name == 'ol':
                    items.append(f"{i + 1}. {text}")
                else:
                    items.append(f"• {text}")
        
        return '\n'.join(items)