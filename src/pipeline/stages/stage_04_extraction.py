"""
Stage 4: Text Extraction

Routes documents to the appropriate extractor based on format
and extracts structured text blocks.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Type
from logging import Logger

from src.config.settings import PipelineConfig
from src.models.enums import DocumentFormat
from src.models.blocks import ExtractedBlock
from src.extractors.base import BaseExtractor
from src.extractors.pdf_extractor import PDFExtractor
from src.extractors.docx_extractor import DOCXExtractor
from src.extractors.html_extractor import HTMLExtractor
from src.extractors.markdown_extractor import MarkdownExtractor
from src.extractors.txt_extractor import TXTExtractor
import traceback

@dataclass
class ExtractionInput:
    """Input for the extraction stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat


@dataclass
class ExtractionOutput:
    """Output from the extraction stage."""
    document_id: str
    stored_path: str
    format: DocumentFormat
    blocks: List[ExtractedBlock]
    total_pages: int
    extraction_warnings: List[str]


class ExtractionError(Exception):
    """Raised when text extraction fails."""
    pass


class NoContentError(Exception):
    """Raised when no content could be extracted from the document."""
    pass


class ExtractionStage:
    """
    Stage 4: Text Extraction
    
    Responsibilities:
    - Route to correct extractor based on format
    - Extract text with structural hints
    - Preserve page numbers where available
    - Handle partial extraction gracefully
    """
    
    STAGE_NAME = "Text Extraction"
    STAGE_NUMBER = 4
    TOTAL_STAGES = 11
    
    # Format to extractor mapping
    EXTRACTOR_MAP: Dict[DocumentFormat, Type[BaseExtractor]] = {
        DocumentFormat.PDF: PDFExtractor,
        DocumentFormat.DOCX: DOCXExtractor,
        DocumentFormat.HTML: HTMLExtractor,
        DocumentFormat.MARKDOWN: MarkdownExtractor,
        DocumentFormat.TXT: TXTExtractor,
    }
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the extraction stage.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self._extractors: Dict[DocumentFormat, BaseExtractor] = {}
    
    def _get_extractor(self, format: DocumentFormat) -> BaseExtractor:
        """
        Get or create an extractor for the given format.
        
        Args:
            format: Document format.
            
        Returns:
            Extractor instance for the format.
            
        Raises:
            ExtractionError: If no extractor is available for the format.
        """
        if format not in self._extractors:
            if format not in self.EXTRACTOR_MAP:
                raise ExtractionError(f"No extractor available for format: {format.value}")
            
            extractor_class = self.EXTRACTOR_MAP[format]
            self._extractors[format] = extractor_class()
        
        return self._extractors[format]
    
    def execute(
        self,
        input_data: ExtractionInput,
        logger: Logger
    ) -> ExtractionOutput:
        """
        Execute the extraction stage.
        
        Args:
            input_data: Extraction input with document info and format.
            logger: Logger instance for progress tracking.
            
        Returns:
            ExtractionOutput with extracted blocks and metadata.
            
        Raises:
            ExtractionError: If extraction fails completely.
            NoContentError: If no content could be extracted.
        """
        start_time = time.time()
        
        stored_path = input_data.stored_path
        doc_format = input_data.format
        warnings: List[str] = []
        
        # Get appropriate extractor
        extractor = self._get_extractor(doc_format)
        # Perform extraction
        try:
            blocks = extractor.extract(stored_path)
        except Exception as e:
            # Try to get partial content if possible
            logger.warning(f"  → Extraction error: {str(e)}. Attempting partial extraction...")
            warnings.append(f"Partial extraction due to: {str(e)}")
            raise RuntimeError(f"Forced to raise for testing {str(e)}")
            
            try:
                blocks = self._attempt_partial_extraction(stored_path, doc_format)
            except Exception as partial_error:
                raise ExtractionError(
                    f"Complete extraction failure for {doc_format.value}: \n {traceback.format_exc()}; "
                ) from partial_error
        
        # Validate we got content
        if not blocks:
            raise NoContentError(
                f"No content could be extracted from document. "
                f"Format: {doc_format.value}, Path: {stored_path}"
            )
        
        # Filter out empty blocks
        non_empty_blocks = [
            block for block in blocks
            if block.raw_text and block.raw_text.strip()
        ]
        
        if not non_empty_blocks:
            raise NoContentError(
                f"All extracted blocks were empty. "
                f"Format: {doc_format.value}, Path: {stored_path}"
            )
        
        # Calculate total pages
        total_pages = self._calculate_total_pages(non_empty_blocks)
        
        # Log warnings for low content extraction
        if len(non_empty_blocks) < len(blocks):
            empty_count = len(blocks) - len(non_empty_blocks)
            warnings.append(f"Filtered out {empty_count} empty blocks")
            logger.warning(f"  → Filtered out {empty_count} empty blocks")
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log completion
        logger.info(
            f"[Stage {self.STAGE_NUMBER}/{self.TOTAL_STAGES}] "
            f"{self.STAGE_NAME} ✓ ({duration_ms}ms)"
        )
        logger.info(
            f"  → Extracted {len(non_empty_blocks)} blocks"
            f"{f' from {total_pages} pages' if total_pages > 0 else ''}"
        )
        
        return ExtractionOutput(
            document_id=input_data.document_id,
            stored_path=stored_path,
            format=doc_format,
            blocks=non_empty_blocks,
            total_pages=total_pages,
            extraction_warnings=warnings
        )
    
    def _calculate_total_pages(self, blocks: List[ExtractedBlock]) -> int:
        """
        Calculate total page count from blocks.
        
        Args:
            blocks: List of extracted blocks.
            
        Returns:
            Total page count (0 if no page info available).
        """
        page_numbers = [
            block.page_number
            for block in blocks
            if block.page_number is not None
        ]
        
        if not page_numbers:
            return 0
        
        return max(page_numbers)
    
    def _attempt_partial_extraction(
        self,
        file_path: str,
        doc_format: DocumentFormat
    ) -> List[ExtractedBlock]:
        """
        Attempt to extract partial content when main extraction fails.
        
        This is a fallback that tries simpler extraction methods.
        
        Args:
            file_path: Path to the file.
            doc_format: Document format.
            
        Returns:
            List of extracted blocks (may be incomplete).
            
        Raises:
            ExtractionError: If partial extraction also fails.
        """
        # For text-based formats, try raw text extraction
        if doc_format in [DocumentFormat.TXT, DocumentFormat.MARKDOWN]:
            txt_extractor = TXTExtractor()
            return txt_extractor.extract(file_path)
        
        # For HTML, try basic text extraction
        if doc_format == DocumentFormat.HTML:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Strip HTML tags crudely as last resort
                import re
                text = re.sub(r'<[^>]+>', ' ', content)
                text = re.sub(r'\s+', ' ', text).strip()
                
                if text:
                    from src.models.enums import BlockType
                    from src.pipeline.helpers.id_generator import generate_block_id
                    
                    return [ExtractedBlock(
                        block_id=generate_block_id(),
                        raw_text=text,
                        page_number=None,
                        structural_hint=BlockType.PARAGRAPH,
                        heading_level=None,
                        source_offset=0
                    )]
            except Exception:
                pass
        
        raise ExtractionError(f"Partial extraction failed for {doc_format.value}")