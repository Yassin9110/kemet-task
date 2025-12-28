"""
Pipeline stages for document ingestion.
"""

from src.pipeline.stages.stage_01_upload import (
    UploadStage,
    UploadInput,
    UploadOutput,
    FileNotFoundError,
    FileTooLargeError,
)
from src.pipeline.stages.stage_02_validation import (
    ValidationStage,
    ValidationInput,
    ValidationOutput,
    UnsupportedFormatError,
    MimeTypeMismatchError,
)
from src.pipeline.stages.stage_03_format_detection import (
    FormatDetectionStage,
    FormatDetectionInput,
    FormatDetectionOutput,
    FormatDetectionError,
)
from src.pipeline.stages.stage_04_extraction import (
    ExtractionStage,
    ExtractionInput,
    ExtractionOutput,
    ExtractionError,
    NoContentError,
)
from src.pipeline.stages.stage_05_normalization import (
    NormalizationStage,
    NormalizationInput,
    NormalizationOutput,
    NormalizationChain,
)
from src.pipeline.stages.stage_06_language_detection import (
    LanguageDetectionStage,
    LanguageDetectionInput,
    LanguageDetectionOutput,
    LanguageBlock,
)
from src.pipeline.stages.stage_07_structure_parsing import (
    StructureParsingStage,
    StructureParsingInput,
    StructureParsingOutput,
    StructuredBlock,
    SectionNode,
)
from src.pipeline.stages.stage_08_chunking import (
    ChunkingStage,
    ChunkingInput,
    ChunkingOutput,
)
from src.pipeline.stages.stage_09_embedding import (
    EmbeddingStage,
    EmbeddingInput,
    EmbeddingOutput,
)
from src.pipeline.stages.stage_10_graph_building import (
    GraphBuildingStage,
    GraphBuildingInput,
    GraphBuildingOutput,
)
from src.pipeline.stages.stage_11_storage import (
    StorageStage,
    StorageInput,
    StorageOutput,
    StorageError,
    StorageCleanup,
)

__all__ = [
    # Stage 1
    "UploadStage",
    "UploadInput",
    "UploadOutput",
    "FileNotFoundError",
    "FileTooLargeError",
    # Stage 2
    "ValidationStage",
    "ValidationInput",
    "ValidationOutput",
    "UnsupportedFormatError",
    "MimeTypeMismatchError",
    # Stage 3
    "FormatDetectionStage",
    "FormatDetectionInput",
    "FormatDetectionOutput",
    "FormatDetectionError",
    # Stage 4
    "ExtractionStage",
    "ExtractionInput",
    "ExtractionOutput",
    "ExtractionError",
    "NoContentError",
    # Stage 5
    "NormalizationStage",
    "NormalizationInput",
    "NormalizationOutput",
    "NormalizationChain",
    # Stage 6
    "LanguageDetectionStage",
    "LanguageDetectionInput",
    "LanguageDetectionOutput",
    "LanguageBlock",
    # Stage 7
    "StructureParsingStage",
    "StructureParsingInput",
    "StructureParsingOutput",
    "StructuredBlock",
    "SectionNode",
    # Stage 8
    "ChunkingStage",
    "ChunkingInput",
    "ChunkingOutput",
    # Stage 9
    "EmbeddingStage",
    "EmbeddingInput",
    "EmbeddingOutput",
    # Stage 10
    "GraphBuildingStage",
    "GraphBuildingInput",
    "GraphBuildingOutput",
    # Stage 11
    "StorageStage",
    "StorageInput",
    "StorageOutput",
    "StorageError",
    "StorageCleanup",
]