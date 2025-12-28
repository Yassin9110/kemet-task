# kemet-task

## 1. Executive Summary

### 1.1 What Is This System?

The **Multilingual RAG Ingestion Pipeline** is a document processing system that prepares documents for Retrieval-Augmented Generation (RAG). It takes documents in various formats (PDF, DOCX, HTML, Markdown, TXT), processes them through multiple stages, and produces searchable chunks that can be used to augment LLM responses with relevant context.

### 1.2 Key Capabilities

| Capability                      | Description                                                          |
|---------------------------------|----------------------------------------------------------------------|
| Multi-format Support            | Processes PDF, DOCX, HTML, Markdown, and plain text files            |
| Multilingual                    | Full support for Arabic, English, and mixed-language documents       |
| Hierarchical Chunking           | Creates parent-child chunk relationships for context preservation    |
| Vector Search                   | Embeds chunks using Cohere's multilingual model for semantic search  |
| Graph Relationships             | Maintains sibling links and semantic similarity edges between chunks |
| Context Expansion               | Retrieves surrounding context for more complete LLM responses        |


### 1.3 What Does It Produce?

For each ingested document, the pipeline produces:

1. **Parent Chunks** - Larger text segments (~512 tokens) representing document sections
2. **Child Chunks** - Smaller text segments (~128 tokens) optimized for retrieval
3. **Vector Embeddings** - 1024-dimensional vectors for each child chunk
4. **Semantic Edges** - Links between semantically similar chunks
5. **Structured Metadata** - Document info, section paths, language tags, etc.

### 1.4 How Will Teams Use This?

| Team | Usage |
| :--- | :--- |
| UI Team | Call the search API to find relevant chunks, display results with section paths |
| LLM Integration Team |  Use `get_retrieval_context()` to get formatted context for prompt augmentation |
| Backend Team |  Integrate document upload endpoints with the `ingest()` function |

---

## 2. System Overview

```text
                ┌─────────────────────────────────────────────────────────────────────────────┐
                │                          USER UPLOADS DOCUMENT                              │
                └─────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           INGESTION PIPELINE                                                │
│                                                                                                             │
│ Document → Validate → Extract → Normalize → Detect Language → Parse → Chunk → Embed → Build Graph → Store   │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                ┌─────────────────────────────────────────────────────────────────────────────┐
                │                               STORAGE LAYER                                 │
                │                                                                             │
                │       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
                │       │  Raw Files  │ │    JSON     │ │   Chroma    │ │    Logs     │       │
                │       │ (Immutable) │ │  Metadata   │ │  (Vectors)  │ │             │       │
                │       └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
                │                                                                             │
                └─────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                ┌─────────────────────────────────────────────────────────────────────────────┐
                │                               RETRIEVAL LAYER                               │
                │                                                                             │
                │      Search Query → Embed Query → Vector Search → Expand Context → Return   │
                │                                                                             │
                └─────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                ┌─────────────────────────────────────────────────────────────────────────────┐
                │                            LLM RECEIVES CONTEXT                             │
                │                                                                             │
                │        "Based on the following context, answer the user's question..."      │
                │                                                                             │
                └─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Supported Document Formats

| Format | Extension | Extraction Method | Structure Detection |
|--------|-----------|-------------------|---------------------|
| PDF | `.pdf` | Digital text extraction (no OCR) | Page numbers, basic headings |
| DOCX | `.docx` | XML parsing | Headings, lists, tables |
| HTML | `.html`, `.htm` | DOM parsing + boilerplate removal | Semantic tags (h1-h6, p, ul, ol, table) |
| Markdown | `.md`, `.markdown` | AST parsing | Heading hierarchy, lists |
| Plain Text | `.txt` | Raw text | Line breaks only |

### 2.3 Supported Languages

| Language | Code | Special Handling |
|----------|------|------------------|
| Arabic | `ar` | Alef normalization, Tashkeel removal, Tatweel removal, 20% extra chunk overlap |
| English | `en` | Standard processing |
| Mixed | `mixed` | Detected when both languages present ≥20%, uses Arabic overlap settings |

---

## 3. Architecture

### 3.1 Directory Structure

```text
multilingual-rag-pipeline/
│
├── src/
│ ├── init.py
│ │
│ ├── config/
│ │ ├── init.py
│ │ └── settings.py # PipelineConfig dataclass
│ │
│ ├── models/
│ │ ├── init.py
│ │ ├── document.py # Document, IngestionStatus
│ │ ├── chunks.py # ParentChunk, ChildChunk
│ │ ├── blocks.py # ExtractedBlock, NormalizedBlock
│ │ ├── edges.py # SemanticEdge
│ │ └── enums.py # DocumentFormat, Language, BlockType
│ │
│ ├── pipeline/
│ │ ├── init.py
│ │ ├── orchestrator.py # Main pipeline controller
│ │ │
│ │ ├── stages/
│ │ │ ├── init.py
│ │ │ ├── stage_01_upload.py
│ │ │ ├── stage_02_validation.py
│ │ │ ├── stage_03_format_detection.py
│ │ │ ├── stage_04_extraction.py
│ │ │ ├── stage_05_normalization.py
│ │ │ ├── stage_06_language_detection.py
│ │ │ ├── stage_07_structure_parsing.py
│ │ │ ├── stage_08_chunking.py
│ │ │ ├── stage_09_embedding.py
│ │ │ ├── stage_10_graph_building.py
│ │ │ └── stage_11_storage.py
│ │ │
│ │ └── helpers/
│ │ ├── init.py
│ │ ├── tokenizer.py # Token counting utilities
│ │ ├── hasher.py # SHA256 hashing
│ │ └── id_generator.py # UUID generation
│ │
│ ├── extractors/
│ │ ├── init.py
│ │ ├── base.py # Abstract base extractor
│ │ ├── pdf_extractor.py
│ │ ├── docx_extractor.py
│ │ ├── html_extractor.py
│ │ ├── markdown_extractor.py
│ │ └── txt_extractor.py
│ │
│ ├── normalizers/
│ │ ├── init.py
│ │ ├── base.py # Abstract base normalizer
│ │ ├── unicode_normalizer.py
│ │ ├── arabic_normalizer.py
│ │ └── whitespace_normalizer.py
│ │
│ ├── chunking/
│ │ ├── init.py
│ │ ├── hierarchy_resolver.py # Decides 1/2/3 level hierarchy
│ │ ├── parent_chunker.py # Creates parent chunks
│ │ ├── child_chunker.py # Creates child chunks
│ │ └── atomic_detector.py # Detects tables, code blocks
│ │
│ ├── embedding/
│ │ ├── init.py
│ │ ├── base.py # Abstract base embedder
│ │ └── cohere_embedder.py # Cohere implementation
│ │
│ ├── storage/
│ │ ├── init.py
│ │ ├── file_storage.py # Raw file storage
│ │ ├── json_storage.py # JSON file operations
│ │ └── vector_storage.py # Chroma operations
│ │
│ ├── retrieval/
│ │ ├── init.py
│ │ ├── searcher.py # Vector search
│ │ └── context_expander.py # Parent/sibling expansion
│ │
│ └── logging/
│ ├── init.py
│ └── pipeline_logger.py # Custom logging setup
│
├── data/ # Created at runtime
│ ├── raw/ # Immutable original files
│ ├── chroma/ # Vector database
│ ├── documents.json # Document metadata
│ ├── parents.json # Parent chunks
│ ├── children.json # Child chunks backup
│ └── edges.json # Semantic edges
│
├── logs/ # Created at runtime
│ └── pipeline.log
│
├── main.py # Entry point
├── requirements.txt
└── README.md
```
### 3.2 Layer Architecture

The system is organized into layers, where each layer only depends on layers above it:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Layer 0: Foundation (No Dependencies)                     │
│ ─────────────────────────────────────────────────────────────────────────── │
│           models/enums.py config/settings.py helpers/hasher.py              │
│                   helpers/id_generator.py                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Layer 1: Data Models                              │
│ ─────────────────────────────────────────────────────────────────────────── │
│           models/blocks.py models/chunks.py models/document.py              │
│     models/edges.py helpers/tokenizer.py logging/pipeline_logger.py         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Layer 2: Abstract Interfaces & Storage                     │
│ ─────────────────────────────────────────────────────────────────────────── │
│            extractors/base.py normalizers/base.py embedding/base.py         │
│                storage/file_storage.py storage/json_storage.py              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Layer 3: Implementations                          │
│ ─────────────────────────────────────────────────────────────────────────── │
│      extractors/pdf_extractor.py extractors/docx_extractor.py               │
│      extractors/html_extractor.py extractors/markdown_extractor.py          │
│      extractors/txt_extractor.py                                            │
│      normalizers/unicode_normalizer.py normalizers/arabic_normalizer.py     │
│      normalizers/whitespace_normalizer.py                                   │
│      embedding/cohere_embedder.py storage/vector_storage.py                 │
│      chunking/hierarchy_resolver.py chunking/parent_chunker.py              │
│      chunking/child_chunker.py chunking/atomic_detector.py                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Layer 4: Pipeline Stages                            │
│ ─────────────────────────────────────────────────────────────────────────── │
│          stages/stage_01_upload.py through stages/stage_11_storage.py       │
│            retrieval/searcher.py retrieval/context_expander.py              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Layer 5: Orchestration & Entry Points                     │
│ ─────────────────────────────────────────────────────────────────────────── │
│                     pipeline/orchestrator.py main.py                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow

```text
    ┌──────────┐
    │ Raw File │
    └────┬─────┘
         │
         ▼
┌──────────────────┐
│ ExtractedBlock[] │ Stage 4: Raw text with structural hints
└────────┬─────────┘
         │
         ▼
┌───────────────────┐
│ NormalizedBlock[] │ Stage 5: Cleaned and normalized text
└─────────┬─────────┘
          │
          ▼
┌─────────────────┐
│ LanguageBlock[] │ Stage 6: Language tagged (ar/en/mixed)
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ StructuredBlock[]│ Stage 7: Section paths attached
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│ ParentChunk[] + ChildChunk[] │ Stage 8: Hierarchical chunks
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ ChildChunk[] with embeddings │ Stage 9: Vector embeddings added
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ SemanticEdge[]               │ Stage 10: Similarity edges
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Persisted to JSON + Chroma   │ Stage 11: Final storage
└──────────────────────────────┘
```
---

## 4. Pipeline Stages

The pipeline consists of 11 sequential stages. Each stage has a single responsibility and produces a specific output.

### 4.1 Stage Overview

| # | Stage Name | Input | Output | Purpose |
|---|------------|-------|--------|---------|
| 1 | Upload | File path | UploadOutput | Accept file, assign ID, copy to storage |
| 2 | Validation | UploadOutput | ValidationOutput | Validate format, compute hash, check duplicates |
| 3 | Format Detection | ValidationOutput | FormatDetectionOutput | Determine document format |
| 4 | Extraction | FormatDetectionOutput | ExtractionOutput | Extract text blocks |
| 5 | Normalization | ExtractionOutput | NormalizationOutput | Clean and normalize text |
| 6 | Language Detection | NormalizationOutput | LanguageDetectionOutput | Detect language per block |
| 7 | Structure Parsing | LanguageDetectionOutput | StructureParsingOutput | Build section hierarchy |
| 8 | Chunking | StructureParsingOutput | ChunkingOutput | Create parent/child chunks |
| 9 | Embedding | ChunkingOutput | EmbeddingOutput | Generate vector embeddings |
| 10 | Graph Building | EmbeddingOutput | GraphBuildingOutput | Create semantic edges |
| 11 | Storage | GraphBuildingOutput | StorageOutput | Persist all data |

### 4.2 Stage 1: File Upload

**Purpose:** Accept a file, assign a unique document ID, and copy the file to immutable storage.

**What It Does:**
1. Validates the file exists
2. Checks file is not empty
3. Checks file size is within limits (default: 50 MB)
4. Generates a UUID v4 document ID
5. Copies the file to `data/raw/{document_id}.{extension}`
6. Extracts basic file info (name, extension, size)


### 4.3 Stage 2: Validation & Fingerprinting

**Purpose**: Validate the file format and compute a fingerprint for duplicate detection.

**What It Does:**

1. Validates MIME type matches supported formats
2. Cross-checks MIME type with file extension
3. Computes SHA256 hash of file contents
4. Checks if a file with the same hash already exists (duplicate detection)

**Duplicate Handling:**
- If a duplicate is detected, a warning is logged
- The file is still processed as a new document (per specification)
- This allows the same content to exist under different document IDs if needed

### 4.4 Stage 3: Format Detection

**Purpose**: Determine the document format for routing to the correct extractor.

**What It Does:**

1. Checks MIME type against known formats (primary method)
2. Falls back to file extension if MIME is ambiguous
3. Uses content sniffing (magic bytes) as last resort
4. Returns a standardized `DocumentFormat` enum

**Detection Priority:**

1. MIME type (most reliable)
2. File extension
3. Magic bytes (first 512 bytes of file)
4. Markdown heuristics (looks for `#`, `*`, `**`, etc.)

### 4.5 Stage 4: Text Extraction

**Purpose:** Extract text content from the document while preserving structural information.

**What It Does:**

1. Routes to the appropriate extractor based on format
2. Extracts text blocks with structural hints (heading, paragraph, table, list)
3. Preserves page numbers (for PDF)
4. Preserves heading levels (1-6)
5. Handles extraction errors gracefully with partial extraction fallback

**Extractor Behaviors:**

| Format  | Extractor  |  Special Handling   |
| :--- | :--- | :--- |
| PDF           | PDFExtractor       | Page numbers, heading detection from font size  |
| DOCX   | DOCXExtractor      | Heading styles, list detection, table preservation  |
| HTML           | HTMLExtractor       | Boilerplate removal, semantic tag interpretation  |
| Markdown   | MarkdownExtractor      | AST parsing, heading hierarchy  |
| TXT           | TXTExtractor       | Line break preservation  |

### 4.6 Stage 5: Normalization & Cleaning

**Purpose:** Clean and normalize text for consistent processing.

**What It Does:**

1. Unicode Normalization (NFC): Ensures consistent character representation
2. Invisible Character Removal: Removes zero-width characters, etc.
3. Arabic Normalization:
    - Alef normalization (أ إ آ → ا)
    - Tashkeel removal (removes diacritics: فَتْحَة → فتحة)
    - Tatweel removal (removes stretching: كـــتاب → كتاب)
4. Whitespace Normalization: Collapses multiple spaces/newlines

### 4.7 Stage 6: Language Detection

**Purpose:** Detect the language of each block for language-aware processing.

**What It Does:**

1. Analyzes each block independently (not document-level)
2. Counts Arabic vs. English characters
3. Classifies as ar, en, or mixed
4. Calculates document-level language distribution

**Classification Rules:**

If Arabic ≥ 80% and English < 20%: `ar`
If English ≥ 80% and Arabic < 20%: `en`
If both Arabic ≥ 20% and English ≥ 20%: `mixed`

### 4.8 Stage 7: Structure Parsing
**Purpose:** Build a hierarchical document structure based on headings.

**What It Does:**

1. Identifies heading blocks
2. Builds a tree structure based on heading levels (h1 → h2 → h3...)
3. Assigns each block to a section
4. Creates section paths for each block (e.g., ["Chapter 1", "Section 1.1"])

**Example Section Tree:**

```text
Document Root
├── Chapter 1: Introduction
│   ├── 1.1 Background
│   └── 1.2 Objectives
├── Chapter 2: Methodology
│   ├── 2.1 Data Collection
│   └── 2.2 Analysis
└── Chapter 3: Results
```

### 4.9 Stage 8: Chunking
**Purpose:** Create parent and child chunks based on document size and structure.

**What It Does:**

**Step 1: Determine Hierarchy Depth**

| Document Size  | Hierarchy  |  Explanation   |
| :--- | :--- | :--- |
| < 1,500 tokens           | 1 level       | Small doc: child chunks only, no parents  |
| 1,500 - 10,000 tokens   | 2 levels      | Medium doc: parent chunks + child chunks  |
| > 10,000 tokens           | 3 levels       | Large doc: sections as parents + child chunks  |


**Step 2: Create Parent Chunks (if applicable)**

- Target size: ~512 tokens
- Split at sentence boundaries
- Respect section boundaries from Stage 7

**Step 3: Create Child Chunks**

- Target size: ~128 tokens
- Split at sentence boundaries
- Add overlap between chunks:
    - English: 50 tokens
    - Arabic/Mixed: 60 tokens (20% more)
- Never split atomic units (tables, lists)

**Step 4: Link Siblings**

- Each child chunk gets prev_chunk_id and next_chunk_id
- Enables context expansion during retrieval


**Visual Example:**

```text
Document (5,000 tokens) → 2-level hierarchy
│
├── Parent Chunk 1 (480 tokens)
│   ├── Child Chunk 1 (128 tokens) ← overlap → Child Chunk 2
│   ├── Child Chunk 2 (128 tokens) ← overlap → Child Chunk 3
│   └── Child Chunk 3 (128 tokens)
│
├── Parent Chunk 2 (510 tokens)
│   ├── Child Chunk 4 (128 tokens) ← overlap → Child Chunk 5
│   ├── Child Chunk 5 (128 tokens) ← overlap → Child Chunk 6
│   └── Child Chunk 6 (128 tokens)
│
└── ...
```

### 4.10 Stage 9: Embedding
**Purpose:** Generate vector embeddings for semantic search.

**What It Does:**

1. Embeds child chunks only (parents are stored as text only)
2. Uses Cohere `embed-multilingual-v3.0` model
3. Produces 1024-dimensional vectors
4. Processes in batches for efficiency
5. Handles API errors with retries (3 attempts)

**Why Child Chunks Only?**

- Child chunks are the retrieval units
- Parent chunks provide context but aren't searched directly
- This reduces embedding costs while maintaining retrieval quality

**Embedding Model Details:**

- Model: `embed-multilingual-v3.0`
- Dimension: 1024
- Input type for documents: `search_document`
- Input type for queries: `search_query` (different encoding for better retrieval)

Output:
Each `ChildChunk` has its `embedding` field populated with a 1024-dimensional vector.


### 4.11 Stage 10: Graph Building

**Purpose:** Create relationships between chunks for graph-based retrieval.

**What It Does:**

1. Verify Structural Edges (already created in Stage 8):

- Parent → Child: `child.parent_id`
- Child → Prev Sibling: `child.prev_chunk_id`
- Child → Next Sibling: `child.next_chunk_id`

2. Compute Semantic Edges (if enabled):

- Compares embeddings of all child chunks within a document
- Creates edges for pairs with cosine similarity ≥ threshold (default: 0.85)
- Excludes adjacent siblings (they're already connected)

**Semantic Edge Use Cases:**

- "These two chunks discuss the same topic, even though they're in different sections"
- Enables cross-section context retrieval

### 4.12 Stage 11: Storage
**Purpose:** Persist all data to storage.

**What It Does:**

| Data  | Storage Location  |  Format   |
| :--- | :--- | :--- |
| Raw files           | data/raw/{document_id}.{ext}       | Original binary  |
| Document metadata   | data/documents.json      | JSON array  |
| Parent chunks           | data/parents.json       | JSON array  |
| Child chunks (backup)           | data/children.json       | JSON array  |
| Child chunks (vectors)           | data/chroma/       | ChromaDB  |
| Semantic edges           | data/edges.json       | JSON array  |


**Why Dual Storage for Child Chunks?**

- ChromaDB stores vectors + basic metadata for fast search
- JSON backup stores complete chunk data for reconstruction
- Enables recovery if vector DB is corrupted

## 6. Module Reference

### 6.1 Embedding Module
**Location:** `src/embedding/`

**Files:**

- `base.py` - Abstract base class for embedders
- `cohere_embedder.py` - Cohere implementation


### 6.2 Pipeline Module
**Location:** `src/pipeline/`

**Files:**

- `orchestrator.py` - Main pipeline controller
- `stages/` - Individual stage implementations
- `helpers/` - Utility functions

### 6.3 Retrieval Module
**Location:** `src/retrieval/`

**Files:**
- `searcher.py` - Vector search functionality
- `context_expander.py` - Context expansion
### 6.4 Storage Module
**Location:** `src/storage/`

**Files:**

- `file_storage.py` - Raw file operations
- `json_storage.py` - JSON metadata operations
- `vector_storage.py` - ChromaDB operations

