# Multilingual RAG System — File Ingestion, Chunking & Embedding
# Architecture & Engineering Plan (Arabic + English)
# Version: 1.0

----------------------------------------------------------------
GOAL
----------------------------------------------------------------
Design a robust, modular, production-grade ingestion pipeline that:
- Accepts unknown document types (pdf, txt, html, md)
- Works reliably for Arabic, English, and mixed content
- Produces high-quality chunks for Retrieval-Augmented Generation (RAG)
- Is extensible, testable, and observable

----------------------------------------------------------------
HIGH-LEVEL SYSTEM FLOW
----------------------------------------------------------------
1. File Upload
2. File Validation & Fingerprinting
3. Format Detection
4. Text Extraction (format-specific)
5. Normalization & Cleaning
6. Language Detection
7. Structural Parsing
8. Chunking (policy-based)
9. Chunk Validation & Enrichment
10. Embedding
11. Vector Store Ingestion
12. Metadata Indexing & Auditing

----------------------------------------------------------------
FULL PIPELINE WALKTHROUGH
----------------------------------------------------------------

----------------------------------------------------------------
1. FILE UPLOAD
----------------------------------------------------------------
Input:
- file
- Optional metadata (user_id, source, tags)

Responsibilities:
- Accept file via its path
- Enforce size limits
- Assign internal document_id (UUID)
- Persist raw file (object storage)

Output:
- document_id
- new_raw_file_path_stored

Notes:
- Never mutate raw files
- Treat raw file as immutable source of truth

----------------------------------------------------------------
2. FILE VALIDATION & FINGERPRINTING
----------------------------------------------------------------
Responsibilities:
- Validate MIME type
- Validate extension
- Compute file hash (SHA256)
- Detect duplicate uploads

Output:
- validated_file
- file_hash

Edge Cases:
- Incorrect extensions
- Corrupted PDFs
- Empty files

----------------------------------------------------------------
3. FORMAT DETECTION
----------------------------------------------------------------
Decision:
- pdf
- docx
- html
- markdown
- txt

Rules:
- Prefer MIME over extension
- Fallback to content sniffing

Output:
- document_format

----------------------------------------------------------------
4. TEXT EXTRACTION (FORMAT-SPECIFIC)
----------------------------------------------------------------

PDF or docx:
- Try digital text extraction first
- Detect if OCR is required
- Preserve page numbers
- Preserve layout hints (headings, tables)

HTML:
- DOM parsing
- Remove boilerplate (nav, footer, ads)
- Preserve semantic tags

Markdown:
- Parse AST
- Preserve heading hierarchy
- Preserve code blocks

TXT:
- Treat as raw text
- Preserve line breaks

Output:
- extracted_blocks[]

Block schema:
- block_id
- raw_text
- page_number (optional)
- structural_hint (heading, paragraph, table, list, code)
- source_offset

----------------------------------------------------------------
5. NORMALIZATION & CLEANING
----------------------------------------------------------------
Responsibilities:
- Unicode normalization
- Remove invisible characters
- Normalize Arabic variants (optional configurable step)
- Fix encoding issues
- Collapse excessive whitespace

DO NOT:
- Translate
- Rewrite
- Summarize

Output:
- normalized_blocks[]

----------------------------------------------------------------
6. LANGUAGE DETECTION
----------------------------------------------------------------
Responsibilities:
- Detect language per block (not per document)
- Support: ar, en, mixed

Output:
- blocks with language field

Notes:
- Arabic + English may coexist in same document
- Chunking behavior depends on language

----------------------------------------------------------------
7. STRUCTURAL PARSING
----------------------------------------------------------------
Responsibilities:
- Build logical document tree
- Track section hierarchy
- Attach blocks to section paths

Section path example:
["Chapter 2", "Methodology", "Data Collection"]

Output:
- structured_blocks[]

----------------------------------------------------------------
8. CHUNKING (POLICY-BASED)
----------------------------------------------------------------

Chunking is NOT format-based.
Chunking is policy-based.

Chunking Policy Inputs:
- language
- block_type
- section_depth
- token_count
- semantic cohesion

Global Constraints:
- MAX_TOKENS = 512
- MIN_TOKENS = 80
- OVERLAP = 50 (Arabic: +20%)

Chunking Strategy Decision:
- If headings available → section-aware chunking
- If no structure → semantic chunking
- Tables → atomic chunks
- Lists → grouped chunks
- Code → function/class-level chunks

Validation Rules:
- Never split tables
- Never split Arabic mid-sentence
- Merge undersized chunks
- Split oversized chunks recursively

Output:
- chunks[]

Chunk schema:
- chunk_id
- text
- language
- token_count
- block_type
- section_path
- page_range
- source_document_id

----------------------------------------------------------------
9. CHUNK VALIDATION & ENRICHMENT
----------------------------------------------------------------
Responsibilities:
- Quality scoring (noise, length, structure)
- Drop low-quality chunks
- Add retrieval metadata

Metadata:
- language
- source_format
- section_path
- page number (if available)
- block_type
- confidence_score

----------------------------------------------------------------
10. EMBEDDING
----------------------------------------------------------------
Responsibilities:
- Select multilingual embedding model
- Batch embedding
- Normalize vectors

Requirements:
- Arabic + English support
- Consistent tokenization
- Deterministic outputs

Output:
- embeddings[]

----------------------------------------------------------------
11. VECTOR STORE INGESTION
----------------------------------------------------------------
Responsibilities:
- Store vector + metadata
- Support filtering by:
  - language
  - document_id
  - section
  - block_type

Notes:
- Vector DB is not source of truth
- Must be re-buildable

----------------------------------------------------------------
12. METADATA INDEXING & AUDITING
----------------------------------------------------------------
Responsibilities:
- Persist document index
- Track ingestion status
- Enable reprocessing
- Enable deletion


----------------------------------------------------------------
ENGINEERING PRINCIPLES
----------------------------------------------------------------
- Every stage is idempotent
- Every stage can be replayed
- No stage depends on vector DB state
- Chunking is deterministic
- Metadata is first-class
- Arabic is not treated as an edge case

----------------------------------------------------------------
NON-GOALS
----------------------------------------------------------------
- No automatic translation
- No summarization at ingestion
- No LLM calls during ingestion

----------------------------------------------------------------
END OF DOCUMENT
----------------------------------------------------------------
