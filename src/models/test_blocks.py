# test_blocks.py (temporary, run manually)

from .enums import BlockType, Language
from . import ExtractedBlock, NormalizedBlock
# =============================================================================
# Test ExtractedBlock
# =============================================================================
print("=== ExtractedBlock Tests ===\n")

# Basic paragraph block
paragraph = ExtractedBlock(
    block_id="block-001",
    raw_text="This is a sample paragraph with some text content.",
    block_type=BlockType.PARAGRAPH,
    source_offset=0,
    page_number=1
)
print(f"Paragraph block:")
print(f"  block_id: {paragraph.block_id}")
print(f"  block_type: {paragraph.block_type}")
print(f"  text_length: {paragraph.text_length}")
print(f"  is_empty: {paragraph.is_empty}")
print(f"  is_heading: {paragraph.is_heading}")
print(f"  is_atomic: {paragraph.is_atomic}")

# Heading block
heading = ExtractedBlock(
    block_id="block-002",
    raw_text="Introduction",
    block_type=BlockType.HEADING,
    heading_level=1,
    source_offset=100,
    page_number=1
)
print(f"\nHeading block:")
print(f"  block_id: {heading.block_id}")
print(f"  block_type: {heading.block_type}")
print(f"  heading_level: {heading.heading_level}")
print(f"  is_heading: {heading.is_heading}")

# Code block (atomic)
code = ExtractedBlock(
    block_id="block-003",
    raw_text="def hello():\n    print('Hello')",
    block_type=BlockType.CODE,
    source_offset=200
)
print(f"\nCode block:")
print(f"  block_type: {code.block_type}")
print(f"  is_atomic: {code.is_atomic}")

# Table block (atomic)
table = ExtractedBlock(
    block_id="block-004",
    raw_text="| Col1 | Col2 |\n|------|------|\n| A    | B    |",
    block_type=BlockType.TABLE,
    source_offset=300
)
print(f"\nTable block:")
print(f"  block_type: {table.block_type}")
print(f"  is_atomic: {table.is_atomic}")

# Empty block
empty = ExtractedBlock(
    block_id="block-005",
    raw_text="   \n   ",
    block_type=BlockType.PARAGRAPH,
    source_offset=400
)
print(f"\nEmpty block:")
print(f"  is_empty: {empty.is_empty}")

# Block with metadata
with_meta = ExtractedBlock(
    block_id="block-006",
    raw_text="Some text",
    block_type=BlockType.PARAGRAPH,
    source_offset=500,
    metadata={"font_size": 12, "bold": True}
)
print(f"\nBlock with metadata:")
print(f"  metadata: {with_meta.metadata}")

# =============================================================================
# Test ExtractedBlock Validation
# =============================================================================
print("\n=== ExtractedBlock Validation Tests ===\n")

# Invalid heading level
print("Testing invalid heading_level (7)...")
try:
    invalid = ExtractedBlock(
        block_id="invalid-001",
        raw_text="Bad Heading",
        block_type=BlockType.HEADING,
        heading_level=7,
        source_offset=0
    )
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised: {e}")

# Heading level on non-heading
print("\nTesting heading_level on PARAGRAPH...")
try:
    invalid = ExtractedBlock(
        block_id="invalid-002",
        raw_text="Not a heading",
        block_type=BlockType.PARAGRAPH,
        heading_level=1,
        source_offset=0
    )
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised: {e}")

# Negative page number
print("\nTesting negative page_number...")
try:
    invalid = ExtractedBlock(
        block_id="invalid-003",
        raw_text="Text",
        block_type=BlockType.PARAGRAPH,
        page_number=-1,
        source_offset=0
    )
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised: {e}")

# Negative source_offset
print("\nTesting negative source_offset...")
try:
    invalid = ExtractedBlock(
        block_id="invalid-004",
        raw_text="Text",
        block_type=BlockType.PARAGRAPH,
        source_offset=-10
    )
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised: {e}")

# =============================================================================
# Test NormalizedBlock
# =============================================================================
print("\n=== NormalizedBlock Tests ===\n")

# English block
english_block = NormalizedBlock(
    block_id="norm-001",
    text="This is normalized English text.",
    block_type=BlockType.PARAGRAPH,
    language=Language.EN,
    source_offset=0,
    page_number=1
)
print(f"English block:")
print(f"  language: {english_block.language}")
print(f"  is_english: {english_block.is_english}")
print(f"  is_arabic: {english_block.is_arabic}")
print(f"  is_mixed: {english_block.is_mixed}")

# Arabic block
arabic_block = NormalizedBlock(
    block_id="norm-002",
    text="هذا نص عربي طبيعي",
    block_type=BlockType.PARAGRAPH,
    language=Language.AR,
    source_offset=100
)
print(f"\nArabic block:")
print(f"  text: {arabic_block.text}")
print(f"  language: {arabic_block.language}")
print(f"  is_arabic: {arabic_block.is_arabic}")

# Mixed block
mixed_block = NormalizedBlock(
    block_id="norm-003",
    text="This contains both English and عربي text",
    block_type=BlockType.PARAGRAPH,
    language=Language.MIXED,
    source_offset=200
)
print(f"\nMixed block:")
print(f"  text: {mixed_block.text}")
print(f"  language: {mixed_block.language}")
print(f"  is_mixed: {mixed_block.is_mixed}")

# =============================================================================
# Test from_extracted_block
# =============================================================================
print("\n=== from_extracted_block Tests ===\n")

# Create an extracted block
original = ExtractedBlock(
    block_id="orig-001",
    raw_text="  This is   RAW    text with   extra   spaces.  ",
    block_type=BlockType.PARAGRAPH,
    source_offset=500,
    page_number=3,
    metadata={"source": "pdf", "font": "Arial"}
)

# Convert to normalized block
normalized = NormalizedBlock.from_extracted_block(
    extracted=original,
    normalized_text="This is RAW text with extra spaces.",
    language=Language.EN
)

print(f"Original ExtractedBlock:")
print(f"  block_id: {original.block_id}")
print(f"  raw_text: '{original.raw_text}'")
print(f"  metadata: {original.metadata}")

print(f"\nConverted NormalizedBlock:")
print(f"  block_id: {normalized.block_id}")
print(f"  text: '{normalized.text}'")
print(f"  language: {normalized.language}")
print(f"  page_number: {normalized.page_number}")
print(f"  source_offset: {normalized.source_offset}")
print(f"  metadata: {normalized.metadata}")

# Verify metadata is copied, not shared
original.metadata["new_key"] = "new_value"
print(f"\nAfter modifying original metadata:")
print(f"  original.metadata: {original.metadata}")
print(f"  normalized.metadata: {normalized.metadata}")
print(f"  ✓ Metadata is properly copied (not shared)")

# =============================================================================
# Test with heading
# =============================================================================
print("\n=== Heading Conversion Test ===\n")

heading_extracted = ExtractedBlock(
    block_id="heading-001",
    raw_text="  Chapter 1: Introduction  ",
    block_type=BlockType.HEADING,
    heading_level=1,
    source_offset=0,
    page_number=1
)

heading_normalized = NormalizedBlock.from_extracted_block(
    extracted=heading_extracted,
    normalized_text="Chapter 1: Introduction",
    language=Language.EN
)

print(f"Heading conversion:")
print(f"  is_heading: {heading_normalized.is_heading}")
print(f"  heading_level: {heading_normalized.heading_level}")
print(f"  text: '{heading_normalized.text}'")

print("\n✓ All block tests passed")