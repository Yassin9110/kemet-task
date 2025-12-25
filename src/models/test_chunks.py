# test_chunks.py (temporary, run manually)

from datetime import datetime
from .enums import Language, BlockType
from . import ParentChunk, ChildChunk

# =============================================================================
# Test ParentChunk
# =============================================================================
print("=== ParentChunk Tests ===\n")

# Basic parent chunk
parent = ParentChunk(
    parent_id="parent-001",
    document_id="doc-001",
    text="This is the full text of a parent chunk. It contains multiple sentences that provide context for the LLM.",
    token_count=25,
    section_path=["Chapter 1", "Introduction"],
    page_range=(1, 2),
    language=Language.EN,
    child_ids=["child-001", "child-002"]
)

print("Basic ParentChunk:")
print(f"  parent_id: {parent.parent_id}")
print(f"  document_id: {parent.document_id}")
print(f"  token_count: {parent.token_count}")
print(f"  text_length: {parent.text_length}")
print(f"  section_path: {parent.section_path}")
print(f"  section_name: {parent.section_name}")
print(f"  section_depth: {parent.section_depth}")
print(f"  page_range: {parent.page_range}")
print(f"  language: {parent.language}")
print(f"  child_ids: {parent.child_ids}")
print(f"  child_count: {parent.child_count}")
print(f"  has_children: {parent.has_children}")
print(f"  is_empty: {parent.is_empty}")
print(f"  is_english: {parent.is_english}")

# Arabic parent chunk
arabic_parent = ParentChunk(
    parent_id="parent-002",
    document_id="doc-001",
    text="هذا نص عربي طويل يحتوي على عدة جمل لتوفير السياق",
    token_count=15,
    language=Language.AR
)

print(f"\nArabic ParentChunk:")
print(f"  text: {arabic_parent.text}")
print(f"  is_arabic: {arabic_parent.is_arabic}")

# Add child to parent
print(f"\n--- Adding child ---")
print(f"Before: {parent.child_ids}")
parent.add_child("child-003")
print(f"After add_child('child-003'): {parent.child_ids}")
parent.add_child("child-001")  # Duplicate, should not add
print(f"After add_child('child-001') duplicate: {parent.child_ids}")

# =============================================================================
# Test ParentChunk Serialization
# =============================================================================
print("\n=== ParentChunk Serialization ===\n")

parent_dict = parent.to_dict()
print("to_dict() result:")
for key, value in parent_dict.items():
    print(f"  {key}: {value}")

restored_parent = ParentChunk.from_dict(parent_dict)
print(f"\nRestored from dict:")
print(f"  parent_id: {restored_parent.parent_id}")
print(f"  section_path: {restored_parent.section_path}")
print(f"  page_range: {restored_parent.page_range}")
print(f"  language: {restored_parent.language}")

# =============================================================================
# Test ParentChunk Validation
# =============================================================================
print("\n=== ParentChunk Validation ===\n")

# Negative token count
print("Testing negative token_count...")
try:
    invalid = ParentChunk(
        parent_id="invalid-001",
        document_id="doc-001",
        text="Text",
        token_count=-5
    )
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised: {e}")

# Invalid page range
print("\nTesting invalid page_range (end < start)...")
try:
    invalid = ParentChunk(
        parent_id="invalid-002",
        document_id="doc-001",
        text="Text",
        token_count=10,
        page_range=(5, 3)
    )
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised: {e}")

# Empty parent_id
print("\nTesting empty parent_id...")
try:
    invalid = ParentChunk(
        parent_id="",
        document_id="doc-001",
        text="Text",
        token_count=10
    )
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised: {e}")

# =============================================================================
# Test ChildChunk
# =============================================================================
print("\n=== ChildChunk Tests ===\n")

# Basic child chunk
child = ChildChunk(
    chunk_id="child-001",
    document_id="doc-001",
    text="This is a child chunk with smaller text for retrieval.",
    token_count=12,
    language=Language.EN,
    block_type=BlockType.PARAGRAPH,
    parent_id="parent-001",
    section_path=["Chapter 1", "Introduction"],
    page_number=1,
    position_in_parent=0
)

print("Basic ChildChunk:")
print(f"  chunk_id: {child.chunk_id}")
print(f"  document_id: {child.document_id}")
print(f"  parent_id: {child.parent_id}")
print(f"  token_count: {child.token_count}")
print(f"  text_length: {child.text_length}")
print(f"  language: {child.language}")
print(f"  block_type: {child.block_type}")
print(f"  section_path: {child.section_path}")
print(f"  section_name: {child.section_name}")
print(f"  page_number: {child.page_number}")
print(f"  position_in_parent: {child.position_in_parent}")
print(f"  has_parent: {child.has_parent}")
print(f"  has_embedding: {child.has_embedding}")
print(f"  is_first_in_parent: {child.is_first_in_parent}")
print(f"  is_atomic: {child.is_atomic}")

# Child without parent (1-level hierarchy)
orphan_child = ChildChunk(
    chunk_id="child-orphan",
    document_id="doc-002",
    text="Small document chunk.",
    token_count=5,
    language=Language.EN,
    block_type=BlockType.PARAGRAPH
)

print(f"\nOrphan ChildChunk (no parent):")
print(f"  parent_id: {orphan_child.parent_id}")
print(f"  has_parent: {orphan_child.has_parent}")

# Code block (atomic)
code_child = ChildChunk(
    chunk_id="child-code",
    document_id="doc-001",
    text="def hello():\n    print('Hello')",
    token_count=8,
    language=Language.EN,
    block_type=BlockType.CODE
)

print(f"\nCode ChildChunk:")
print(f"  block_type: {code_child.block_type}")
print(f"  is_atomic: {code_child.is_atomic}")

# =============================================================================
# Test Sibling Linking
# =============================================================================
print("\n=== Sibling Linking ===\n")

chunk1 = ChildChunk(
    chunk_id="chunk-1",
    document_id="doc-001",
    text="First chunk",
    token_count=3,
    language=Language.EN,
    block_type=BlockType.PARAGRAPH,
    position_in_parent=0
)

chunk2 = ChildChunk(
    chunk_id="chunk-2",
    document_id="doc-001",
    text="Second chunk",
    token_count=3,
    language=Language.EN,
    block_type=BlockType.PARAGRAPH,
    position_in_parent=1
)

chunk3 = ChildChunk(
    chunk_id="chunk-3",
    document_id="doc-001",
    text="Third chunk",
    token_count=3,
    language=Language.EN,
    block_type=BlockType.PARAGRAPH,
    position_in_parent=2
)

# Link siblings
chunk1.link_next(chunk2.chunk_id)
chunk2.link_prev(chunk1.chunk_id)
chunk2.link_next(chunk3.chunk_id)
chunk3.link_prev(chunk2.chunk_id)

print("After linking:")
print(f"  chunk1: prev={chunk1.prev_chunk_id}, next={chunk1.next_chunk_id}")
print(f"  chunk2: prev={chunk2.prev_chunk_id}, next={chunk2.next_chunk_id}")
print(f"  chunk3: prev={chunk3.prev_chunk_id}, next={chunk3.next_chunk_id}")

print(f"\n  chunk1.has_prev_sibling: {chunk1.has_prev_sibling}")
print(f"  chunk1.has_next_sibling: {chunk1.has_next_sibling}")
print(f"  chunk3.has_prev_sibling: {chunk3.has_prev_sibling}")
print(f"  chunk3.has_next_sibling: {chunk3.has_next_sibling}")

# =============================================================================
# Test Embedding
# =============================================================================
print("\n=== Embedding Tests ===\n")

print(f"Before embedding:")
print(f"  has_embedding: {child.has_embedding}")
print(f"  embedding_dimension: {child.embedding_dimension}")

child.set_embedding([0.1, 0.2, 0.3, 0.4, 0.5])

print(f"\nAfter set_embedding([0.1, 0.2, 0.3, 0.4, 0.5]):")
print(f"  has_embedding: {child.has_embedding}")
print(f"  embedding_dimension: {child.embedding_dimension}")
print(f"  embedding: {child.embedding}")

# =============================================================================
# Test ChildChunk Serialization
# =============================================================================
print("\n=== ChildChunk Serialization ===\n")

child_dict = child.to_dict()
print("to_dict() result (with embedding):")
for key, value in child_dict.items():
    if key == "embedding":
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: {value}")

child_dict_no_embed = child.to_dict(include_embedding=False)
print(f"\nto_dict(include_embedding=False):")
print(f"  'embedding' in dict: {'embedding' in child_dict_no_embed}")

restored_child = ChildChunk.from_dict(child_dict)
print(f"\nRestored from dict:")
print(f"  chunk_id: {restored_child.chunk_id}")
print(f"  language: {restored_child.language}")
print(f"  block_type: {restored_child.block_type}")
print(f"  has_embedding: {restored_child.has_embedding}")

# =============================================================================
# Test ChildChunk Validation
# =============================================================================
print("\n=== ChildChunk Validation ===\n")

# Negative token count
print("Testing negative token_count...")
try:
    invalid = ChildChunk(
        chunk_id="invalid-001",
        document_id="doc-001",
        text="Text",
        token_count=-5,
        language=Language.EN,
        block_type=BlockType.PARAGRAPH
    )
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised: {e}")

# Invalid page number
print("\nTesting page_number = 0...")
try:
    invalid = ChildChunk(
        chunk_id="invalid-002",
        document_id="doc-001",
        text="Text",
        token_count=5,
        language=Language.EN,
        block_type=BlockType.PARAGRAPH,
        page_number=0
    )
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised: {e}")

# Negative position
print("\nTesting negative position_in_parent...")
try:
    invalid = ChildChunk(
        chunk_id="invalid-003",
        document_id="doc-001",
        text="Text",
        token_count=5,
        language=Language.EN,
        block_type=BlockType.PARAGRAPH,
        position_in_parent=-1
    )
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised: {e}")

print("\n✓ All chunk tests passed")