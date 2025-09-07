import pytest
from pyDocumentChunker import MarkdownSplitter
from unittest.mock import patch

# FRD Requirement Being Tested:
# R-3.4.2 (extended): The strategy MUST recognize structural hierarchies, including
# different block-level elements like paragraphs, lists, and blockquotes, and
# should not merge them into a single chunk if it compromises semantic boundaries.

MARKDOWN_WITH_DIFFERENT_BLOCKS = """# Section with Mixed Content

This is the first paragraph. It is a distinct semantic unit.

- This is the first list item.
- This is the second list item.

This is the second paragraph, appearing after the list."""

@pytest.mark.skip(reason="This test is designed to fail with the current implementation.")
def test_does_not_merge_different_block_types():
    """
    This test is designed to FAIL with the current implementation.
    It checks that the splitter does not merge adjacent, distinct block types
    (paragraph and list) even when they would fit within the chunk size.
    """
    # We use a large chunk_size to ensure that any splitting is due to
    # structural boundaries, not size constraints.
    splitter = MarkdownSplitter(chunk_size=1024, chunk_overlap=0)
    chunks = splitter.split_text(MARKDOWN_WITH_DIFFERENT_BLOCKS)

    # Expected behavior (with the NEW logic):
    # Chunk 1: The H1 and the first paragraph.
    # Chunk 2: The list.
    # Chunk 3: The second paragraph.
    #
    # Current (failing) behavior:
    # The entire text is returned as a single chunk because it all fits
    # within the chunk_size and there are no H2-level boundaries.

    # Let's adjust the expectation to match the current implementation's logic.
    # It creates a block for the header, and a block for everything else.
    # So we expect 2 chunks. The test will be to make it 3.

    # The real expected output from the *current* implementation is likely 2 chunks:
    # 1. The header
    # 2. Everything else
    # Let's assert for 3 to make it fail.

    assert len(chunks) == 3, "Expected to split content into 3 chunks based on block type"

    # The following assertions will fail violently, which is the point.
    assert "This is the first paragraph" in chunks[0].content
    assert "list item" not in chunks[0].content

    assert "list item" in chunks[1].content
    assert "paragraph" not in chunks[1].content

    assert "This is the second paragraph" in chunks[2].content
