import pytest
from text_segmentation.strategies.structure.markdown import MarkdownSplitter

# FRD Requirement Being Tested:
# R-3.4.3: The strategy MUST prioritize splitting at higher-level structural boundaries.

HIERARCHICAL_MD = """
# Section 1 Title

This is the first paragraph of the first section. It is short and should be one chunk.

## Section 1.1 Title

This is a subsection. The content under Section 1, including this subsection, should ideally be one chunk if it fits.

# Section 2 Title

This is a paragraph directly under the H1. The text in this entire H1 section (including its H2 children) is intentionally made longer than the chunk size to force a split.

## Section 2.1 Title

This is the first subsection of Section 2. It's a single paragraph. We expect this to become a distinct chunk because the parent section is too large.

## Section 2.2 Title

This is the second subsection of Section 2. It contains a list to ensure different block types are handled.

- List item 1
- List item 2

This subsection should also be a distinct chunk.

# Section 3 Title

This is a very long section with no subheadings. It is designed to be much larger than the chunk size to test the fallback mechanism. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. More and more and more text to ensure it is very long indeed.
"""

def test_hierarchical_splitting_logic():
    """
    Tests that the splitter correctly splits a document based on header hierarchy.
    """
    # This chunk size is chosen to be larger than any single subsection,
    # but smaller than the entire "Section 2" or "Section 1".
    splitter = MarkdownSplitter(chunk_size=300, chunk_overlap=0)
    chunks = splitter.split_text(HIERARCHICAL_MD)

    # Expected behavior:
    # The splitter should see that Section 1 (>300 chars) is too big and split it.
    # Chunk 1: The H1 and first paragraph of Section 1.
    # Chunk 2: The H2 and subsection content of Section 1.1.
    # The splitter should see that Section 2 (>300 chars) is too big and split it by its H2s.
    # Chunk 3: The H1 and first paragraph of Section 2.
    # Chunk 4: The H2 and content of Section 2.1.
    # Chunk 5: The H2 and content of Section 2.2.
    # The splitter should see that Section 3 is too big and has no sub-headers, so it uses fallback.
    # Chunk 6 & 7: Section 3, split by the fallback mechanism.

    # For debugging:
    # for i, chunk in enumerate(chunks):
    #     print(f"--- Chunk {i+1} ({chunk.start_index}-{chunk.end_index}) ---")
    #     print(chunk.content)
    #     print(f"Context: {chunk.hierarchical_context}")
    #     print("-" * 20)

    assert len(chunks) == 7

    # Chunk 1: The paragraph before the first H2 in Section 1
    assert chunks[0].content.strip().startswith("# Section 1 Title")
    assert "This is the first paragraph" in chunks[0].content
    assert "Section 1.1 Title" not in chunks[0].content
    assert chunks[0].hierarchical_context == {"H1": "Section 1 Title"}

    # Chunk 2: Section 1.1
    assert chunks[1].content.strip().startswith("## Section 1.1 Title")
    assert chunks[1].hierarchical_context == {"H1": "Section 1 Title", "H2": "Section 1.1 Title"}

    # Chunk 3: The paragraph before the first H2 in Section 2
    assert chunks[2].content.strip().startswith("# Section 2 Title")
    assert "This is a paragraph directly under the H1" in chunks[2].content
    assert "Section 2.1 Title" not in chunks[2].content
    assert chunks[2].hierarchical_context == {"H1": "Section 2 Title"}

    # Chunk 4: Section 2.1
    assert chunks[3].content.strip().startswith("## Section 2.1 Title")
    assert "This is the first subsection of Section 2" in chunks[3].content
    assert chunks[3].hierarchical_context == {"H1": "Section 2 Title", "H2": "Section 2.1 Title"}

    # Chunk 5: Section 2.2
    assert chunks[4].content.strip().startswith("## Section 2.2 Title")
    assert "List item 1" in chunks[4].content
    assert chunks[4].hierarchical_context == {"H1": "Section 2 Title", "H2": "Section 2.2 Title"}

    # Chunk 6 & 7: Fallback split of Section 3
    assert chunks[5].content.strip().startswith("# Section 3 Title")
    assert "Lorem ipsum" in chunks[5].content
    assert "More and more" not in chunks[5].content # Check that it was split
    assert chunks[5].hierarchical_context == {"H1": "Section 3 Title"}
    assert "More and more" in chunks[6].content
    assert chunks[6].hierarchical_context == {"H1": "Section 3 Title"}


def test_proves_new_behavior_is_correct():
    """
    A test to prove the new behavior is hierarchical and not the old flat behavior.
    This test will fail with the old implementation.
    """
    splitter = MarkdownSplitter(chunk_size=300, chunk_overlap=0)
    chunks = splitter.split_text(HIERARCHICAL_MD)

    # The new implementation should produce fewer, more coherent chunks than the old one.
    # The old implementation would have created 10+ chunks. The new one creates 7.
    assert len(chunks) < 10

    # The old implementation would split after every single block element.
    # e.g., chunk 0 would be "# Section 1 Title" and nothing else.
    # The new implementation should have more content in the first chunk.
    assert chunks[0].content.strip() != "# Section 1 Title"
    assert "This is the first paragraph" in chunks[0].content
