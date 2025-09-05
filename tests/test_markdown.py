import pytest

try:
    from text_segmentation.strategies.structure.markdown import MarkdownSplitter
    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MARKDOWN_IT_AVAILABLE = False

MARKDOWN_TEXT = """
# Main Title

This is the introduction.

## Section 1

Here is the content for the first section. It's a paragraph.

- List item 1
- List item 2

Another paragraph in section 1.

## Section 2

This is the second section. It has a `code block`.

```python
def hello():
    print("Hello, World!")
```
"""

@pytest.mark.skipif(not MARKDOWN_IT_AVAILABLE, reason="markdown-it-py not installed")
def test_markdown_basic_splitting():
    """Tests that markdown is split by headers."""
    splitter = MarkdownSplitter(chunk_size=200, chunk_overlap=0)
    chunks = splitter.split_text(MARKDOWN_TEXT)

    assert len(chunks) == 3
    assert chunks[0].content.strip().startswith("Main Title")
    assert chunks[0].content.strip().endswith("This is the introduction.")
    assert chunks[1].content.strip().startswith("Section 1")
    assert chunks[2].content.strip().startswith("Section 2")

@pytest.mark.skipif(not MARKDOWN_IT_AVAILABLE, reason="markdown-it-py not installed")
def test_markdown_hierarchical_context():
    """Tests that the hierarchical_context metadata is correctly populated."""
    splitter = MarkdownSplitter(chunk_size=1000, chunk_overlap=0) # Large chunk size to get all content
    chunks = splitter.split_text(MARKDOWN_TEXT)

    assert len(chunks) > 0

    # Find the chunk containing "List item 1"
    list_chunk = next(c for c in chunks if "List item 1" in c.content)

    assert list_chunk.hierarchical_context.get("H1") == "Main Title"
    assert list_chunk.hierarchical_context.get("H2") == "Section 1"

    # Find the chunk with the code block
    code_chunk = next(c for c in chunks if 'print("Hello, World!")' in c.content)
    assert code_chunk.hierarchical_context.get("H1") == "Main Title"
    assert code_chunk.hierarchical_context.get("H2") == "Section 2"


@pytest.mark.skipif(not MARKDOWN_IT_AVAILABLE, reason="markdown-it-py not installed")
def test_markdown_fallback_splitting():
    """Tests fallback for a single large block."""
    long_section = "# Title\n" + "a " * 500
    splitter = MarkdownSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_text(long_section)

    assert len(chunks) > 1
    # Check that the context is preserved in fallback chunks
    assert all(c.hierarchical_context.get("H1") == "Title" for c in chunks)
    assert "".join(c.content for c in chunks).strip().startswith("Title\na a a")


@pytest.mark.skipif(not MARKDOWN_IT_AVAILABLE, reason="markdown-it-py not installed")
def test_start_index_correctness():
    """Tests that the start_index metadata is accurate."""
    splitter = MarkdownSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(MARKDOWN_TEXT)

    for chunk in chunks:
        # A simple check: the text at the chunk's start_index should match the chunk's content
        original_slice = MARKDOWN_TEXT[chunk.start_index : chunk.start_index + len(chunk.content)]
        assert original_slice == chunk.content
