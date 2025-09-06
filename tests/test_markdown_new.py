import pytest
from text_segmentation.strategies.structure.markdown import MarkdownSplitter

# A complex markdown document for testing
COMPLEX_MD = """\
# Document Title

This is the first paragraph. It serves as an introduction.

## Section 1: Lists

Here is a list of items:
- Item 1
- Item 2
  - Sub-item 2.1
  - Sub-item 2.2

And an ordered list:
1. First item
2. Second item

## Section 2: Code and Quotes

Here is a block of Python code:
```python
def hello_world():
    print("Hello, world!")
```

> This is a blockquote.
> It can span multiple lines.

### Subsection 2.1

This is a nested section.
"""

def test_initialization():
    """Test that the splitter initializes correctly."""
    splitter = MarkdownSplitter()
    assert splitter is not None
    assert splitter.md_parser is not None

def test_simple_paragraph_splitting():
    """Test splitting a simple text with only paragraphs."""
    text = "First paragraph.\\n\\nSecond paragraph.".replace("\\n", "\n")
    splitter = MarkdownSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_text(text)

    assert len(chunks) == 2
    assert chunks[0].content == "First paragraph."
    assert chunks[1].content == "Second paragraph."
    assert chunks[0].start_index == 0
    assert chunks[1].start_index == 18

def test_header_splitting_and_context():
    """Test that text is split by headers and context is captured."""
    text = "# Title\\n\\nParagraph 1.\\n\\n## Subtitle\\n\\nParagraph 2.".replace("\\n", "\n")
    splitter = MarkdownSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_text(text)

    assert len(chunks) == 4

    # Test Header 1
    assert chunks[0].content == "# Title"
    assert chunks[0].hierarchical_context == {"H1": "Title"}

    # Test Paragraph 1
    assert chunks[1].content == "Paragraph 1."
    assert chunks[1].hierarchical_context == {"H1": "Title"}

    # Test Header 2
    assert chunks[2].content == "## Subtitle"
    assert chunks[2].hierarchical_context == {"H1": "Title", "H2": "Subtitle"}

    # Test Paragraph 2
    assert chunks[3].content == "Paragraph 2."
    assert chunks[3].hierarchical_context == {"H1": "Title", "H2": "Subtitle"}

def test_list_splitting():
    """Test that lists are treated as single blocks."""
    text = "Intro paragraph.\\n\\n- Item 1\\n- Item 2\\n- Item 3\\n\\nOutro.".replace("\\n", "\n")
    splitter = MarkdownSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_text(text)

    assert len(chunks) == 3
    assert chunks[0].content == "Intro paragraph."
    assert "Item 1" in chunks[1].content
    assert "Item 3" in chunks[1].content
    assert chunks[2].content == "Outro."

def test_code_block_splitting():
    """Test that code blocks are treated as single blocks."""
    text = "P1\\n\\n```python\\ndef test():\\n    pass\\n```\\n\\nP2".replace("\\n", "\n")
    splitter = MarkdownSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_text(text)

    assert len(chunks) == 3
    assert chunks[0].content == "P1"
    assert chunks[1].content.startswith("```python")
    assert chunks[2].content == "P2"

def test_oversized_block_fallback():
    """Test that a block larger than chunk_size is split by the fallback."""
    long_paragraph = "This is a very long paragraph that is designed to be much larger than the tiny chunk size that we will set for this specific test case."
    text = f"# Title\\n\\n{long_paragraph}".replace("\\n", "\n")
    splitter = MarkdownSplitter(chunk_size=40, chunk_overlap=10)
    chunks = splitter.split_text(text)

    assert len(chunks) > 1 # Should be split into at least one header and one content chunk
    assert chunks[0].content == "# Title"

    # The second block (the long paragraph) should have been split by the fallback
    assert chunks[1].chunking_strategy_used == "markdown-fallback"
    assert len(chunks[1].content) < len(long_paragraph)
    assert long_paragraph.startswith(chunks[1].content)
    assert chunks[1].hierarchical_context == {"H1": "Title"}

def test_complex_document_structure_and_indices():
    """Test the full parsing of a complex document."""
    splitter = MarkdownSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(COMPLEX_MD)

    # Expected blocks: H1, P, H2, P, UL, P, OL, H2, P, Code, Quote, H3, P
    assert len(chunks) == 13

    # Check content and context of a few key blocks
    assert chunks[0].content == "# Document Title"
    assert chunks[0].hierarchical_context == {"H1": "Document Title"}

    # Check the paragraph before the first list
    assert chunks[3].content == "Here is a list of items:"
    assert chunks[3].hierarchical_context == {"H1": "Document Title", "H2": "Section 1: Lists"}

    # Check the bullet list
    assert chunks[4].content.startswith("- Item 1")
    assert chunks[4].hierarchical_context == {"H1": "Document Title", "H2": "Section 1: Lists"}

    # Check the code block
    assert chunks[9].content.startswith("```python")
    assert chunks[9].hierarchical_context == {"H1": "Document Title", "H2": "Section 2: Code and Quotes"}

    # Check the final paragraph
    assert chunks[12].content == "This is a nested section."
    assert chunks[12].hierarchical_context == {"H1": "Document Title", "H2": "Section 2: Code and Quotes", "H3": "Subsection 2.1"}

    # Check indices
    p1_content = "This is the first paragraph. It serves as an introduction."
    assert COMPLEX_MD[chunks[1].start_index:chunks[1].end_index].strip() == p1_content

    quote_content = "> This is a blockquote.\n> It can span multiple lines."
    assert COMPLEX_MD[chunks[10].start_index:chunks[10].end_index].strip() == quote_content
