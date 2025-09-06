import pytest

try:
    from text_segmentation.strategies.structure.html import HTMLSplitter
    BS4_LXML_AVAILABLE = True
except ImportError:
    BS4_LXML_AVAILABLE = False

HTML_TEXT = """
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
    <h1>Main Title</h1>
    <p>This is the introduction.</p>
    <div class="section">
        <h2>Section 1</h2>
        <p>Here is the content for the first section.</p>
        <ul>
            <li>List item 1</li>
            <li>List item 2</li>
        </ul>
    </div>
    <div class="section">
        <h2>Section 2</h2>
        <p>This is the second section.</p>
    </div>
</body>
</html>
"""

@pytest.mark.skipif(not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed")
def test_html_basic_splitting():
    """Tests that HTML is split by block tags."""
    splitter = HTMLSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_text(HTML_TEXT)

    assert len(chunks) > 1
    # Check that chunks are based on block elements
    assert "Main Title" in chunks[0].content
    assert "This is the introduction" in chunks[0].content
    assert "Section 1" in chunks[0].content


@pytest.mark.skipif(not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed")
def test_html_hierarchical_context():
    """Tests that the hierarchical_context is correctly populated from parent headers."""
    splitter = HTMLSplitter(chunk_size=500, chunk_overlap=50) # Use a large chunk size
    chunks = splitter.split_text(HTML_TEXT)

    assert len(chunks) > 0

    # Find the chunk containing "List item 1"
    list_chunk = next(c for c in chunks if "List item 1" in c.content)

    # The direct parent header is H2, but it should also find the H1
    assert list_chunk.hierarchical_context.get("H1") == "Main Title"
    assert list_chunk.hierarchical_context.get("H2") == "Section 1"


@pytest.mark.skipif(not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed")
def test_html_stripping_tags():
    """Tests that irrelevant tags like <script> are stripped."""
    text_with_script = f'<body><p>Some content.</p><script>alert("you are hacked");</script></body>'
    splitter = HTMLSplitter()
    chunks = splitter.split_text(text_with_script)

    assert len(chunks) == 1
    assert "<script>" not in chunks[0].content
    assert "alert" not in chunks[0].content
    assert chunks[0].content.strip() == "Some content."

@pytest.mark.skipif(not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed")
def test_html_start_index_correctness():
    """Tests that the start_index metadata is accurate."""
    splitter = HTMLSplitter(chunk_size=50, chunk_overlap=5)
    chunks = splitter.split_text(HTML_TEXT)

    # This is a bit harder to test precisely due to whitespace handling in HTML parsing.
    # A good enough check is that the text content of the chunk can be found at or near
    # the recorded start_index.
    for chunk in chunks:
        # We search in a small window around the start_index
        window_start = max(0, chunk.start_index - 20)
        window_end = chunk.start_index + 20
        window = HTML_TEXT[window_start:window_end]

        # Check if the beginning of the chunk's content appears in the window
        assert chunk.content.split()[0] in window
