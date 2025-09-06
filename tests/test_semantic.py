import pytest
from typing import List

# Attempt to import numpy to determine if tests should be skipped.
try:
    import numpy as np
    from text_segmentation.strategies.semantic import SemanticSplitter
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# A mock embedding function for testing.
# Sentences about topic A get similar vectors.
# Sentences about topic B get similar vectors, but different from A.
def mock_embedding_function(texts: List[str]) -> "np.ndarray":
    embeddings = []
    for text in texts:
        if "Topic A" in text:
            embeddings.append([1.0, 0.0, 0.0])
        elif "Topic B" in text:
            embeddings.append([0.0, 1.0, 0.0])
        else: # transition or noise
            embeddings.append([0.0, 0.0, 1.0])
    return np.array(embeddings)

TEXT_FOR_SEMANTIC_SPLIT = (
    "This is a sentence about Topic A. Here is another sentence on Topic A. "
    "This sentence is a transition. "
    "Now we talk about Topic B. And here is more on Topic B."
)

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_semantic_splitting_identifies_breakpoints():
    """
    Tests that the SemanticSplitter correctly identifies multiple breakpoints
    based on embedding similarity drop-offs.
    """
    # Similarities: [1.0, 0.0, 0.0, 1.0]. 50th percentile is 0.5.
    # Breakpoints should be triggered for the two 0.0 similarity scores.
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="percentile",
        breakpoint_threshold=50
    )
    chunks = splitter.split_text(TEXT_FOR_SEMANTIC_SPLIT)

    # Expecting 3 chunks based on the two breakpoints found.
    # Chunk 1: Sentences about Topic A
    # Chunk 2: The transition sentence
    # Chunk 3: Sentences about Topic B
    assert len(chunks) == 3
    assert chunks[0].content == "This is a sentence about Topic A. Here is another sentence on Topic A."
    assert chunks[1].content == "This sentence is a transition."
    assert chunks[2].content == "Now we talk about Topic B. And here is more on Topic B."
    # No overlap should be generated when not using a fallback splitter with overlap
    assert chunks[0].overlap_content_next is None
    assert chunks[1].overlap_content_previous is None

@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_fallback_populates_overlap():
    """
    Tests that when a semantic chunk is too large and a fallback splitter with
    overlap is used, the final chunk list has overlap metadata correctly populated.
    This is the main test for the fix to the metadata population.
    """
    # This whole text is one semantic topic, so it won't be split by similarity.
    # However, it's larger than chunk_size, so it will be passed to the fallback.
    text = "This is a very long sentence about Topic A part one. This is a very long sentence about Topic A part two. This is a very long sentence about Topic A part three."

    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="percentile",
        breakpoint_threshold=95, # High threshold, should not split
        chunk_size=100,
        chunk_overlap=20 # This overlap is for the fallback splitter
    )

    chunks = splitter.split_text(text)

    # The fallback RecursiveCharacterSplitter should have been used.
    assert len(chunks) > 1
    # This is the key assertion: the overlap metadata should now be populated.
    assert chunks[0].overlap_content_next is not None
    assert chunks[1].overlap_content_previous is not None
    assert chunks[0].overlap_content_next == chunks[1].overlap_content_previous
    # Make the assertion less brittle, just check for a known part of the overlapping sentence.
    assert "Topic A part two" in chunks[1].content
    assert "about Topic A" in chunks[0].overlap_content_next

from unittest.mock import patch

def test_dependency_import_error():
    """
    Tests that an ImportError is raised if numpy is not installed.
    """
    import sys
    import importlib

    with patch.dict(sys.modules, {"numpy": None}):
        import text_segmentation.strategies.semantic
        importlib.reload(text_segmentation.strategies.semantic)

        from text_segmentation.strategies.semantic import SemanticSplitter
        with pytest.raises(ImportError, match="numpy is not installed"):
            SemanticSplitter(embedding_function=lambda x: [])
