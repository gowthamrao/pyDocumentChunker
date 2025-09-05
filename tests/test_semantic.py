import numpy as np
import pytest

from text_segmentation.strategies.semantic import SemanticSplitter

# A mock embedding function for predictable results
def mock_embedding_function(texts):
    """
    Creates mock embeddings.
    - Sentences with "high" in them get a vector like [1, 0, 0, ...].
    - Sentences with "low" in them get a vector like [0, 1, 0, ...].
    - This creates predictable high/low similarity scores.
    """
    embeddings = []
    for text in texts:
        if "high" in text.lower():
            vec = np.array([1.0] + [0.0] * 9)
        elif "low" in text.lower():
            vec = np.array([0.0] + [1.0] * 1 + [0.0] * 8)
        else: # Neutral
            vec = np.array([0.0] * 2 + [1.0] * 1 + [0.0] * 7)
        embeddings.append(vec)
    return np.array(embeddings)


# Text designed to have clear semantic breaks
SAMPLE_TEXT = (
    "High similarity sentence one. High similarity sentence two. " # Group 1
    "Low similarity sentence one. Low similarity sentence two. "  # Group 2
    "High similarity sentence three. High similarity sentence four. " # Group 3
    "This is a short runt." # Runt group
)


def test_semantic_splitter_merging_and_overlap():
    """
    Tests that semantic groups are correctly merged into chunks with overlap.
    """
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="absolute",
        breakpoint_threshold=0.5, # Split when similarity is less than 0.5
        chunk_size=100,
        chunk_overlap=30,
    )

    chunks = splitter.split_text(SAMPLE_TEXT)

    # Expected breaks:
    # 1. Between "High... two." and "Low... one." (sim ~ 0)
    # 2. Between "Low... two." and "High... three." (sim ~ 0)
    # 3. Between "High... four." and "This is a short runt." (sim ~ 0)
    # This creates 4 "base units" to be merged.

    # With chunk_size=100, the first two groups should be merged.
    # "High... one. High... two. Low... one. Low... two. " is ~120 chars, so it will be one chunk.
    # The next chunk will start with an overlap from the first one.

    assert len(chunks) > 0

    chunk1 = chunks[0]
    chunk2 = chunks[1]

    # Check content
    assert "High similarity sentence one" in chunk1.content
    assert "Low similarity sentence two" in chunk1.content
    assert "High similarity sentence three" in chunk2.content

    # Check for overlap
    assert chunk1.overlap_content_next is not None
    assert chunk2.overlap_content_previous is not None
    assert chunk1.overlap_content_next == chunk2.overlap_content_previous
    assert chunk1.content.endswith(chunk1.overlap_content_next)
    assert len(chunk1.overlap_content_next) > 25 # Should be around 30

def test_semantic_splitter_runt_handling():
    """
    Tests that a small final semantic group is merged with the previous chunk.
    """
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="absolute",
        breakpoint_threshold=0.5,
        chunk_size=160, # Large enough to hold the first 3 groups
        chunk_overlap=20,
    )

    chunks = splitter.split_text(SAMPLE_TEXT)

    # With a large chunk size, the first 3 semantic groups will form one chunk.
    # The last group ("This is a short runt.") is very small.
    # The merge_chunks utility should merge this runt into the previous chunk.

    assert len(chunks) == 1

    final_chunk = chunks[0]
    assert final_chunk.content.endswith("This is a short runt.")

def test_semantic_splitter_std_dev_method():
    """Tests the standard deviation breakpoint method."""
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="std_dev",
        breakpoint_threshold=1.0, # Split when similarity is 1 std dev below the mean
    )
    chunks = splitter.split_text(SAMPLE_TEXT)

    # The similarities will be [~1, ~0, ~1, ~0, ~1, ~0].
    # Mean is ~0.5, std dev is ~0.5.
    # Threshold is ~0.0. All ~0 similarities will be breakpoints.
    # This should result in chunks being created.
    assert len(chunks) > 1

def test_semantic_splitter_percentile_method():
    """Tests the percentile breakpoint method."""
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="percentile",
        breakpoint_threshold=50, # Split on anything below the 50th percentile
    )
    chunks = splitter.split_text(SAMPLE_TEXT)

    # The similarities will be [~1, ~0, ~1, ~0, ~1, ~0].
    # The 50th percentile is ~0.5. The ~0 similarities will be breakpoints.
    assert len(chunks) > 1
