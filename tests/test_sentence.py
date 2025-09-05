import pytest

# Attempt to import NLTK to determine if tests should be skipped.
try:
    import nltk
    from text_segmentation.strategies.sentence import SentenceSplitter
    # We also need to check for the 'punkt' model.
    try:
        nltk.data.find("tokenizers/punkt")
        NLTK_AVAILABLE = True
    except LookupError:
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False

TEXT = "This is the first sentence. This is the second sentence. A third one follows. And a fourth. Finally, the fifth."

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available")
def test_sentence_basic_splitting():
    """Tests basic sentence splitting and aggregation into chunks."""
    splitter = SentenceSplitter(chunk_size=80, chunk_overlap=0) # chunk_overlap is for fallback
    chunks = splitter.split_text(TEXT)

    assert len(chunks) == 2
    # First chunk should contain the first two sentences
    assert chunks[0].content == "This is the first sentence. This is the second sentence."
    assert chunks[0].start_index == 0
    # Second chunk should contain the rest
    assert chunks[1].content == "A third one follows. And a fourth. Finally, the fifth."
    assert chunks[1].start_index == 52

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available")
def test_sentence_overlap():
    """Tests the sentence-based overlap functionality."""
    splitter = SentenceSplitter(chunk_size=80, overlap_sentences=1)
    chunks = splitter.split_text(TEXT)

    assert len(chunks) == 2
    # First chunk is the same
    assert chunks[0].content == "This is the first sentence. This is the second sentence."
    # Second chunk should now start with the overlapping second sentence
    assert chunks[1].content == "This is the second sentence. A third one follows. And a fourth. Finally, the fifth."

    # Check that the overlap metadata is correctly populated
    assert chunks[1].overlap_content_previous == "This is the second sentence."
    assert chunks[0].overlap_content_next == "This is the second sentence."

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available")
def test_oversized_sentence_fallback():
    """Tests that a single sentence larger than chunk_size is split by the fallback mechanism."""
    long_sentence = "This is a single very long sentence that is designed to be much larger than the tiny chunk size we are going to set for this specific test case."
    splitter = SentenceSplitter(chunk_size=40, overlap_sentences=0)
    chunks = splitter.split_text(long_sentence)

    assert len(chunks) > 1
    assert "".join(c.content for c in chunks) == long_sentence
    assert chunks[0].content.startswith("This is a single very long sentence")
    assert chunks[1].start_index > 0

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available")
def test_no_sentences_found():
    """Tests that an empty list is returned if no sentences are found."""
    splitter = SentenceSplitter()
    assert splitter.split_text("         ") == []
    assert splitter.split_text("...") == []

def test_dependency_import_error():
    """
    Tests that an ImportError is raised if NLTK is not installed.
    This test can only be run in an environment where NLTK is not installed.
    We can simulate this by temporarily removing it from sys.modules.
    """
    import sys
    original_nltk = sys.modules.pop("nltk", None)

    # We must import the class inside the test function after unloading nltk
    from text_segmentation.strategies.sentence import SentenceSplitter

    with pytest.raises(ImportError, match="NLTK is not installed"):
        SentenceSplitter()

    # Restore nltk if it was originally there
    if original_nltk:
        sys.modules["nltk"] = original_nltk
