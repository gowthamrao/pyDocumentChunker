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
    splitter = SentenceSplitter(chunk_size=80, overlap_sentences=0)
    chunks = splitter.split_text(TEXT)

    assert len(chunks) == 2
    assert chunks[0].content == "This is the first sentence. This is the second sentence. A third one follows."
    assert chunks[0].start_index == 0
    assert chunks[1].content == "And a fourth. Finally, the fifth."
    assert chunks[1].start_index == 78

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available")
def test_sentence_overlap():
    """Tests the sentence-based overlap functionality."""
    splitter = SentenceSplitter(chunk_size=80, overlap_sentences=1)
    chunks = splitter.split_text(TEXT)

    assert len(chunks) == 2
    assert chunks[0].content == "This is the first sentence. This is the second sentence. A third one follows."
    assert chunks[1].content == "A third one follows. And a fourth. Finally, the fifth."
    assert chunks[0].overlap_content_next is not None


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

def test_dependency_import_error(mocker):
    """
    Tests that an ImportError is raised if NLTK is not installed.
    We use the mocker fixture to simulate its absence.
    """
    import sys
    import importlib

    # Simulate that the `nltk` package is not available
    mocker.patch.dict(sys.modules, {"nltk": None})

    # The module that imports `nltk` must be reloaded for the patch to take effect
    import text_segmentation.strategies.sentence
    importlib.reload(text_segmentation.strategies.sentence)

    from text_segmentation.strategies.sentence import SentenceSplitter
    with pytest.raises(ImportError, match="NLTK is not installed"):
        SentenceSplitter()

    # It's good practice to restore the original state after the test,
    # which can be done by reloading the module again.
    importlib.reload(text_segmentation.strategies.sentence)
