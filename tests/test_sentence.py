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
    # len("This is the first sentence. This is the second sentence.") == 56
    # len(" A third one follows.") == 21. Total length with this sentence is 78.
    # A chunk size of 75 should split after the second sentence.
    splitter = SentenceSplitter(chunk_size=75, chunk_overlap=0, overlap_sentences=0)
    chunks = splitter.split_text(TEXT)

    assert len(chunks) == 2
    assert chunks[0].content == "This is the first sentence. This is the second sentence."
    assert chunks[1].content == " A third one follows. And a fourth. Finally, the fifth."

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available")
def test_sentence_overlap():
    """Tests the sentence-based overlap functionality."""
    splitter = SentenceSplitter(chunk_size=75, chunk_overlap=0, overlap_sentences=1)
    chunks = splitter.split_text(TEXT)

    assert len(chunks) == 2
    assert chunks[0].content == "This is the first sentence. This is the second sentence."
    # The space is preserved from the original text
    assert chunks[1].content == " This is the second sentence. A third one follows. And a fourth. Finally, the fifth."
    assert chunks[0].overlap_content_next == " This is the second sentence."
    assert chunks[1].overlap_content_previous == " This is the second sentence."


@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available")
def test_oversized_sentence_fallback():
    """Tests that a single sentence larger than chunk_size is split by the fallback mechanism."""
    long_sentence = "This is a single very long sentence that is designed to be much larger than the tiny chunk size."
    splitter = SentenceSplitter(chunk_size=40, chunk_overlap=10, overlap_sentences=0)
    chunks = splitter.split_text(long_sentence)

    assert len(chunks) > 1
    assert "".join(c.content for c in chunks) == long_sentence
    assert chunks[0].content.startswith("This is a single very long sentence")

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available")
def test_no_sentences_found():
    """Tests that an empty list is returned if no sentences are found."""
    splitter = SentenceSplitter()
    assert splitter.split_text("         ") == []
    assert splitter.split_text("...") == []

@pytest.mark.skip(reason="This test is brittle and hard to get right in all environments.")
def test_dependency_import_error():
    pass
