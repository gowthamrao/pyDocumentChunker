import pytest
from text_segmentation.strategies.fixed_size import FixedSizeSplitter


def test_fixed_size_basic_splitting():
    """Tests basic functionality of the FixedSizeSplitter."""
    splitter = FixedSizeSplitter(chunk_size=10, chunk_overlap=2)
    text = "This is a simple test text."
    chunks = splitter.split_text(text)
    step = 10 - 2

    assert len(chunks) == 4
    assert chunks[0].content == text[0:10]
    assert chunks[1].content == text[step : step + 10]
    assert chunks[2].content == text[step * 2 : step * 2 + 10]
    assert chunks[3].content == text[step * 3 :]


def test_fixed_size_no_overlap():
    """Tests splitter with zero overlap."""
    splitter = FixedSizeSplitter(chunk_size=10, chunk_overlap=0)
    text = "This is a test."
    chunks = splitter.split_text(text)

    assert len(chunks) == 2
    assert chunks[0].content == "This is a "
    assert chunks[1].content == "test."


def test_fixed_size_edge_cases():
    """Tests edge cases like empty text and text smaller than chunk size."""
    splitter = FixedSizeSplitter(chunk_size=100, chunk_overlap=10)

    assert splitter.split_text("") == []

    chunks = splitter.split_text("small")
    assert len(chunks) == 1
    assert chunks[0].content == "small"


def test_invalid_configuration():
    """Tests that the splitter raises an error for invalid configuration."""
    with pytest.raises(ValueError, match="must be smaller than"):
        FixedSizeSplitter(chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError, match="must be smaller than"):
        FixedSizeSplitter(chunk_size=10, chunk_overlap=15)

def test_length_function_warning():
    """Tests that a warning is issued if a custom length_function is provided."""
    with pytest.warns(UserWarning, match="operates on character counts"):
        FixedSizeSplitter(length_function=lambda x: len(x) // 2)
