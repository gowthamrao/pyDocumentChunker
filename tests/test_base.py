import pytest
from text_segmentation.strategies.fixed_size import FixedSizeSplitter
from text_segmentation.core import Chunk

import pytest
from text_segmentation.strategies.fixed_size import FixedSizeSplitter
from text_segmentation.core import Chunk

class TestBaseFunctionality:
    def setup_method(self):
        self.length_function = len

    def test_whitespace_normalization(self):
        text = "This   is a    test.\n\nIt has   extra whitespace."
        splitter = FixedSizeSplitter(
            chunk_size=20,
            chunk_overlap=5,
            normalize_whitespace=True
        )
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "  " not in chunk.content
            assert "\n" not in chunk.content

        expected_text = "This is a test. It has extra whitespace."
        reconstructed_text = "".join(c.content for c in chunks)
        # The reconstructed text might not be identical due to chunking,
        # but it should be a substring of the normalized text.
        assert chunks[0].content in expected_text

    def test_unicode_normalization_nfkc(self):
        # The 'ﬁ' ligature character, which NFKC normalization splits into 'fi'
        text = "ﬁne dining"
        splitter = FixedSizeSplitter(
            chunk_size=20, # Increased size to avoid truncation
            chunk_overlap=0,
            unicode_normalize="NFKC"
        )
        chunks = splitter.split_text(text)

        assert chunks[0].content == "fine dining"

    def test_unicode_normalization_nfc(self):
        # A sequence of 'e' + combining acute accent, which NFC combines
        text = "cafe\u0301" # café
        splitter = FixedSizeSplitter(
            chunk_size=10,
            chunk_overlap=0,
            unicode_normalize="NFC"
        )
        chunks = splitter.split_text(text)

        assert chunks[0].content == "café" # The single character version

    def test_minimum_chunk_size_discard(self):
        text = "This is a sentence. This is another. And a short one."
        splitter = FixedSizeSplitter(
            chunk_size=22,
            chunk_overlap=0,
            minimum_chunk_size=10,
            min_chunk_merge_strategy="discard"
        )
        # Without discard, this would produce:
        # 1. "This is a sentence. Th"
        # 2. "is is another. And a s"
        # 3. "hort one." (length 9) -> Should be discarded
        chunks = splitter.split_text(text)

        assert len(chunks) == 2
        assert chunks[0].content == "This is a sentence. Th"
        assert chunks[1].content == "is is another. And a s"

    def test_minimum_chunk_size_merge_with_previous(self):
        text = "This is a sentence. This is another. And a short one."
        splitter = FixedSizeSplitter(
            chunk_size=22,
            chunk_overlap=0,
            minimum_chunk_size=10,
            min_chunk_merge_strategy="merge_with_previous"
        )
        # Without merge, this would produce 3 chunks. With merge, the 3rd is merged into the 2nd.
        chunks = splitter.split_text(text)

        assert len(chunks) == 2
        assert chunks[0].content == "This is a sentence. Th"
        assert chunks[1].content == "is is another. And a short one."

    def test_minimum_chunk_size_merge_logic(self):
        # Test the _enforce_minimum_chunk_size logic directly

        # Chunks to be used in tests
        c_small_1 = Chunk(content="12345", start_index=0, end_index=5, sequence_number=0)
        c_large = Chunk(content="123456789", start_index=6, end_index=15, sequence_number=1)
        c_small_2 = Chunk(content="123", start_index=16, end_index=19, sequence_number=2)

        splitter = FixedSizeSplitter(
            chunk_size=100, # large enough to not interfere
            chunk_overlap=0,
            minimum_chunk_size=6,
            min_chunk_merge_strategy="merge_with_previous"
        )

        # Case 1: A small chunk at the beginning is NOT merged with the next one,
        # because it has no preceding chunk.
        chunks = splitter._enforce_minimum_chunk_size([c_small_1, c_large])
        assert len(chunks) == 2
        assert chunks[0].content == "12345"

        # Case 2: A small chunk in the middle IS merged with the previous one.
        chunks = splitter._enforce_minimum_chunk_size([c_large, c_small_1, c_large])
        assert len(chunks) == 2
        assert chunks[0].content == "12345678912345"
        assert chunks[1].content == "123456789"

        # Case 3: A small chunk at the end IS merged with the previous one.
        chunks = splitter._enforce_minimum_chunk_size([c_large, c_small_1])
        assert len(chunks) == 1
        assert chunks[0].content == "12345678912345"

        # Case 4: Multiple small chunks are merged sequentially.
        chunks = splitter._enforce_minimum_chunk_size([c_large, c_small_1, c_small_2])
        assert len(chunks) == 1
        # The second small chunk (c_small_2) is merged into the result of the first merge.
        assert chunks[0].content == "12345678912345123"


    def test_invalid_unicode_form_raises_error(self):
        with pytest.raises(ValueError, match="Invalid unicode_normalize form"):
            FixedSizeSplitter(unicode_normalize="INVALID")

    def test_invalid_merge_strategy_raises_error(self):
        with pytest.raises(ValueError, match="Invalid min_chunk_merge_strategy"):
            FixedSizeSplitter(min_chunk_merge_strategy="INVALID")

    def test_min_chunk_size_too_large_raises_error(self):
        with pytest.raises(ValueError, match="minimum_chunk_size .* must be smaller than"):
            FixedSizeSplitter(chunk_size=100, minimum_chunk_size=100, chunk_overlap=0)
