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
        assert chunks[0].content in expected_text

    def test_unicode_normalization_nfkc(self):
        text = "ﬁne dining"
        splitter = FixedSizeSplitter(
            chunk_size=20,
            chunk_overlap=0,
            unicode_normalize="NFKC"
        )
        chunks = splitter.split_text(text)
        assert chunks[0].content == "fine dining"

    def test_unicode_normalization_nfc(self):
        text = "cafe\u0301" # café
        splitter = FixedSizeSplitter(
            chunk_size=10,
            chunk_overlap=0,
            unicode_normalize="NFC"
        )
        chunks = splitter.split_text(text)
        assert chunks[0].content == "café"

    def test_minimum_chunk_size_discard(self):
        text = "This is a sentence. Runt. This is another."
        splitter = FixedSizeSplitter(
            chunk_size=20,
            chunk_overlap=0,
            minimum_chunk_size=10,
            min_chunk_merge_strategy="discard"
        )
        # Produces chunks of len 20, 20, 18. Then enforces min size.
        # Chunks before runt handling:
        # 1. "This is a sentence. " (20)
        # 2. "Runt. This is anothe" (20)
        # 3. "r." (2) -> Runt, gets discarded.
        chunks = splitter.split_text(text)
        assert len(chunks) == 2
        assert chunks[0].content == "This is a sentence. "
        assert chunks[1].content == "Runt. This is anothe"

    def test_runt_handling_merging_logic(self):
        """
        Tests the _enforce_minimum_chunk_size logic directly for all merge strategies.
        This provides a more reliable unit test than checking splitter output.
        """
        c_large = Chunk(content="1234567890", start_index=0, end_index=10, sequence_number=0)
        c_runt = Chunk(content="12345", start_index=11, end_index=16, sequence_number=1)
        c_large2 = Chunk(content="abcdefghij", start_index=17, end_index=27, sequence_number=2)

        # --- Test merge_with_previous ---
        splitter_prev = FixedSizeSplitter(
            chunk_size=100,
            chunk_overlap=0, # Fix: Added to prevent ValueError
            minimum_chunk_size=6,
            min_chunk_merge_strategy="merge_with_previous"
        )

        # Runt in the middle is merged with previous
        chunks_prev = splitter_prev._enforce_minimum_chunk_size([c_large, c_runt, c_large2])
        assert len(chunks_prev) == 2
        assert chunks_prev[0].content == "123456789012345"
        assert chunks_prev[0].end_index == 16
        assert chunks_prev[1].content == "abcdefghij"
        assert chunks_prev[1].sequence_number == 1

        # Runt at the start is NOT merged
        chunks_prev = splitter_prev._enforce_minimum_chunk_size([c_runt, c_large])
        assert len(chunks_prev) == 2

        # --- Test merge_with_next ---
        splitter_next = FixedSizeSplitter(
            chunk_size=100,
            chunk_overlap=0, # Fix: Added to prevent ValueError
            minimum_chunk_size=6,
            min_chunk_merge_strategy="merge_with_next"
        )

        # Runt in the middle is merged with next
        chunks_next = splitter_next._enforce_minimum_chunk_size([c_large, c_runt, c_large2])
        assert len(chunks_next) == 2
        assert chunks_next[0].content == "1234567890"
        assert chunks_next[1].content == "12345abcdefghij"
        assert chunks_next[1].start_index == 11
        assert chunks_next[1].end_index == 27
        assert chunks_next[1].sequence_number == 1

        # Runt at the end is NOT merged
        chunks_next = splitter_next._enforce_minimum_chunk_size([c_large, c_runt])
        assert len(chunks_next) == 2

    def test_invalid_unicode_form_raises_error(self):
        with pytest.raises(ValueError, match="Invalid unicode_normalize form"):
            FixedSizeSplitter(unicode_normalize="INVALID", chunk_overlap=0)

    def test_invalid_merge_strategy_raises_error(self):
        with pytest.raises(ValueError, match="Invalid min_chunk_merge_strategy"):
            FixedSizeSplitter(min_chunk_merge_strategy="INVALID", chunk_overlap=0)

    def test_min_chunk_size_too_large_raises_error(self):
        with pytest.raises(ValueError, match="minimum_chunk_size .* must be smaller than"):
            FixedSizeSplitter(chunk_size=100, minimum_chunk_size=100, chunk_overlap=0)

    def test_preprocess_null_bytes(self):
        text = "This text contains\x00 a null byte."
        splitter = FixedSizeSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert "\x00" not in chunks[0].content
        assert chunks[0].content == "This text contains a null byte."
