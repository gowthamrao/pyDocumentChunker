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

    def test_runt_handling_and_metadata_update(self):
        """
        Tests that when a runt is merged, the overlap metadata of the affected
        chunks is correctly recalculated. This is a direct test of the bugfix.
        """
        original_text = "aaaaabbbbbccccc"

        # Chunks before merge:
        # c1 and c_runt will be merged.
        # The new merged chunk should have a recalculated overlap with c2.
        c1 = Chunk(content="aaaaa", start_index=0, end_index=5, sequence_number=0)
        c_runt = Chunk(content="bb", start_index=5, end_index=7, sequence_number=1)
        c2 = Chunk(content="bccccc", start_index=6, end_index=12, sequence_number=2)

        # Manually set the "before" state. The overlap between c1 and c2 is non-existent,
        # and the overlap between the runt and c2 is "b".
        c1.overlap_content_next = None
        c_runt.overlap_content_next = "b"
        c2.overlap_content_previous = "b"

        splitter = FixedSizeSplitter(
            chunk_size=100,
            chunk_overlap=10,
            minimum_chunk_size=3,
            min_chunk_merge_strategy="merge_with_previous"
        )

        # --- Action: Enforce minimum chunk size ---
        # c_runt (len 2) is smaller than min_chunk_size (3), so it merges with c1.
        initial_chunks = [c1, c_runt, c2]
        merged_chunks = splitter._enforce_minimum_chunk_size(initial_chunks, original_text)

        # --- Assertions ---
        assert len(merged_chunks) == 2, "Runt chunk should have been merged"

        # The new merged chunk (old c1 + c_runt)
        merged_c1 = merged_chunks[0]
        # The next chunk (old c2)
        next_c = merged_chunks[1]

        # Check content and indices of the merged chunk
        assert merged_c1.content == "aaaaabb"
        assert merged_c1.start_index == 0
        assert merged_c1.end_index == 7

        # *** Key assertion: Check that overlap metadata was recalculated ***
        # The overlap between the new merged chunk (ending at index 7) and the
        # next chunk (starting at index 6) should be original_text[6:7] = "b".
        assert merged_c1.overlap_content_next == "b"
        assert next_c.overlap_content_previous == "b"

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

    def test_preprocess_removes_bom(self):
        """
        Tests that the UTF-8 Byte-Order Mark (BOM) is stripped from the
        beginning of the text during preprocessing, as per FRD R-2.2.2.
        """
        text_with_bom = "\ufeffThis text starts with a BOM."
        expected_text = "This text starts with a BOM."

        # Instantiate any concrete splitter to get access to the _preprocess method
        splitter = FixedSizeSplitter(chunk_size=100, chunk_overlap=0)

        # Directly test the internal _preprocess method
        processed_text = splitter._preprocess(text_with_bom)

        assert processed_text == expected_text
        assert not processed_text.startswith("\ufeff")

        # Also test it as part of the full split_text pipeline
        chunks = splitter.split_text(text_with_bom)
        assert chunks[0].content == expected_text

    def test_runt_handling_at_boundaries(self):
        """
        Tests that runts at the beginning or end of a list are handled correctly,
        even if the primary merge direction is not possible.
        """
        # --- Test 'merge_with_next' for a runt at the END ---
        c_normal_1 = Chunk(content="This is a normal chunk.", start_index=0, end_index=23, sequence_number=0)
        c_runt_1 = Chunk(content=" Runt.", start_index=23, end_index=29, sequence_number=1)
        text_1 = c_normal_1.content + c_runt_1.content

        splitter_next = FixedSizeSplitter(
            chunk_size=100,
            chunk_overlap=0,
            minimum_chunk_size=10, # Runt is len 6
            min_chunk_merge_strategy="merge_with_next"
        )
        chunks_next = splitter_next._enforce_minimum_chunk_size([c_normal_1, c_runt_1], text_1)

        assert len(chunks_next) == 1, "Runt at the end should have merged with previous"
        assert chunks_next[0].content == "This is a normal chunk. Runt."
        assert chunks_next[0].start_index == 0
        assert chunks_next[0].end_index == 29

        # --- Test 'merge_with_previous' for a runt at the BEGINNING ---
        c_runt_2 = Chunk(content="Runt. ", start_index=0, end_index=6, sequence_number=0)
        c_normal_2 = Chunk(content="This is a normal chunk.", start_index=6, end_index=29, sequence_number=1)
        text_2 = c_runt_2.content + c_normal_2.content

        splitter_prev = FixedSizeSplitter(
            chunk_size=100,
            chunk_overlap=0,
            minimum_chunk_size=10, # Runt is len 6
            min_chunk_merge_strategy="merge_with_previous"
        )
        chunks_prev = splitter_prev._enforce_minimum_chunk_size([c_runt_2, c_normal_2], text_2)

        assert len(chunks_prev) == 1, "Runt at the beginning should have merged with next"
        assert chunks_prev[0].content == "Runt. This is a normal chunk."
        assert chunks_prev[0].start_index == 0
        assert chunks_prev[0].end_index == 29
