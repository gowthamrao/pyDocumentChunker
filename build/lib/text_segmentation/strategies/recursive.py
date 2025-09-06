import re
from typing import Any, List, Optional, Tuple

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.utils import _populate_overlap_metadata


class RecursiveCharacterSplitter(TextSplitter):
    """
    Splits text recursively based on a prioritized list of character separators.

    This strategy is one of the most common and versatile for text chunking. It
    works by attempting to split the text hierarchically using a list of
    separators. If a split results in a segment that is still too large, the
    strategy recursively attempts to split that segment using the next separator
    in the list.

    This implementation follows a two-stage process to ensure that the `start_index`
    metadata is accurately preserved:
    1.  **Splitting:** The text is recursively broken down into small pieces using the
        provided separators. The output of this stage is a list of text fragments,
        each with its start index relative to the original document.
    2.  **Merging:** The small, indexed fragments are then merged back together into
        chunks that respect the `chunk_size` and `chunk_overlap` parameters.
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initializes the RecursiveCharacterSplitter.

        Args:
            separators: A prioritized list of strings or regex patterns to split on.
                Defaults to `["\\n\\n", "\\n", ". ", " ", ""]`.
            keep_separator: If True, the separator is kept as part of the preceding
                chunk. This is generally recommended to preserve context.
            *args, **kwargs: Additional arguments passed to the base `TextSplitter`.
        """
        super().__init__(*args, **kwargs)
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self._keep_separator = keep_separator

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """
        Splits text by a separator, robustly handling regex patterns.

        This method uses a capturing group in `re.split` to ensure that the
        separator is kept, and it works for both fixed-width and variable-width
        regex patterns.
        """
        if not separator:
            return list(text)

        # Use re.split with a capturing group `()` to keep the separator.
        # e.g., re.split('(a)', 'b-a-c') -> ['b-', 'a', '-c']
        # This works for variable-width patterns, unlike the lookbehind approach.
        splits = re.split(f"({separator})", text)

        if not self._keep_separator:
            # If we don't want to keep the separator, we can just return the
            # parts of the split that are not the separator itself. The separator
            # will be every other element in the list starting from index 1.
            return [s for s in splits[::2] if s]

        # Merge the separator with the preceding part of the text
        # ['b-', 'a', '-c'] -> ['b-a', '-c']
        merged_splits = []
        for i in range(0, len(splits), 2):
            # The fragment is at the even index
            fragment = splits[i]
            # The separator is at the odd index, if it exists
            separator_part = splits[i + 1] if i + 1 < len(splits) else ""
            if fragment or separator_part:
                merged_splits.append(fragment + separator_part)

        return [s for s in merged_splits if s]

    def _recursive_split(
        self, text: str, separators: List[str], start_index: int
    ) -> List[Tuple[str, int]]:
        """Recursively splits text and returns fragments with their start indices."""
        if not text:
            return []

        text_length = self.length_function(text)
        if text_length <= self.chunk_size:
            return [(text, start_index)]

        if not separators:
            # Base case: no more separators, but text is still too long.
            # Force split by character, respecting overlap.
            fragments = []
            step = self.chunk_size - self.chunk_overlap
            for i in range(0, len(text), step):
                chunk_text = text[i : i + self.chunk_size]
                fragments.append((chunk_text, start_index + i))
            return fragments

        # Try to split with the current separator
        current_separator = separators[0]
        remaining_separators = separators[1:]

        splits = [s for s in self._split_text_with_separator(text, current_separator) if s]

        # If the separator didn't split the text, try the next one
        if len(splits) <= 1:
            return self._recursive_split(text, remaining_separators, start_index)

        # Process the splits
        fragments = []
        current_offset = 0
        for part in splits:
            part_start_index = start_index + text.find(part, current_offset)
            fragments.extend(
                self._recursive_split(part, remaining_separators, part_start_index)
            )
            current_offset = part_start_index - start_index + len(part)

        return fragments

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Splits the input text using the recursive and merging strategy.

        Args:
            text: The text to be split.
            source_document_id: Optional identifier for the source document.

        Returns:
            A list of `Chunk` objects.
        """
        text = self._preprocess(text)
        if not text:
            return []

        # Stage 1: Recursively split text into indexed fragments
        fragments = self._recursive_split(text, self._separators, 0)

        # Stage 2: Merge fragments into chunks, with correct length calculation.
        chunks: List[Chunk] = []
        current_chunk_fragments: List[Tuple[str, int]] = []
        sequence_number = 0

        for fragment_text, fragment_start_index in fragments:
            # Check if adding the next fragment would exceed the chunk size.
            # This is done by joining the content and measuring, which is correct
            # for non-additive length functions like tokenizers.
            potential_content = "".join(f[0] for f in current_chunk_fragments) + fragment_text
            if self.length_function(potential_content) > self.chunk_size and current_chunk_fragments:
                # Finalize the current chunk
                content = "".join(f[0] for f in current_chunk_fragments)
                start_idx = current_chunk_fragments[0][1]
                chunk = Chunk(
                    content=content,
                    start_index=start_idx,
                    end_index=start_idx + len(content),
                    sequence_number=sequence_number,
                    source_document_id=source_document_id,
                    chunking_strategy_used="recursive_character",
                )
                chunks.append(chunk)
                sequence_number += 1

                # Start a new chunk, handling overlap.
                # We slide a window backwards from the end of the last chunk's fragments.
                overlap_fragments_start_idx = len(current_chunk_fragments) - 1
                overlap_fragments: List[Tuple[str, int]] = []

                while overlap_fragments_start_idx >= 0:
                    # Prepend the fragment to the beginning of our overlap list
                    current_fragment = current_chunk_fragments[overlap_fragments_start_idx]
                    overlap_fragments.insert(0, current_fragment)

                    # Measure the length of the potential overlap string
                    overlap_content = "".join(f[0] for f in overlap_fragments)
                    if self.length_function(overlap_content) > self.chunk_overlap:
                        # The overlap is now too big, so we discard the last fragment we added
                        # and break the loop.
                        overlap_fragments.pop(0)
                        break

                    overlap_fragments_start_idx -= 1

                # The new chunk starts with the overlap fragments
                current_chunk_fragments = overlap_fragments

            # Add the new fragment to the current chunk
            current_chunk_fragments.append((fragment_text, fragment_start_index))

        # Add the last remaining chunk
        if current_chunk_fragments:
            content = "".join(f[0] for f in current_chunk_fragments)
            start_idx = current_chunk_fragments[0][1]
            chunk = Chunk(
                content=content,
                start_index=start_idx,
                end_index=start_idx + len(content),
                sequence_number=sequence_number,
                source_document_id=source_document_id,
                chunking_strategy_used="recursive_character",
            )
            chunks.append(chunk)

        # Post-process to add overlap metadata using the shared utility.
        _populate_overlap_metadata(chunks, text)

        return self._enforce_minimum_chunk_size(chunks, text)
