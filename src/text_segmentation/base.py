import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Callable, List, Optional
import copy

from text_segmentation.core import Chunk


class TextSplitter(ABC):
    """
    Abstract Base Class for all text splitters in the package.

    This class establishes a common interface for all chunking strategies, ensuring
    modularity and extensibility. It also handles core configuration parameters
    common across most strategies.

    Attributes:
        chunk_size (int): The target maximum size of each chunk.
        chunk_overlap (int): The amount of overlap between consecutive chunks.
        length_function (Callable[[str], int]): Function to measure text length.
        normalize_whitespace (bool): If True, normalizes all whitespace.
        unicode_normalize (Optional[str]): The form for Unicode normalization
            (e.g., 'NFC', 'NFKC').
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        length_function: Optional[Callable[[str], int]] = None,
        normalize_whitespace: bool = False,
        unicode_normalize: Optional[str] = None,
        minimum_chunk_size: Optional[int] = None,
        min_chunk_merge_strategy: str = "merge_with_previous",
    ):
        """
        Initializes the TextSplitter.

        Args:
            chunk_size: The maximum size of a chunk.
            chunk_overlap: The overlap between consecutive chunks.
            length_function: The function to measure text length. Defaults to `len`.
            normalize_whitespace: If True, collapses consecutive whitespace, newlines,
                and tabs into a single space.
            unicode_normalize: The Unicode normalization form to apply. Can be one
                of 'NFC', 'NFKC', 'NFD', 'NFKD'. Defaults to None.
            minimum_chunk_size: Optional integer. If a generated chunk's size is
                below this, it will be handled by the specified strategy.
            min_chunk_merge_strategy: How to handle chunks smaller than
                `minimum_chunk_size`. Can be 'discard' or 'merge_with_previous'.

        Raises:
            ValueError: If `chunk_overlap` is not smaller than `chunk_size`.
            ValueError: If `unicode_normalize` is not a valid normalization form.
            ValueError: If `min_chunk_merge_strategy` is not a valid strategy.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"Chunk overlap ({chunk_overlap}) must be smaller than "
                f"chunk size ({chunk_size})."
            )
        if unicode_normalize and unicode_normalize not in [
            "NFC",
            "NFKC",
            "NFD",
            "NFKD",
        ]:
            raise ValueError(
                f"Invalid unicode_normalize form: {unicode_normalize}. "
                "Must be one of 'NFC', 'NFKC', 'NFD', 'NFKD'."
            )
        if minimum_chunk_size and minimum_chunk_size >= chunk_size:
            raise ValueError(
                f"minimum_chunk_size ({minimum_chunk_size}) must be smaller than "
                f"chunk_size ({chunk_size})."
            )
        if min_chunk_merge_strategy not in ["discard", "merge_with_previous"]:
            raise ValueError(
                f"Invalid min_chunk_merge_strategy: {min_chunk_merge_strategy}. "
                "Must be one of 'discard', 'merge_with_previous'."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
        self.normalize_whitespace = normalize_whitespace
        self.unicode_normalize = unicode_normalize
        self.minimum_chunk_size = minimum_chunk_size or 0
        self.min_chunk_merge_strategy = min_chunk_merge_strategy

    def _enforce_minimum_chunk_size(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Enforces the minimum chunk size by merging or discarding small chunks.

        This method is called as a post-processing step by concrete splitters.
        """
        if self.minimum_chunk_size <= 0:
            return chunks

        if self.min_chunk_merge_strategy == "discard":
            return [
                c
                for c in chunks
                if self.length_function(c.content) >= self.minimum_chunk_size
            ]

        if self.min_chunk_merge_strategy == "merge_with_previous":
            merged_chunks: List[Chunk] = []
            for chunk in chunks:
                if self.length_function(chunk.content) < self.minimum_chunk_size and merged_chunks:
                    # Merge this small chunk with the previous one
                    last_chunk = merged_chunks[-1]
                    last_chunk.content += chunk.content
                    last_chunk.end_index = chunk.end_index
                else:
                    # Append a copy to avoid modifying the original list's objects
                    merged_chunks.append(copy.copy(chunk))

            # Re-assign sequence numbers
            for i, chunk in enumerate(merged_chunks):
                chunk.sequence_number = i

            return merged_chunks

        return chunks

    def _preprocess(self, text: str) -> str:
        """
        Applies configured preprocessing steps to the input text.

        This method is called by concrete splitter implementations before chunking.
        """
        if self.unicode_normalize:
            text = unicodedata.normalize(self.unicode_normalize, text)
        if self.normalize_whitespace:
            # Collapse consecutive whitespace (spaces, tabs, newlines) into a single space
            text = re.sub(r"\s+", " ", text).strip()
        return text

    @abstractmethod
    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Abstract method to split a document text into a list of `Chunk` objects.

        Every concrete implementation of a chunking strategy must implement this
        method.

        Args:
            text: The input text of the document to be split.
            source_document_id: An optional identifier for the source document,
                which will be attached to each resulting chunk.

        Returns:
            A list of `Chunk` objects, each representing a segment of the input text.
        """
        pass

    def chunk(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        A convenience method that provides a more intuitive alias for `split_text`.

        Args:
            text: The input text to be chunked.
            source_document_id: An optional identifier for the source document.

        Returns:
            A list of `Chunk` objects.
        """
        return self.split_text(text, source_document_id=source_document_id)
