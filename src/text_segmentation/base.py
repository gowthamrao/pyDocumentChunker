from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from text_segmentation.core import Chunk


class TextSplitter(ABC):
    """
    Abstract Base Class for all text splitters in the package.

    This class establishes a common interface for all chunking strategies, ensuring
    modularity and extensibility as per the FRD (R-6.1.1). It also handles the
    core configuration parameters that are common across most strategies (R-4.1).

    Attributes:
        chunk_size (int): The target maximum size of each chunk, as measured by the
            `length_function` (R-4.1.2).
        chunk_overlap (int): The amount of overlap between consecutive chunks,
            ensuring context continuity (R-4.1.3).
        length_function (Callable[[str], int]): The function used to measure the
            length of a string. Defaults to `len` (R-4.1.1).
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        length_function: Optional[Callable[[str], int]] = None,
    ):
        """
        Initializes the TextSplitter.

        Args:
            chunk_size: The maximum size of a chunk.
            chunk_overlap: The overlap between consecutive chunks.
            length_function: The function to measure text length. Defaults to `len`.

        Raises:
            ValueError: If `chunk_overlap` is not smaller than `chunk_size`.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"Chunk overlap ({chunk_overlap}) must be smaller than "
                f"chunk size ({chunk_size})."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len

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
