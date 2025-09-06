import warnings
from typing import List, Optional

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk


class FixedSizeSplitter(TextSplitter):
    """
    Splits text into chunks of a fixed character size.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.length_function is not len:
            warnings.warn(
                "The 'FixedSizeSplitter' operates on character counts, but a custom "
                "`length_function` was provided. The splitting logic will not use this "
                "function, which may lead to unexpected chunk sizes in terms of tokens.",
                UserWarning,
            )

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Splits the input text into fixed-size character chunks.
        """
        text = self._preprocess(text)
        if not text:
            return []

        chunks: List[Chunk] = []
        start_index = 0
        sequence_number = 0
        step = self.chunk_size - self.chunk_overlap

        while start_index < len(text):
            end_index = start_index + self.chunk_size
            chunk_content = text[start_index:end_index]

            if not chunk_content:
                break

            chunk = Chunk(
                content=chunk_content,
                start_index=start_index,
                end_index=start_index + len(chunk_content),
                sequence_number=sequence_number,
                source_document_id=source_document_id,
                chunking_strategy_used="fixed_size",
            )
            chunks.append(chunk)

            start_index += step
            sequence_number += 1

        return self._enforce_minimum_chunk_size(chunks)
