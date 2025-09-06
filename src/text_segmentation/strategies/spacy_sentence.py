from typing import Any, Callable, List, Optional, Tuple

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter
from text_segmentation.utils import _populate_overlap_metadata

# Attempt to import spacy and load the model.
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
except ImportError:
    # This will catch if spacy is not installed.
    NLP = None
except OSError:
    # This will catch if the model is not found.
    raise ImportError(
        "Spacy model 'en_core_web_sm' not found. Please run "
        "'python -m spacy download en_core_web_sm' to install it."
    ) from None


class SpacySentenceSplitter(TextSplitter):
    """
    A sentence splitter that uses the `spacy` library for robust sentence
    boundary detection (SBD).
    ...
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        length_function: Optional[Callable[[str], int]] = None,
        overlap_sentences: int = 0,
        **kwargs: Any,
    ):
        """
        Initializes the SpacySentenceSplitter.
        ...
        """
        if NLP is None:
            raise ImportError(
                "Spacy is not installed. Please install it with "
                "`pip install 'advanced-text-segmentation[spacy]'`."
            )
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be a non-negative integer.")

        super().__init__(
            chunk_size=chunk_size,
            length_function=length_function,
            chunk_overlap=0,
            **kwargs,
        )
        self.overlap_sentences = overlap_sentences

        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size * 0.1), # Add some overlap for fallbacks
            length_function=self.length_function,
            normalize_whitespace=False,
            unicode_normalize=None,
        )

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Splits a document into chunks of aggregated sentences using spacy.
        """
        processed_text = self._preprocess(text)
        if not processed_text:
            return []

        Sentence = Tuple[str, int, int]
        doc = NLP(processed_text)
        raw_sentences: List[Sentence] = [
            (sent.text.strip(), sent.start_char, sent.end_char)
            for sent in doc.sents if sent.text.strip()
        ]

        if not raw_sentences:
            return []

        chunks: List[Chunk] = []
        current_chunk_sents: List[Sentence] = []

        def finalize_chunk(current_sents: List[Sentence]) -> List[Sentence]:
            """Helper to create a chunk and handle overlap."""
            if not current_sents:
                return []

            content = " ".join(s[0] for s in current_sents)
            chunks.append(
                Chunk(
                    content=content,
                    start_index=current_sents[0][1],
                    end_index=current_sents[-1][2],
                    sequence_number=len(chunks),
                    source_document_id=source_document_id,
                    chunking_strategy_used="spacy_sentence",
                )
            )
            # Handle overlap for the next chunk
            if self.overlap_sentences > 0:
                overlap_idx = max(0, len(current_sents) - self.overlap_sentences)
                return current_sents[overlap_idx:]
            return []

        for sent_text, start, end in raw_sentences:
            sent_len = self.length_function(sent_text)

            # Case 1: The sentence itself is larger than the chunk size
            if sent_len > self.chunk_size:
                # First, finalize any existing chunk of smaller sentences
                current_chunk_sents = finalize_chunk(current_chunk_sents)

                # Then, split the oversized sentence using the fallback
                fallback_chunks = self._fallback_splitter.split_text(sent_text)
                for fb_chunk in fallback_chunks:
                    chunks.append(
                        Chunk(
                            content=fb_chunk.content,
                            start_index=start + fb_chunk.start_index,
                            end_index=start + fb_chunk.end_index,
                            sequence_number=len(chunks),
                            source_document_id=source_document_id,
                            chunking_strategy_used="spacy_sentence_fallback",
                        )
                    )
                continue

            # Case 2: The current sentence fits, but adding it would exceed the size
            current_len = self.length_function(" ".join(s[0] for s in current_chunk_sents))
            potential_len = current_len + (1 if current_chunk_sents else 0) + sent_len
            if potential_len > self.chunk_size and current_chunk_sents:
                current_chunk_sents = finalize_chunk(current_chunk_sents)

            # Add the current sentence to the buffer
            current_chunk_sents.append((sent_text, start, end))

        # Finalize the last remaining chunk
        finalize_chunk(current_chunk_sents)

        _populate_overlap_metadata(chunks, processed_text)
        return self._enforce_minimum_chunk_size(chunks, processed_text)
