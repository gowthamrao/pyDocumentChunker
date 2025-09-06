from typing import Any, Callable, List, Optional, Tuple

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter
from text_segmentation.utils import _populate_overlap_metadata

# Attempt to import spacy and load the model.
try:
    import spacy
    # We use a lightweight model by default.
    try:
        NLP = spacy.load("en_core_web_sm")
    except OSError:
        raise ImportError(
            "Spacy model 'en_core_web_sm' not found. Please run "
            "'python -m spacy download en_core_web_sm' to install it."
        ) from None
except ImportError:
    NLP = None


class SpacySentenceSplitter(TextSplitter):
    """
    A sentence splitter that uses the `spacy` library for robust sentence
    boundary detection (SBD).

    This strategy is generally more accurate than using regular expressions or simple
    delimiters, as it leverages trained NLP models to understand sentence
    structure. It requires the `[spacy]` optional dependency to be installed.

    The splitter first divides the text into sentences and then aggregates them
    into chunks that respect the configured `chunk_size`.

    Attributes:
        overlap_sentences (int): The number of sentences to overlap between
            consecutive chunks. This provides context continuity.
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

        Args:
            chunk_size: The target maximum size of a chunk, measured by the
                `length_function`.
            length_function: The function to measure text length. Defaults to `len`.
            overlap_sentences: The number of sentences to include as overlap
                between consecutive chunks.
            **kwargs: Additional arguments to pass to the `TextSplitter` base class.

        Raises:
            ImportError: If the `spacy` library is not installed or the default
                model is not downloaded.
            ValueError: If `overlap_sentences` is negative.
        """
        if NLP is None:
            raise ImportError(
                "Spacy is not installed. Please install it with "
                "`pip install 'advanced-text-segmentation[spacy]'`."
            )
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be a non-negative integer.")

        # Pass chunk_overlap=0 to the parent; this strategy handles overlap
        # at the sentence level.
        super().__init__(
            chunk_size=chunk_size,
            length_function=length_function,
            chunk_overlap=0,
            **kwargs,
        )
        self.overlap_sentences = overlap_sentences

        # Fallback splitter for sentences that are too long
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0, # Overlap is handled by the main logic
            length_function=self.length_function,
            normalize_whitespace=False, # Already handled by _preprocess
            unicode_normalize=None,   # Already handled by _preprocess
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

        # Type annotation for a sentence tuple: (text, start_char, end_char)
        Sentence = Tuple[str, int, int]

        doc = NLP(processed_text)
        raw_sentences: List[Sentence] = [
            (sent.text.strip(), sent.start_char, sent.end_char)
            for sent in doc.sents if sent.text.strip()
        ]

        if not raw_sentences:
            return []

        # Handle sentences that are individually larger than the chunk size
        all_sentences: List[Sentence] = []
        for sentence_text, start, end in raw_sentences:
            if self.length_function(sentence_text) > self.chunk_size:
                # This sentence is too long, so we split it with the fallback
                fallback_chunks = self._fallback_splitter.split_text(sentence_text)
                for fb_chunk in fallback_chunks:
                    all_sentences.append(
                        (fb_chunk.content, start + fb_chunk.start_index, start + fb_chunk.end_index)
                    )
            else:
                all_sentences.append((sentence_text, start, end))


        chunks: List[Chunk] = []
        current_chunk_sents: List[Sentence] = []
        current_chunk_len = 0

        for sent, start, end in all_sentences:
            sent_len = self.length_function(sent)

            # Check if adding the new sentence would exceed the chunk size
            # The length check accounts for the spaces that will be added between sentences
            potential_len = current_chunk_len + sent_len
            if current_chunk_sents:
                potential_len += 1 # For the space separator

            if potential_len > self.chunk_size and current_chunk_sents:
                # Finalize the current chunk
                content = " ".join(s[0] for s in current_chunk_sents)
                chunks.append(
                    Chunk(
                        content=content,
                        start_index=current_chunk_sents[0][1],
                        end_index=current_chunk_sents[-1][2],
                        sequence_number=len(chunks),
                        source_document_id=source_document_id,
                        chunking_strategy_used="spacy_sentence",
                    )
                )

                # Start a new chunk with the specified sentence overlap
                overlap_idx = max(0, len(current_chunk_sents) - self.overlap_sentences)
                current_chunk_sents = current_chunk_sents[overlap_idx:]

            current_chunk_sents.append((sent, start, end))
            # Recalculate current_chunk_len after potential overlap change
            current_chunk_len = self.length_function(" ".join(s[0] for s in current_chunk_sents))

        # Add the last remaining chunk
        if current_chunk_sents:
            content = " ".join(s[0] for s in current_chunk_sents)
            chunks.append(
                Chunk(
                    content=content,
                    start_index=current_chunk_sents[0][1],
                    end_index=current_chunk_sents[-1][2],
                    sequence_number=len(chunks),
                    source_document_id=source_document_id,
                    chunking_strategy_used="spacy_sentence",
                )
            )

        # Populate overlap metadata between the final chunks
        _populate_overlap_metadata(chunks, processed_text)

        # Handle any runts (undersized chunks) as per the configuration
        return self._enforce_minimum_chunk_size(chunks, processed_text)
