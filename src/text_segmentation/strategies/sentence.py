from typing import Any, List, Optional

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter

try:
    import nltk
    from nltk.tokenize.punkt import PunktSentenceTokenizer
except ImportError:
    nltk = None  # type: ignore


class SentenceSplitter(TextSplitter):
    """
    Splits text based on sentence boundaries and then aggregates sentences into chunks.

    This strategy first tokenizes the text into individual sentences and then groups
    them into chunks of a desired size. It is particularly useful for prose and
    other grammatically structured texts.

    This implementation requires the NLTK library and its 'punkt' tokenizer model.
    It handles sentence-based overlap and uses a fallback splitter for sentences
    that individually exceed the `chunk_size`.
    """

    def __init__(
        self,
        overlap_sentences: int = 1,
        nlp_backend: str = "nltk",
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initializes the SentenceSplitter.

        Args:
            overlap_sentences: The number of sentences to overlap between consecutive chunks.
            nlp_backend: The NLP library to use for sentence tokenization. Currently,
                only 'nltk' is supported.
            *args, **kwargs: Additional arguments for the base `TextSplitter`.
        """
        super().__init__(*args, **kwargs)

        if nlp_backend != "nltk":
            raise ValueError("Currently, only the 'nltk' backend is supported.")
        if nltk is None:
            raise ImportError(
                "NLTK is not installed. Please install it via `pip install \"advanced-text-segmentation[nlp]\"` "
                "or `pip install nltk` to use the SentenceSplitter."
            )
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            raise RuntimeError(
                "NLTK's 'punkt' tokenizer model is not downloaded. Please run "
                "`python -c \"import nltk; nltk.download('punkt')\"` to download it."
            )

        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be a non-negative integer.")
        self.overlap_sentences = overlap_sentences

        # Use RecursiveCharacterSplitter as a fallback for oversized sentences
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
        )

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Splits the text by sentences and merges them into chunks.

        Args:
            text: The text to be split.
            source_document_id: Optional identifier for the source document.

        Returns:
            A list of `Chunk` objects.
        """
        if not text:
            return []

        # 1. Get sentences with their character spans
        tokenizer: PunktSentenceTokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        sentence_spans = list(tokenizer.span_tokenize(text))

        # Create a list of (text, start_index, end_index) tuples
        all_sentences = [
            (text[start:end], start, end) for start, end in sentence_spans
        ]

        if not all_sentences:
            return []

        # 2. Handle sentences that are larger than the chunk size
        processed_sentences = []
        for sentence_text, start, end in all_sentences:
            if self.length_function(sentence_text) > self.chunk_size:
                # This sentence is too long; split it with the fallback splitter.
                fallback_chunks = self._fallback_splitter.split_text(sentence_text)
                for fb_chunk in fallback_chunks:
                    # Adjust start/end indices to be relative to the original document
                    processed_sentences.append(
                        (fb_chunk.content, start + fb_chunk.start_index, start + fb_chunk.end_index)
                    )
            else:
                processed_sentences.append((sentence_text, start, end))

        # 3. Aggregate sentences into chunks
        chunks: List[Chunk] = []
        current_chunk_sents: List[Tuple[str, int, int]] = []
        current_length = 0
        sequence_number = 0

        for sent_text, sent_start, sent_end in processed_sentences:
            sent_len = self.length_function(sent_text)
            if current_length > 0 and current_length + sent_len > self.chunk_size:
                # Finalize the current chunk
                content = "".join(s[0] for s in current_chunk_sents)
                chunk = Chunk(
                    content=content,
                    start_index=current_chunk_sents[0][1],
                    end_index=current_chunk_sents[-1][2],
                    sequence_number=sequence_number,
                    source_document_id=source_document_id,
                    chunking_strategy_used="sentence",
                )
                chunks.append(chunk)
                sequence_number += 1

                # Start a new chunk with sentence-based overlap
                overlap_idx = max(0, len(current_chunk_sents) - self.overlap_sentences)
                current_chunk_sents = current_chunk_sents[overlap_idx:]
                current_length = sum(self.length_function(s[0]) for s in current_chunk_sents)

            current_chunk_sents.append((sent_text, sent_start, sent_end))
            current_length += sent_len

        # Add the final chunk
        if current_chunk_sents:
            content = "".join(s[0] for s in current_chunk_sents)
            chunk = Chunk(
                content=content,
                start_index=current_chunk_sents[0][1],
                end_index=current_chunk_sents[-1][2],
                sequence_number=sequence_number,
                source_document_id=source_document_id,
                chunking_strategy_used="sentence",
            )
            chunks.append(chunk)

        # Post-process to add overlap metadata
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            if next_chunk.start_index < current_chunk.end_index:
                overlap_content = text[next_chunk.start_index:current_chunk.end_index]
                current_chunk.overlap_content_next = overlap_content
                next_chunk.overlap_content_previous = overlap_content

        return chunks
