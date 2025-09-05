import re
from typing import Any, List, Optional

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter

try:
    import nltk
    from nltk.tokenize.punkt import PunktSentenceTokenizer
except ImportError:
    nltk = None


class SentenceSplitter(TextSplitter):
    """
    Splits text based on sentence boundaries and then aggregates sentences into chunks.
    """

    def __init__(
        self,
        overlap_sentences: int = 1,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if nltk is None:
            raise ImportError("NLTK is not installed.")
        try:
            # This is a check to ensure the model is available, raising a clear error if not.
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            raise RuntimeError("NLTK's 'punkt' tokenizer model is not downloaded.")

        self.overlap_sentences = overlap_sentences
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
        )

    def _get_sentences(self, text: str) -> List[Chunk]:
        """Splits text into sentences and returns them as Chunk objects."""
        if not text.strip():
            return []

        tokenizer: PunktSentenceTokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        sentence_spans = list(tokenizer.span_tokenize(text))

        if not sentence_spans:
            return []

        sentences = []
        for i, (start, end) in enumerate(sentence_spans):
            sentence_text = text[start:end]
            # Filter out sentences that are only punctuation or whitespace
            if re.search(r'[a-zA-Z0-9]', sentence_text):
                sentences.append(
                    Chunk(content=sentence_text, start_index=start, end_index=end, sequence_number=i)
                )
        return sentences

    def split_text(self, text: str, source_document_id: Optional[str] = None) -> List[Chunk]:
        if not text.strip():
            return []

        sentences = self._get_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_chunk_sentences: List[Chunk] = []

        for sentence in sentences:
            # Case 1: The sentence itself is larger than the chunk size.
            # Split it with the fallback splitter and add the results directly as chunks.
            if self.length_function(sentence.content) > self.chunk_size:
                # First, finalize any existing chunk.
                if current_chunk_sentences:
                    start_idx = current_chunk_sentences[0].start_index
                    end_idx = current_chunk_sentences[-1].end_index
                    chunks.append(Chunk(content=text[start_idx:end_idx], start_index=start_idx, end_index=end_idx, sequence_number=len(chunks)))
                    current_chunk_sentences = []

                # Split the oversized sentence and add its chunks.
                fallback_chunks = self._fallback_splitter.split_text(sentence.content)
                for fb_chunk in fallback_chunks:
                    chunks.append(Chunk(
                        content=fb_chunk.content,
                        start_index=sentence.start_index + fb_chunk.start_index,
                        end_index=sentence.start_index + fb_chunk.end_index,
                        sequence_number=len(chunks)
                    ))
                continue

            # Case 2: The current chunk is empty, so we start a new one.
            if not current_chunk_sentences:
                current_chunk_sentences.append(sentence)
                continue

            # Case 3: Check if adding the new sentence would exceed the chunk size.
            potential_chunk_text = text[current_chunk_sentences[0].start_index : sentence.end_index]
            if self.length_function(potential_chunk_text) > self.chunk_size:
                # Finalize the current chunk *without* the new sentence.
                start_idx = current_chunk_sentences[0].start_index
                end_idx = current_chunk_sentences[-1].end_index
                chunks.append(Chunk(content=text[start_idx:end_idx], start_index=start_idx, end_index=end_idx, sequence_number=len(chunks)))

                # Start a new chunk, respecting sentence overlap.
                overlap_start_idx = max(0, len(current_chunk_sentences) - self.overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[overlap_start_idx:]

            current_chunk_sentences.append(sentence)

        # Add the final chunk if there are any remaining sentences.
        if current_chunk_sentences:
            start_idx = current_chunk_sentences[0].start_index
            end_idx = current_chunk_sentences[-1].end_index
            chunks.append(Chunk(content=text[start_idx:end_idx], start_index=start_idx, end_index=end_idx, sequence_number=len(chunks)))

        # Final pass for metadata.
        for i, chunk in enumerate(chunks):
            chunk.sequence_number = i
            chunk.source_document_id = source_document_id
            chunk.chunking_strategy_used = "sentence"
            if i > 0:
                prev_chunk = chunks[i-1]
                if chunk.start_index < prev_chunk.end_index:
                    overlap_content = text[chunk.start_index:prev_chunk.end_index]
                    chunk.overlap_content_previous = overlap_content
                    prev_chunk.overlap_content_next = overlap_content

        return chunks
