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
    def __init__(
        self,
        overlap_sentences: int = 1,
        nlp_backend: str = "nltk",
        *args: Any,
        **kwargs: Any,
    ):
        if "chunk_overlap" not in kwargs:
            chunk_size = kwargs.get("chunk_size", 1024)
            kwargs["chunk_overlap"] = int(chunk_size * 0.1)

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

        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0,
            length_function=self.length_function,
        )

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        if not text:
            return []

        import string
        tokenizer: PunktSentenceTokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        sentence_spans = list(tokenizer.span_tokenize(text))

        all_sentences = [
            (text[start:end], start, end)
            for start, end in sentence_spans
            if text[start:end].strip(string.punctuation + string.whitespace)
        ]

        if not all_sentences:
            return []

        processed_sentences = []
        for sentence_text, start, end in all_sentences:
            if self.length_function(sentence_text) > self.chunk_size:
                fallback_chunks = self._fallback_splitter.split_text(sentence_text)
                for fb_chunk in fallback_chunks:
                    processed_sentences.append(
                        (fb_chunk.content, start + fb_chunk.start_index, start + fb_chunk.end_index)
                    )
            else:
                processed_sentences.append((sentence_text, start, end))

        chunks: List[Chunk] = []
        current_chunk_sents: List[Tuple[str, int, int]] = []

        i = 0
        while i < len(processed_sentences):
            sent, start, end = processed_sentences[i]

            current_content = " ".join(s[0] for s in current_chunk_sents)
            if current_chunk_sents and self.length_function(current_content + " " + sent) > self.chunk_size:
                chunks.append(
                    Chunk(
                        content=current_content,
                        start_index=current_chunk_sents[0][1],
                        end_index=current_chunk_sents[-1][2],
                        sequence_number=len(chunks),
                        source_document_id=source_document_id,
                        chunking_strategy_used="sentence",
                    )
                )

                overlap_idx = max(0, len(current_chunk_sents) - self.overlap_sentences)
                current_chunk_sents = current_chunk_sents[overlap_idx:]
                continue

            current_chunk_sents.append((sent, start, end))
            i += 1

        if current_chunk_sents:
            content = " ".join(s[0] for s in current_chunk_sents)
            chunks.append(
                Chunk(
                    content=content,
                    start_index=current_chunk_sents[0][1],
                    end_index=current_chunk_sents[-1][2],
                    sequence_number=len(chunks),
                    source_document_id=source_document_id,
                    chunking_strategy_used="sentence",
                )
            )

        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            if next_chunk.start_index < current_chunk.end_index:
                overlap_content = text[next_chunk.start_index:current_chunk.end_index]
                current_chunk.overlap_content_next = overlap_content
                next_chunk.overlap_content_previous = overlap_content

        return chunks
