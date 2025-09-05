from typing import Any, Callable, List, Literal, Optional

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.sentence import SentenceSplitter
from text_segmentation.utils import merge_chunks

try:
    import numpy as np
    from numpy.linalg import norm
except ImportError:
    np = None  # type: ignore

BreakpointMethod = Literal["percentile", "std_dev", "absolute"]


class SemanticSplitter(TextSplitter):
    """
    Splits text into semantically coherent chunks based on embedding similarity.

    This advanced strategy works by splitting the text into sentences, embedding each
    sentence, and then identifying breakpoints where the semantic similarity between
    adjacent sentences drops significantly. These sentence groups are then merged
    into chunks that respect the configured size and overlap.
    """

    def __init__(
        self,
        embedding_function: Callable[[List[str]], Any],
        breakpoint_method: BreakpointMethod = "percentile",
        breakpoint_threshold: float = 95.0,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initializes the SemanticSplitter.

        Args:
            embedding_function: A callable that takes a list of strings and returns
                a list or numpy array of their embeddings.
            breakpoint_method: The method for determining breakpoints ('percentile',
                'std_dev', 'absolute').
            breakpoint_threshold: The sensitivity for the breakpoint method.
                - For 'percentile', this is the percentile (0-100) of similarity scores.
                - For 'std_dev', it's the number of standard deviations from the mean.
                - For 'absolute', it's the fixed similarity value.
            *args, **kwargs: Additional arguments for the base `TextSplitter`.
        """
        super().__init__(*args, **kwargs)

        if np is None:
            raise ImportError(
                "numpy is not installed. Please install it via `pip install "
                "\"advanced-text-segmentation[semantic]\"` or `pip install numpy`."
            )
        self.embedding_function = embedding_function
        self.breakpoint_method = breakpoint_method
        self.breakpoint_threshold = breakpoint_threshold

        # Use SentenceSplitter to get correctly indexed sentences
        self._sentence_splitter = SentenceSplitter()

    def _calculate_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculates cosine similarity between adjacent embedding vectors."""
        normalized_embeddings = embeddings / norm(embeddings, axis=1, keepdims=True)
        return np.sum(normalized_embeddings[:-1] * normalized_embeddings[1:], axis=1)

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the text by merging semantically-grouped sentences."""
        # 1. Get sentences as the base unit for comparison.
        sentences = self._sentence_splitter.split_text(text)
        if len(sentences) < 2:
            return sentences  # Return single sentence as a chunk if not enough to compare

        sentence_texts = [s.content for s in sentences]
        embeddings = np.array(self.embedding_function(sentence_texts))
        similarities = self._calculate_similarities(embeddings)

        # 2. Determine the similarity threshold for a breakpoint.
        if self.breakpoint_method == "percentile":
            if not (0 <= self.breakpoint_threshold <= 100):
                raise ValueError("Percentile threshold must be between 0 and 100.")
            threshold = np.percentile(similarities, self.breakpoint_threshold)
        elif self.breakpoint_method == "std_dev":
            # A lower similarity score means a larger semantic gap.
            # We want to split where the similarity is much lower than the average.
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            threshold = mean_sim - self.breakpoint_threshold * std_sim
        else:  # 'absolute'
            threshold = self.breakpoint_threshold

        # 3. Identify sentence indices that are breakpoints.
        breakpoint_indices = [i for i, s in enumerate(similarities) if s < threshold]

        # 4. Create base units (groups of sentences) based on breakpoints.
        base_units: List[Chunk] = []
        start_sent_idx = 0
        for bp_idx in breakpoint_indices:
            end_sent_idx = bp_idx + 1
            group = sentences[start_sent_idx:end_sent_idx]
            if not group:
                continue

            content = "".join(s.content for s in group)
            base_units.append(
                Chunk(
                    content=content,
                    start_index=group[0].start_index,
                    end_index=group[-1].end_index,
                    sequence_number=-1, # Placeholder
                )
            )
            start_sent_idx = end_sent_idx

        # Handle the last group of sentences
        last_group = sentences[start_sent_idx:]
        if last_group:
            content = "".join(s.content for s in last_group)
            base_units.append(
                Chunk(
                    content=content,
                    start_index=last_group[0].start_index,
                    end_index=last_group[-1].end_index,
                    sequence_number=-1,
                )
            )

        if not base_units:
            return []

        # 5. Use the utility to merge base units into final chunks
        final_chunks = merge_chunks(
            base_units=base_units,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            minimum_chunk_size=self.chunk_size // 4, # Sensible default
        )

        # Final pass to add source_document_id and strategy name
        for i, chunk in enumerate(final_chunks):
            chunk.sequence_number = i
            chunk.source_document_id = source_document_id
            chunk.chunking_strategy_used = "semantic"

        return final_chunks
