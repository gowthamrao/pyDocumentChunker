import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter

# List of tags that are typically block-level and contain content.
DEFAULT_BLOCK_TAGS = [
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "th", "td", "caption"
]
# Tags to be removed completely from the document before processing.
DEFAULT_STRIP_TAGS = ["script", "style", "head", "nav", "footer", "aside"]


class HTMLSplitter(TextSplitter):
    """
    Splits an HTML document based on its structural tags.
    This strategy parses the HTML to identify structural elements (like paragraphs,
    headers, list items, etc.) and uses them as the primary basis for splitting.
    This is highly effective for preserving the logical sections of a web page
    or HTML document.
    The splitter requires `BeautifulSoup4` and `lxml` to be installed.
    """

    def __init__(
        self,
        block_tags: Optional[List[str]] = None,
        strip_tags: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the HTMLSplitter."""
        super().__init__(*args, **kwargs)

        self.block_tags = block_tags or DEFAULT_BLOCK_TAGS
        self.strip_tags = strip_tags or DEFAULT_STRIP_TAGS
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            normalize_whitespace=self.normalize_whitespace,
            unicode_normalize=self.unicode_normalize,
        )

    def _extract_blocks(
        self, soup: BeautifulSoup, line_start_indices: List[int]
    ) -> List[Tuple[str, int, int, Dict[str, Any]]]:
        """Extracts text blocks from the parsed soup, with metadata."""
        blocks = []
        for tag in soup.find_all(self.block_tags):
            # Get the raw text to calculate original end index
            raw_text = tag.get_text(separator=" ", strip=False)
            if not raw_text.strip():
                continue

            # Preprocess the text for the chunk content
            processed_content = self._preprocess(tag.get_text(separator=" ", strip=True))
            if not processed_content:
                continue

            start_index = 0
            if hasattr(tag, "sourceline") and tag.sourceline is not None:
                line = tag.sourceline - 1
                if line < len(line_start_indices):
                    # Approximate start position based on line number
                    start_index = line_start_indices[line]

            end_index = start_index + len(raw_text)

            header_context: Dict[str, Any] = {}
            current = tag
            while current:
                # Find preceding headers for context
                for sibling in current.find_previous_siblings():
                    if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        level = sibling.name.upper()
                        if level not in header_context:
                            header_context[level] = self._preprocess(
                                sibling.get_text(separator=" ", strip=True)
                            )
                current = current.parent

            sorted_context = dict(sorted(header_context.items(), key=lambda item: item[0]))
            blocks.append((processed_content, start_index, end_index, sorted_context))

        return blocks

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the HTML text into structure-aware chunks."""
        if not text.strip():
            return []

        # Pre-calculate start index of each line for position mapping
        line_start_indices = [0] + [m.end() for m in re.finditer("\n", text)]
        try:
            soup = BeautifulSoup(text, "html5lib")
        except Exception as e:
            warnings.warn(f"Failed to parse HTML with html5lib, falling back to lxml. Error: {e}")
            soup = BeautifulSoup(text, "lxml")


        for tag_name in self.strip_tags:
            for tag_obj in soup.select(tag_name):
                tag_obj.decompose()

        blocks = self._extract_blocks(soup, line_start_indices)

        chunks: List[Chunk] = []
        current_chunk_blocks: List[Tuple[str, int, int, Dict[str, Any]]] = []
        current_length = 0
        sequence_number = 0

        for block_text, block_start, block_end, block_context in blocks:
            block_len = self.length_function(block_text)

            if block_len > self.chunk_size:
                # Flush the current chunk if it exists
                if current_chunk_blocks:
                    content = " ".join(b[0] for b in current_chunk_blocks)
                    start_idx = current_chunk_blocks[0][1]
                    end_idx = current_chunk_blocks[-1][2]
                    merged_context = {}
                    for b in reversed(current_chunk_blocks):
                        merged_context.update(b[3])
                    chunks.append(
                        Chunk(
                            content=content,
                            start_index=start_idx,
                            end_index=end_idx,
                            sequence_number=sequence_number,
                            source_document_id=source_document_id,
                            hierarchical_context=merged_context,
                            chunking_strategy_used="html",
                        )
                    )
                    sequence_number += 1
                    current_chunk_blocks, current_length = [], 0

                # Split the oversized block using the fallback
                fallback_chunks = self._fallback_splitter.split_text(block_text)
                for fb_chunk in fallback_chunks:
                    # Note: Fallback indices are relative to the block, not the doc.
                    # This is a limitation when a single HTML block is too large.
                    chunks.append(
                        Chunk(
                            content=fb_chunk.content,
                            start_index=block_start, # The whole block's start index
                            end_index=block_end,     # The whole block's end index
                            sequence_number=sequence_number,
                            source_document_id=source_document_id,
                            hierarchical_context=block_context,
                            chunking_strategy_used="html-fallback",
                        )
                    )
                    sequence_number += 1
                continue

            # If adding the next block exceeds size, flush the current chunk
            if current_length > 0 and current_length + block_len > self.chunk_size:
                content = " ".join(b[0] for b in current_chunk_blocks)
                start_idx = current_chunk_blocks[0][1]
                end_idx = current_chunk_blocks[-1][2]
                merged_context = {}
                for b in reversed(current_chunk_blocks):
                    merged_context.update(b[3])
                chunks.append(
                    Chunk(
                        content=content,
                        start_index=start_idx,
                        end_index=end_idx,
                        sequence_number=sequence_number,
                        source_document_id=source_document_id,
                        hierarchical_context=merged_context,
                        chunking_strategy_used="html",
                    )
                )
                sequence_number += 1
                current_chunk_blocks, current_length = [], 0

            current_chunk_blocks.append((block_text, block_start, block_end, block_context))
            current_length += block_len

        # Flush any remaining blocks
        if current_chunk_blocks:
            content = " ".join(b[0] for b in current_chunk_blocks)
            start_idx = current_chunk_blocks[0][1]
            end_idx = current_chunk_blocks[-1][2]
            merged_context = {}
            for b in reversed(current_chunk_blocks):
                merged_context.update(b[3])
            chunks.append(
                Chunk(
                    content=content,
                    start_index=start_idx,
                    end_index=end_idx,
                    sequence_number=sequence_number,
                    source_document_id=source_document_id,
                    hierarchical_context=merged_context,
                    chunking_strategy_used="html",
                )
            )

        return self._enforce_minimum_chunk_size(chunks)
