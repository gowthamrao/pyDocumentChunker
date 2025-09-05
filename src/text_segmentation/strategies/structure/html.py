import warnings
from typing import Any, Dict, List, Optional, Tuple

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter

from bs4 import BeautifulSoup, NavigableString

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
        )

    def _extract_blocks(self, soup: BeautifulSoup) -> List[Tuple[str, int, Dict[str, Any]]]:
        """Extracts text blocks from the parsed soup, with metadata."""
        blocks = []
        for tag in soup.find_all(self.block_tags):
            content = tag.get_text(separator=" ", strip=True)
            if not content:
                continue

            # Get start index from sourceline/sourcepos, provided by lxml
            start_index = 0
            if hasattr(tag, 'sourcepos'):
                start_index = tag.sourcepos
            else:
                 warnings.warn(
                    f"Tag '{tag.name}' does not have 'sourcepos' attribute. "
                    "Start index may be inaccurate. Ensure you are using the 'lxml' parser.",
                    UserWarning
                )

            # Build hierarchical context from parent headers
            header_context: Dict[str, Any] = {}
            for parent in tag.find_parents():
                if parent.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    level = parent.name.upper()
                    if level not in header_context: # Only take the closest header of each level
                         header_context[level] = parent.get_text(separator=" ", strip=True)

            # Sort context by header level
            sorted_context = dict(sorted(header_context.items(), key=lambda item: item[0]))

            blocks.append((content, start_index, sorted_context))

        # Sort blocks by their appearance in the document
        blocks.sort(key=lambda x: x[1])
        return blocks

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the HTML text into structure-aware chunks."""
        if not text.strip():
            return []

        soup = BeautifulSoup(text, "lxml")

        # Strip irrelevant tags
        for tag in self.strip_tags:
            for s in soup.select(tag):
                s.decompose()

        blocks = self._extract_blocks(soup)

        # Process blocks into chunks
        chunks: List[Chunk] = []
        current_chunk_blocks: List[Tuple[str, int, Dict[str, Any]]] = []
        current_length = 0
        sequence_number = 0

        for block_text, block_start, block_context in blocks:
            block_len = self.length_function(block_text)

            if block_len > self.chunk_size:
                if current_chunk_blocks:
                    content = " ".join(b[0] for b in current_chunk_blocks)
                    start_idx = current_chunk_blocks[0][1]
                    chunks.append(Chunk(content=content, start_index=start_idx, end_index=start_idx + len(content), sequence_number=sequence_number, source_document_id=source_document_id, hierarchical_context=current_chunk_blocks[0][2], chunking_strategy_used="html"))
                    sequence_number += 1
                    current_chunk_blocks, current_length = [], 0

                fallback_chunks = self._fallback_splitter.split_text(block_text)
                for fb_chunk in fallback_chunks:
                    chunks.append(Chunk(content=fb_chunk.content, start_index=block_start + fb_chunk.start_index, end_index=block_start + fb_chunk.end_index, sequence_number=sequence_number, source_document_id=source_document_id, hierarchical_context=block_context, chunking_strategy_used="html-fallback"))
                    sequence_number += 1
                continue

            if current_length > 0 and current_length + block_len > self.chunk_size:
                content = " ".join(b[0] for b in current_chunk_blocks)
                start_idx = current_chunk_blocks[0][1]
                chunks.append(Chunk(content=content, start_index=start_idx, end_index=start_idx + len(content), sequence_number=sequence_number, source_document_id=source_document_id, hierarchical_context=current_chunk_blocks[0][2], chunking_strategy_used="html"))
                sequence_number += 1
                current_chunk_blocks, current_length = [], 0

            current_chunk_blocks.append((block_text, block_start, block_context))
            current_length += block_len

        if current_chunk_blocks:
            content = " ".join(b[0] for b in current_chunk_blocks)
            start_idx = current_chunk_blocks[0][1]
            chunks.append(Chunk(content=content, start_index=start_idx, end_index=start_idx + len(content), sequence_number=sequence_number, source_document_id=source_document_id, hierarchical_context=current_chunk_blocks[0][2], chunking_strategy_used="html"))

        return chunks
