import warnings
from typing import Any, Dict, List, Optional

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter
from text_segmentation.utils import merge_chunks

try:
    from bs4 import BeautifulSoup
    BS4_LXML_AVAILABLE = True
except ImportError:
    BS4_LXML_AVAILABLE = False


class HTMLSplitter(TextSplitter):
    """
    Splits HTML documents into chunks based on structural tags.
    """

    def __init__(
        self,
        block_tags: Optional[List[str]] = None,
        strip_tags: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if not BS4_LXML_AVAILABLE:
            raise ImportError(
                "BeautifulSoup4 is not installed. Please install it via `pip install "
                '"advanced-text-segmentation[html]"` or `pip install beautifulsoup4 lxml`.'
            )

        self.block_tags = block_tags or [
            "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "table"
        ]
        self.strip_tags = strip_tags or ["script", "style", "nav", "footer", "header"]

    def _extract_base_units(self, text: str) -> List[Chunk]:
        """Extracts structural blocks from HTML as base units."""
        soup = BeautifulSoup(text, "lxml")

        for tag_name in self.strip_tags:
            for tag_obj in soup.select(tag_name):
                tag_obj.decompose()

        base_units: List[Chunk] = []

        for tag in soup.find_all(self.block_tags):
            # We only want to process tags that are direct children of other block tags
            # or the body, to avoid over-chunking nested elements (like a p inside a li).
            if tag.parent and tag.parent.name in self.block_tags and tag.parent.name != 'body':
                 continue

            content = tag.get_text(strip=True)
            if not content:
                continue

            # Finding start_index is hard in BeautifulSoup. We search for the content.
            # This is not perfect but better than sourcepos.
            try:
                start_index = text.find(content)
            except:
                start_index = -1

            if start_index == -1:
                warnings.warn(f"Could not reliably determine start index for content: '{content[:50]}...'")
                continue

            header_context: Dict[str, Any] = {}
            for parent in tag.find_parents():
                if parent.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    level = parent.name.upper()
                    if level not in header_context:
                         header_context[level] = parent.get_text(strip=True)

            sorted_context = dict(sorted(header_context.items(), key=lambda item: item[0]))

            base_units.append(Chunk(
                content=content,
                start_index=start_index,
                end_index=start_index + len(content),
                sequence_number=-1, # Placeholder
                hierarchical_context=sorted_context
            ))

        # Sort by appearance in the original text
        base_units.sort(key=lambda c: c.start_index)
        return base_units

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the HTML text into structure-aware chunks."""
        if not text.strip():
            return []

        base_units = self._extract_base_units(text)
        if not base_units:
            fallback_splitter = RecursiveCharacterSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=self.length_function
            )
            return fallback_splitter.split_text(text, source_document_id)

        final_chunks = merge_chunks(
            base_units=base_units,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            minimum_chunk_size=self.chunk_size // 4,
        )

        for i, chunk in enumerate(final_chunks):
            chunk.sequence_number = i
            chunk.source_document_id = source_document_id
            chunk.chunking_strategy_used = "html"

        return final_chunks
