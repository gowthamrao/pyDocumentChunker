from typing import Any, Dict, List, Optional

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.utils import merge_chunks
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter

try:
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MARKDOWN_IT_AVAILABLE = False


class MarkdownSplitter(TextSplitter):
    """
    Splits a Markdown document based on its structural elements.

    This strategy parses the Markdown to identify structural boundaries (like
    headers, paragraphs, and lists) and uses them as the primary basis for
    splitting. This preserves the logical sections of the document.

    The splitter first breaks the document down into fundamental blocks,
    capturing the hierarchical context of each. It then merges these blocks
    into chunks of the desired size using the common merging utility.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the MarkdownSplitter."""
        super().__init__(*args, **kwargs)
        if not MARKDOWN_IT_AVAILABLE:
            raise ImportError(
                "markdown-it-py is not installed. Please install it via `pip install "
                '"advanced-text-segmentation[markdown]"`.'
            )
        self.md_parser = MarkdownIt()

    def _extract_base_units(self, text: str) -> List[Chunk]:
        """Parses text and extracts a list of structural blocks as base units."""
        tokens = self.md_parser.parse(text)
        base_units: List[Chunk] = []
        header_context: Dict[str, Any] = {}

        for i, token in enumerate(tokens):
            if token.type.endswith("_open"):
                # Find the content and the closing tag
                content_parts = []
                end_token_idx = i
                for j in range(i + 1, len(tokens)):
                    inner_token = tokens[j]
                    if inner_token.type == token.type.replace("_open", "_close") and inner_token.level == token.level:
                        end_token_idx = j
                        break
                    if inner_token.content:
                        content_parts.append(inner_token.content)

                if not token.map or not tokens[end_token_idx].map:
                    continue

                content = "".join(content_parts).strip()
                if not content:
                    continue

                # Calculate character indices from line numbers
                start_line, end_line = token.map[0], tokens[end_token_idx].map[1]
                lines = text.splitlines(keepends=True)
                start_index = sum(len(line) for line in lines[:start_line])
                # Find the actual start of the content, not the markdown tag
                content_start_offset = text[start_index:].find(content)
                if content_start_offset != -1:
                    start_index += content_start_offset

                end_index = start_index + len(content)

                # Update header context
                if token.tag.startswith("h"):
                    level = int(token.tag[1])
                    keys_to_del = [k for k in header_context if int(k[1]) >= level]
                    for k in keys_to_del:
                        del header_context[k]
                    header_context[token.tag.upper()] = content

                base_units.append(
                    Chunk(
                        content=text[start_index:end_index],
                        start_index=start_index,
                        end_index=end_index,
                        sequence_number=-1, # Placeholder
                        hierarchical_context=header_context.copy(),
                    )
                )

        return base_units


    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the Markdown text into structure-aware chunks."""
        if not text.strip():
            return []

        base_units = self._extract_base_units(text)
        if not base_units:
            # Fallback if no blocks were extracted
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
            minimum_chunk_size=self.chunk_size // 4, # Sensible default
        )

        for i, chunk in enumerate(final_chunks):
            chunk.sequence_number = i
            chunk.source_document_id = source_document_id
            chunk.chunking_strategy_used = "markdown"

        return final_chunks
