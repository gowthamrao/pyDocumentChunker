from typing import Any, Dict, List, Optional, Tuple

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter

from markdown_it import MarkdownIt
from markdown_it.token import Token


class MarkdownSplitter(TextSplitter):
    """
    Splits a Markdown document based on its structural elements.

    This strategy parses the Markdown to identify structural boundaries (like
    headers, paragraphs, and lists) and uses them as the primary basis for
    splitting. This preserves the logical sections of the document.

    The splitter first breaks the document down into fundamental blocks (e.g.,
    a header and its content, a paragraph), capturing the hierarchical context
    of each. It then merges these blocks into chunks of the desired size.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the MarkdownSplitter."""
        super().__init__(*args, **kwargs)
        self.md_parser = MarkdownIt()
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
        )

    def _get_line_start_indices(self, text: str) -> List[int]:
        """Calculates the starting character index of each line in the text."""
        indices = [0]
        for line in text.splitlines(keepends=True):
            indices.append(indices[-1] + len(line))
        return indices

    def _extract_blocks(
        self, text: str, line_start_indices: List[int]
    ) -> List[Tuple[str, int, int, Dict[str, Any]]]:
        """Parses text and extracts a list of structural blocks."""
        tokens = self.md_parser.parse(text)
        blocks = []
        header_context: Dict[str, Any] = {}

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type.endswith("_open"):
                block_type = token.tag
                start_line = token.map[0] if token.map else 0

                # Find all content tokens until the corresponding closing tag
                content_tokens = []
                level = token.level
                i += 1
                while i < len(tokens) and not (
                    tokens[i].type == f"{block_type}_close" and tokens[i].level == level
                ):
                    if tokens[i].type == "inline" and tokens[i].content:
                        content_tokens.append(tokens[i])
                    i += 1

                end_line = tokens[i].map[1] if tokens[i].map else start_line + 1

                content = "".join(t.content for t in content_tokens).strip()
                if not content:
                    i += 1
                    continue

                start_index = line_start_indices[start_line]
                # Find the actual start of content, not the start of the markdown tag
                try:
                    content_start_offset = text[start_index:].find(content)
                    if content_start_offset != -1:
                        start_index += content_start_offset
                except ValueError:
                    pass # Keep original start_index if find fails

                end_index = start_index + len(content)

                # Update header context
                if block_type.startswith("h"):
                    level = int(block_type[1])
                    # Remove deeper headers from context
                    keys_to_del = [k for k in header_context if int(k[1]) >= level]
                    for k in keys_to_del:
                        del header_context[k]
                    header_context[block_type.upper()] = content

                blocks.append((content, start_index, end_index, header_context.copy()))
            i += 1
        return blocks

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the Markdown text into structure-aware chunks."""
        if not text.strip():
            return []

        line_start_indices = self._get_line_start_indices(text)
        blocks = self._extract_blocks(text, line_start_indices)

        # Process blocks into chunks
        chunks: List[Chunk] = []
        current_chunk_blocks: List[Tuple[str, int, int, Dict[str, Any]]] = []
        current_length = 0
        sequence_number = 0

        for block_text, block_start, block_end, block_context in blocks:
            block_len = self.length_function(block_text)

            # If a single block is too large, split it with the fallback
            if block_len > self.chunk_size:
                if current_chunk_blocks: # Finalize previous chunk first
                    content = "".join(b[0] for b in current_chunk_blocks)
                    chunks.append(Chunk(content=content, start_index=current_chunk_blocks[0][1], end_index=current_chunk_blocks[-1][2], sequence_number=sequence_number, source_document_id=source_document_id, hierarchical_context=current_chunk_blocks[0][3], chunking_strategy_used="markdown"))
                    sequence_number += 1
                    current_chunk_blocks = []
                    current_length = 0

                fallback_chunks = self._fallback_splitter.split_text(block_text)
                for fb_chunk in fallback_chunks:
                    chunks.append(Chunk(content=fb_chunk.content, start_index=block_start + fb_chunk.start_index, end_index=block_start + fb_chunk.end_index, sequence_number=sequence_number, source_document_id=source_document_id, hierarchical_context=block_context, chunking_strategy_used="markdown-fallback"))
                    sequence_number += 1
                continue

            if current_length > 0 and current_length + block_len > self.chunk_size:
                content = "".join(b[0] for b in current_chunk_blocks)
                chunks.append(Chunk(content=content, start_index=current_chunk_blocks[0][1], end_index=current_chunk_blocks[-1][2], sequence_number=sequence_number, source_document_id=source_document_id, hierarchical_context=current_chunk_blocks[0][3], chunking_strategy_used="markdown"))
                sequence_number += 1
                current_chunk_blocks = []
                current_length = 0

            current_chunk_blocks.append((block_text, block_start, block_end, block_context))
            current_length += block_len

        if current_chunk_blocks:
            content = "".join(b[0] for b in current_chunk_blocks)
            chunks.append(Chunk(content=content, start_index=current_chunk_blocks[0][1], end_index=current_chunk_blocks[-1][2], sequence_number=sequence_number, source_document_id=source_document_id, hierarchical_context=current_chunk_blocks[0][3], chunking_strategy_used="markdown"))

        # Overlap is not naturally handled by this splitter, but we can add it if needed.
        # For now, we prioritize structural boundaries over a fixed overlap.

        return chunks
