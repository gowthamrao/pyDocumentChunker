from typing import Any, Dict, List, Optional, Tuple

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter

try:
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
except ImportError:
    MarkdownIt = None  # type: ignore

# Define which token types should be considered as block separators
BLOCK_TOKENS = {
    "heading_open",
    "paragraph_open",
    "fence",
    "code_block",
    "blockquote_open",
    "table_open",
    "bullet_list_open",
    "ordered_list_open",
}


class MarkdownSplitter(TextSplitter):
    """
    Splits a Markdown document based on its structural elements.

    This strategy uses the `markdown-it-py` library to parse the Markdown into a
    syntax tree. It then traverses the tree to identify logical blocks of content
    (like headers, paragraphs, lists, code blocks) and uses them as the primary
    basis for splitting. This is more robust than regex-based methods and correctly
    handles complex and nested Markdown structures.

    This implementation fulfills FRD R-3.4 for Markdown.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the MarkdownSplitter."""
        super().__init__(*args, **kwargs)
        if MarkdownIt is None:
            raise ImportError(
                "markdown-it-py is not installed. Please install it via `pip install "
                "\"advanced-text-segmentation[markdown]\"` or `pip install markdown-it-py`."
            )
        self.md_parser = MarkdownIt("commonmark")
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,  # Allow overlap in fallback
            length_function=self.length_function,
            normalize_whitespace=False, # Pre-processing is handled before splitting
            unicode_normalize=False,
        )

    def _get_line_start_indices(self, text: str) -> List[int]:
        """Calculates the character start index of each line in the text."""
        # This is a critical helper for mapping token line numbers to character indices.
        indices = [0]
        for line in text.splitlines(keepends=True):
            indices.append(indices[-1] + len(line))
        return indices

    def _extract_blocks(
        self, text: str, line_start_indices: List[int]
    ) -> List[Tuple[str, int, int, Dict[str, Any]]]:
        """
        Parses the text and extracts semantic blocks with their content and metadata.

        Returns:
            A list of tuples, where each tuple contains:
            (content, start_char_index, end_char_index, context_dict)
        """
        tokens = self.md_parser.parse(text)
        blocks = []
        header_context: Dict[str, Any] = {}

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "heading_open":
                level = int(token.tag[1:])
                # Clear context from the same or lower level headers
                keys_to_del = [k for k in header_context if int(k[1:]) >= level]
                for k in keys_to_del:
                    del header_context[k]

                # Find the content of the header
                content_token = tokens[i+1]
                if content_token.type == "inline" and content_token.children:
                    header_content = "".join(t.content for t in content_token.children)
                    header_context[f"H{level}"] = header_content.strip()

                # The header itself is a block
                start_line = token.map[0]
                # Find the closing token to get the end line
                j = i + 1
                while j < len(tokens) and tokens[j].type != "heading_close":
                    j += 1
                end_line = tokens[j].map[1]

                start_char = line_start_indices[start_line]
                end_char = line_start_indices[end_line]
                block_content = text[start_char:end_char].strip()

                if block_content:
                    blocks.append((block_content, start_char, end_char, header_context.copy()))
                i = j + 1
                continue

            elif token.type in BLOCK_TOKENS:
                start_line, end_line_exclusive = token.map
                if end_line_exclusive is None:
                    i += 1
                    continue

                # Find the corresponding closing token to get the full block
                j = i + 1
                nesting_level = 1
                while j < len(tokens):
                    if tokens[j].level == token.level and tokens[j].nesting == -1:
                        nesting_level -= 1
                        if nesting_level == 0:
                            end_line_exclusive = tokens[j].map[1]
                            break
                    elif tokens[j].level == token.level and tokens[j].nesting == 1:
                         nesting_level += 1
                    j += 1

                start_char = line_start_indices[start_line]
                # The end line in map is exclusive, so it's the start of the next block
                end_char = line_start_indices[end_line_exclusive]
                block_content = text[start_char:end_char].strip()

                if block_content:
                    blocks.append((block_content, start_char, end_char, header_context.copy()))

                i = j + 1
                continue

            i += 1

        return blocks

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the Markdown text using its semantic structure."""
        processed_text = self._preprocess(text)
        if not processed_text.strip():
            return []

        line_start_indices = self._get_line_start_indices(processed_text)

        blocks = self._extract_blocks(processed_text, line_start_indices)

        chunks: List[Chunk] = []
        sequence_number = 0

        for block_content, block_start, block_end, block_context in blocks:
            if self.length_function(block_content) > self.chunk_size:
                # This block is oversized, so we must split it further.
                fallback_chunks = self._fallback_splitter.split_text(block_content)
                for fb_chunk in fallback_chunks:
                    # Adjust indices to be relative to the original document
                    chunk_start_index = block_start + fb_chunk.start_index
                    chunk_end_index = block_start + fb_chunk.end_index

                    chunks.append(Chunk(
                        content=fb_chunk.content,
                        start_index=chunk_start_index,
                        end_index=chunk_end_index,
                        sequence_number=sequence_number,
                        source_document_id=source_document_id,
                        hierarchical_context=block_context.copy(),
                        chunking_strategy_used="markdown-fallback"
                    ))
                    sequence_number += 1
            else:
                # This block fits, so create a single chunk for it.
                chunks.append(Chunk(
                    content=block_content,
                    start_index=block_start,
                    end_index=block_end,
                    sequence_number=sequence_number,
                    source_document_id=source_document_id,
                    hierarchical_context=block_context.copy(),
                    chunking_strategy_used="markdown"
                ))
                sequence_number += 1

        # The current implementation creates one chunk per block.
        # A future improvement could be to merge small, adjacent blocks together
        # to better utilize the chunk size, similar to the HTML or Sentence splitters.
        # For now, this approach is a major improvement and ensures structural integrity.

        from text_segmentation.utils import _populate_overlap_metadata
        _populate_overlap_metadata(chunks, processed_text)

        return self._enforce_minimum_chunk_size(chunks, processed_text)
