from typing import Any, Dict, List, Optional, Tuple

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter

try:
    from markdown_it import MarkdownIt
    from markdown_it.tree import SyntaxTreeNode
except ImportError:
    MarkdownIt = None  # type: ignore


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
        # Enable the 'sourcepos' option to ensure the 'map' attribute is populated.
        self.md_parser = MarkdownIt("commonmark", {"sourcepos": True})
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            normalize_whitespace=False,
            unicode_normalize=False,
        )

    def _get_line_start_indices(self, text: str) -> List[int]:
        """Calculates the character start index of each line in the text."""
        indices = [0]
        for line in text.splitlines(keepends=True):
            indices.append(indices[-1] + len(line))
        return indices

    def _extract_blocks(
        self, text: str, line_start_indices: List[int]
    ) -> List[Tuple[str, int, int, Dict[str, Any]]]:
        """
        Parses the text and extracts top-level semantic blocks using a syntax tree.
        """
        tokens = self.md_parser.parse(text)
        root_node = SyntaxTreeNode(tokens)

        blocks = []
        header_context: Dict[str, Any] = {}

        for node in root_node.children:
            if not node.map:
                continue

            # Update header context when we encounter a new heading
            if node.type == "heading":
                level = int(node.tag[1:])
                keys_to_del = [k for k in header_context if int(k[1:]) >= level]
                for k in keys_to_del:
                    del header_context[k]

                # The inline content of the header is in its children
                if node.children and node.children[0].type == "inline":
                    header_content = node.children[0].content.strip()
                    header_context[f"H{level}"] = header_content

            start_line, end_line = node.map
            start_char = line_start_indices[start_line]

            if end_line >= len(line_start_indices):
                end_char = len(text)
            else:
                end_char = line_start_indices[end_line]

            block_content = text[start_char:end_char].strip()

            if block_content:
                # The start/end indices from the map are for the raw block.
                # We use these to create chunks later.
                blocks.append(
                    (block_content, start_char, end_char, header_context.copy())
                )

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
                fallback_chunks = self._fallback_splitter.split_text(block_content)
                for fb_chunk in fallback_chunks:
                    # Correct the indices of the fallback chunks to be relative to the
                    # original document by adding the start offset of the block.
                    corrected_start_index = block_start + fb_chunk.start_index
                    corrected_end_index = block_start + fb_chunk.end_index

                    chunks.append(Chunk(
                        content=fb_chunk.content,
                        start_index=corrected_start_index,
                        end_index=corrected_end_index,
                        sequence_number=sequence_number,
                        source_document_id=source_document_id,
                        hierarchical_context=block_context.copy(),
                        chunking_strategy_used="markdown-fallback"
                    ))
                    sequence_number += 1
            else:
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

        from text_segmentation.utils import _populate_overlap_metadata
        _populate_overlap_metadata(chunks, processed_text)

        return self._enforce_minimum_chunk_size(chunks, processed_text)
