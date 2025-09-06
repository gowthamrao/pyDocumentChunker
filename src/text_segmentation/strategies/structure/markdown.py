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
    Splits a Markdown document based on its structural elements in a hierarchical manner.

    This strategy uses `markdown-it-py` to parse the Markdown into a syntax tree.
    It then recursively traverses the tree to split the document, prioritizing
    higher-level structural boundaries (like H1, H2) over lower-level ones, in
    accordance with FRD R-3.4.3.

    If a semantic section is larger than the chunk size, it is recursively
    split by its sub-headers. If a section has no sub-headers and is still too
    large, a fallback `RecursiveCharacterSplitter` is used.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the MarkdownSplitter."""
        super().__init__(*args, **kwargs)
        if MarkdownIt is None:
            raise ImportError(
                "markdown-it-py is not installed. Please install it via `pip install "
                "\"advanced-text-segmentation[markdown]\"` or `pip install markdown-it-py`."
            )
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

    def _get_node_text(
        self, node: SyntaxTreeNode, text: str, line_indices: List[int]
    ) -> Tuple[str, int, int]:
        """Extracts the raw text content of a node, including all its children."""
        if not node.map or not node.children:
            start_line, end_line = node.map or (0, 0)
            start_char = line_indices[start_line]
            end_char = line_indices[end_line] if end_line < len(line_indices) else len(text)
            return text[start_char:end_char], start_char, end_char

        start_line, _ = node.children[0].map or (0, 0)
        end_line_node = node.children[-1]

        # Find the true end line by looking at the last descendant
        while end_line_node.children:
            end_line_node = end_line_node.children[-1]
        _, end_line = end_line_node.map or (0, 0)

        start_char = line_indices[start_line]
        end_char = line_indices[end_line] if end_line < len(line_indices) else len(text)
        return text[start_char:end_char], start_char, end_char

    def _get_nodes_text(
        self, nodes: List[SyntaxTreeNode], text: str, line_indices: List[int]
    ) -> Tuple[str, int, int]:
        """Extracts the raw text content of a list of nodes."""
        if not nodes:
            return "", 0, 0
        start_line, _ = nodes[0].map
        _, end_line = nodes[-1].map
        start_char = line_indices[start_line]
        end_char = line_indices[end_line] if end_line < len(line_indices) else len(text)
        return text[start_char:end_char], start_char, end_char

    def _extract_blocks(
        self, text: str
    ) -> List[Tuple[str, Dict[str, Any], int, int]]:
        """
        Pass 1: Parse the document and create a flat list of semantic blocks
        with their content, context, and character indices.
        """
        tokens = self.md_parser.parse(text)
        root_node = SyntaxTreeNode(tokens)
        line_indices = self._get_line_start_indices(text)

        blocks = []
        header_context: Dict[str, Any] = {}

        for node in root_node.children:
            if not node.map:
                continue

            start_line, end_line = node.map
            start_char = line_indices[start_line]
            end_char = line_indices[end_line] if end_line < len(line_indices) else len(text)
            content = text[start_char:end_char]

            if node.type == "heading":
                level = int(node.tag[1:])
                # Clear deeper or same-level headers from context
                keys_to_del = [k for k in header_context if int(k[1:]) >= level]
                for k in keys_to_del:
                    del header_context[k]
                header_content = node.children[0].content.strip() if node.children else ""
                header_context[f"H{level}"] = header_content

            if content.strip():
                blocks.append((content, header_context.copy(), start_char, end_char))

        return blocks

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the Markdown text using a two-pass, non-recursive method."""
        processed_text = self._preprocess(text)
        if not processed_text.strip():
            return []

        # Pass 1: Extract all semantic blocks with their context.
        blocks = self._extract_blocks(processed_text)
        if not blocks:
            return []

        # Pass 2: Group blocks into chunks.
        chunks: List[Chunk] = []
        current_chunk_blocks: List[Tuple[str, Dict[str, Any], int, int]] = []

        for block in blocks:
            # If the current chunk is empty, start it with the current block.
            if not current_chunk_blocks:
                current_chunk_blocks.append(block)
                continue

            # Get the combined text if we were to add the new block
            potential_text = "".join(b[0] for b in current_chunk_blocks) + block[0]

            # Get the context of the running chunk and the new block
            current_context = current_chunk_blocks[-1][1]
            new_context = block[1]

            # Check for a "context boundary". This happens if the new block starts
            # a new section that shouldn't be merged with the current one.
            context_boundary = False
            if new_context != current_context:
                # A new header always defines a new context. The question is whether
                # to merge it with the previous content or create a new chunk.
                # We create a new chunk if the new header is of the same or higher
                # level (e.g., H2 followed by H2, or H2 followed by H1).

                # Find the level of the header that defines the new block's context.
                new_header_level = 0
                for key in new_context:
                    if key not in current_context:
                        new_header_level = int(key[1:])
                        break

                # Find the level of the header that defines the current chunk's context
                current_header_level = 0
                for key in current_context:
                    if key not in new_context:
                        current_header_level = int(key[1:])
                        break

                # If there's no clear level change, default to the deepest level.
                if new_header_level == 0:
                    new_header_level = max((int(k[1:]) for k in new_context), default=99)
                if current_header_level == 0:
                    current_header_level = max((int(k[1:]) for k in current_context), default=99)

                if new_header_level <= current_header_level:
                    context_boundary = True

            # If adding the new block exceeds size, or we cross a context boundary,
            # finalize the current chunk.
            if (self.length_function(potential_text) > self.chunk_size) or context_boundary:
                chunk_content = "".join(b[0] for b in current_chunk_blocks)
                start_index = current_chunk_blocks[0][2]
                end_index = current_chunk_blocks[-1][3]

                chunks.append(Chunk(
                    content=chunk_content,
                    start_index=start_index,
                    end_index=end_index,
                    sequence_number=len(chunks),
                    source_document_id=source_document_id,
                    hierarchical_context=current_chunk_blocks[0][1],
                    chunking_strategy_used="markdown"
                ))
                # Start a new chunk with the current block
                current_chunk_blocks = [block]
            else:
                # Merge the block into the current chunk
                current_chunk_blocks.append(block)

        # Add the last remaining chunk
        if current_chunk_blocks:
            chunk_content = "".join(b[0] for b in current_chunk_blocks)
            start_index = current_chunk_blocks[0][2]
            end_index = current_chunk_blocks[-1][3]
            chunks.append(Chunk(
                content=chunk_content,
                start_index=start_index,
                end_index=end_index,
                sequence_number=len(chunks),
                source_document_id=source_document_id,
                hierarchical_context=current_chunk_blocks[0][1],
                chunking_strategy_used="markdown"
            ))

        # Pass 3: Use fallback for any chunks that are still too large.
        final_chunks: List[Chunk] = []
        for chunk in chunks:
            if self.length_function(chunk.content) > self.chunk_size:
                # This chunk is too big, and was formed by grouping smaller blocks.
                # We need to split it using a fallback mechanism.
                fallback_chunks = self._fallback_splitter.split_text(chunk.content)
                for fb_chunk in fallback_chunks:
                    final_chunks.append(Chunk(
                        content=fb_chunk.content,
                        start_index=chunk.start_index + fb_chunk.start_index,
                        end_index=chunk.start_index + fb_chunk.end_index,
                        sequence_number=len(final_chunks), # Assign sequence number immediately
                        source_document_id=source_document_id,
                        hierarchical_context=chunk.hierarchical_context,
                        chunking_strategy_used="markdown-fallback"
                    ))
            else:
                # This chunk is already valid.
                chunk.sequence_number = len(final_chunks) # Assign correct sequence number
                final_chunks.append(chunk)

        from text_segmentation.utils import _populate_overlap_metadata
        _populate_overlap_metadata(final_chunks, processed_text)

        return self._enforce_minimum_chunk_size(final_chunks, processed_text)
