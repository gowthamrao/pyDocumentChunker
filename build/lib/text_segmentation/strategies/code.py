from typing import Any, Dict, List, Optional, Set

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter

try:
    from tree_sitter import Language, Node, Parser
    from tree_sitter_languages import get_language, get_parser
except ImportError:
    Parser = None  # type: ignore

# Language-specific queries to find top-level chunkable nodes.
# This can be expanded for more languages and more granular control.
LANGUAGE_QUERIES: Dict[str, Set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "lexical_declaration",  # const/let variables
        "export_statement",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "rust": {
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "trait_item",
    },
    # Add other languages here
}


class CodeSplitter(TextSplitter):
    """
    Splits source code into chunks based on its syntactic structure.

    This strategy uses the `tree-sitter` library to parse source code into a
    concrete syntax tree. It then traverses the tree to split the code along
    syntactic boundaries, such as functions, classes, or methods. This ensures
    that chunks are syntactically complete and logically coherent.
    """

    def __init__(self, language: str, *args: Any, **kwargs: Any):
        """
        Initializes the CodeSplitter.

        Args:
            language: The programming language of the code (e.g., 'python', 'javascript').
            *args, **kwargs: Additional arguments for the base `TextSplitter`.
        """
        super().__init__(*args, **kwargs)
        if Parser is None:
            raise ImportError(
                "tree-sitter is not installed. Please install it via `pip install "
                "\"advanced-text-segmentation[code]\"` or `pip install tree-sitter tree-sitter-languages`."
            )
        try:
            self.language: Language = get_language(language)
        except Exception:
            raise ValueError(
                f"Language '{language}' is not supported or could not be loaded. "
                "Please ensure it is a valid language supported by tree-sitter-languages."
            )

        self.parser: Parser = get_parser(language)
        self._chunkable_nodes: Set[str] = LANGUAGE_QUERIES.get(language, set())
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            normalize_whitespace=self.normalize_whitespace,
            unicode_normalize=self.unicode_normalize,
        )

    def _node_to_chunks(
        self, node: Node, text_bytes: bytes, source_document_id: Optional[str]
    ) -> List[Chunk]:
        """Recursively splits a tree-sitter node into chunks."""
        node_text = text_bytes[node.start_byte : node.end_byte].decode("utf-8")

        if self.length_function(node_text) <= self.chunk_size:
            return [
                Chunk(
                    content=node_text,
                    start_index=node.start_byte,  # Note: using bytes for indices
                    end_index=node.end_byte,
                    sequence_number=0,  # Will be re-sequenced later
                    source_document_id=source_document_id,
                    chunking_strategy_used="code",
                )
            ]

        # Node is too large, try to split by its children
        child_chunks = []
        for child in node.children:
            # Only consider children that are themselves chunkable constructs
            if child.type in self._chunkable_nodes:
                child_chunks.extend(
                    self._node_to_chunks(child, text_bytes, source_document_id)
                )

        if child_chunks:
            return child_chunks

        # No suitable children to split by, use fallback splitter
        fallback_chunks = self._fallback_splitter.split_text(node_text, source_document_id)
        for chunk in fallback_chunks:
            chunk.start_index += node.start_byte  # Adjust index to be relative to the whole document
            chunk.end_index += node.start_byte
        return fallback_chunks

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the source code using its syntax tree."""
        text = self._preprocess(text)
        if not text:
            return []

        text_bytes = text.encode("utf-8")
        tree = self.parser.parse(text_bytes)
        root_node = tree.root_node

        if not root_node.children:
            return self._fallback_splitter.split_text(text, source_document_id)

        # Start by splitting the root node
        chunks = self._node_to_chunks(root_node, text_bytes, source_document_id)

        # Re-assign sequence numbers
        for i, chunk in enumerate(chunks):
            chunk.sequence_number = i

        return self._enforce_minimum_chunk_size(chunks)
