from typing import Any, Dict, List, Optional, Set, Tuple

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.utils import merge_chunks

try:
    from tree_sitter import Language, Node, Parser
    from tree_sitter_languages import get_language, get_parser
except ImportError:
    Parser = None  # type: ignore

# Language-specific queries to find top-level chunkable nodes.
# This can be expanded for more languages and more granular control.
LANGUAGE_QUERIES: Dict[str, Dict[str, str]] = {
    "python": {
        "function_definition": "name",
        "class_definition": "name",
        "decorated_definition": "definition",
    },
    "javascript": {
        "function_declaration": "name",
        "class_declaration": "name",
        "lexical_declaration": "name",
        "export_statement": "declaration",
    },
    "go": {
        "function_declaration": "name",
        "method_declaration": "name",
        "type_declaration": "name",
    },
    "rust": {
        "function_item": "name",
        "struct_item": "name",
        "enum_item": "name",
        "impl_item": "type",
        "trait_item": "name",
    },
}

# Node types that represent a named entity we can use for context
CONTEXT_NODE_TYPES: Dict[str, Set[str]] = {
    "python": {"class_definition", "function_definition"},
    "javascript": {"class_declaration", "function_declaration"},
    # Add more for other languages
}


class CodeSplitter(TextSplitter):
    """
    Splits source code into chunks based on its syntactic structure.

    This strategy uses the `tree-sitter` library to parse source code into a
    concrete syntax tree. It then traverses the tree to identify logical code
    units (like functions or classes), which are then merged into chunks of a
    specified size. This ensures that chunks are syntactically meaningful.
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
        self._chunkable_node_types: Set[str] = set(LANGUAGE_QUERIES.get(language, {}).keys())

    def _get_node_name(self, node: Node, text_bytes: bytes) -> Optional[str]:
        """Extracts the name from a node (e.g., function or class name)."""
        name_child_field = LANGUAGE_QUERIES.get(self.parser.language.name, {}).get(node.type)
        if not name_child_field:
            return None

        name_node = node.child_by_field_name(name_child_field)
        if name_node:
            return text_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
        return None

    def _get_base_units(self, text: str) -> List[Chunk]:
        """
        Extracts high-level syntactic units (functions, classes, etc.) as base chunks.
        """
        text_bytes = text.encode("utf-8")
        tree = self.parser.parse(text_bytes)
        root_node = tree.root_node

        base_units: List[Chunk] = []

        # UTF-8 byte to character index mapping
        byte_to_char = [len(text_bytes[:i].decode('utf-8', errors='ignore')) for i in range(len(text_bytes) + 1)]

        def traverse(node: Node, context_stack: List[Tuple[str, str]]):
            # Check if the current node is a chunkable unit
            if node.type in self._chunkable_node_types:
                node_text = text_bytes[node.start_byte : node.end_byte].decode("utf-8")

                # Critical Bug Fix (R-5.2.3, R-5.2.4): Convert byte indices to character indices
                start_char = byte_to_char[node.start_byte]
                end_char = byte_to_char[node.end_byte]

                # Hierarchical Context (R-5.2.6)
                hierarchical_context = dict(context_stack)

                base_units.append(
                    Chunk(
                        content=node_text,
                        start_index=start_char,
                        end_index=end_char,
                        sequence_number=-1,  # Placeholder
                        hierarchical_context=hierarchical_context,
                    )
                )
                # After processing a whole unit, we don't need to process its children further
                return

            # Update context and recurse
            new_context = list(context_stack)
            if node.type in CONTEXT_NODE_TYPES.get(self.parser.language.name, set()):
                name = self._get_node_name(node, text_bytes)
                if name:
                    # e.g., ('class', 'MyClassName')
                    new_context.append((node.type.replace("_definition", ""), name))

            for child in node.children:
                traverse(child, new_context)

        traverse(root_node, [])

        # Sort units by their appearance in the document
        base_units.sort(key=lambda c: c.start_index)
        return base_units

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the source code by merging syntactic units into sized chunks."""
        base_units = self._get_base_units(text)

        if not base_units:
            # Fallback for code that can't be parsed into units
            fallback_splitter = RecursiveCharacterSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=self.length_function,
            )
            return fallback_splitter.split_text(text, source_document_id)

        # Use the utility to merge base units into final chunks
        final_chunks = merge_chunks(
            base_units=base_units,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            minimum_chunk_size=self.chunk_size // 4, # Sensible default for min code chunk
        )

        # Final pass to add source_document_id and strategy name
        for i, chunk in enumerate(final_chunks):
            chunk.sequence_number = i
            chunk.source_document_id = source_document_id
            chunk.chunking_strategy_used = f"code_{self.parser.language.name}"

        return final_chunks
