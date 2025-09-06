from typing import Any, Dict, List, Optional, Tuple

from text_segmentation.base import TextSplitter
from text_segmentation.core import Chunk
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter

from markdown_it import MarkdownIt
from markdown_it.token import Token


class MarkdownSplitter(TextSplitter):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.md_parser = MarkdownIt()
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0,  # No overlap in fallback since we chunk section by section
            length_function=self.length_function,
            normalize_whitespace=self.normalize_whitespace,
            unicode_normalize=self.unicode_normalize,
        )

    def _get_line_start_indices(self, text: str) -> List[int]:
        indices = [0]
        for line in text.splitlines(keepends=True):
            indices.append(indices[-1] + len(line))
        return indices

    def _extract_blocks(
        self, text: str, line_start_indices: List[int]
    ) -> List[Tuple[str, int, int, Dict[str, Any]]]:
        # This method is currently unused and marked as flawed.
        # The regex-based approach in split_text is used instead.
        return []

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        text = self._preprocess(text)
        if not text.strip():
            return []

        import re
        # Split by headers, keeping the header in the following part.
        # This regex looks for lines starting with 1-6 '#' characters.
        sections = re.split(r'(?m)(^#{1,6} .*)', text)

        # The first element is the text before the first header.
        # Subsequent elements are pairs of (header, content).
        final_sections = []
        # Add the content before the first header if it exists
        if sections[0].strip():
            final_sections.append(("", sections[0]))

        # Group headers with their content
        for i in range(1, len(sections), 2):
            header = sections[i]
            content = sections[i+1] if (i+1) < len(sections) else ""
            final_sections.append((header, header + content))

        chunks: List[Chunk] = []
        current_pos = 0
        sequence_number = 0
        header_context: Dict[str, Any] = {}

        for header, section_text in final_sections:
            if not section_text.strip():
                continue

            section_start_index = text.find(section_text, current_pos)
            if section_start_index == -1:
                # This can happen if preprocessing alters the text in a way
                # that makes find() fail. A more robust search may be needed,
                # but for now, we'll proceed.
                current_pos += len(section_text)
                continue

            if header.strip():
                level = len(header.split(" ")[0])
                header_content = header.lstrip("# ").strip()
                tag = f"H{level}"

                # Clear context from the same or lower level headers
                keys_to_del = [k for k in header_context if int(k[1:]) >= level]
                for k in keys_to_del:
                    del header_context[k]
                header_context[tag] = header_content

            # Now we have clean sections with context. Chunk them.
            if self.length_function(section_text) > self.chunk_size:
                fallback_chunks = self._fallback_splitter.split_text(section_text)
                for fb_chunk in fallback_chunks:
                    chunk_start_index = section_start_index + fb_chunk.start_index
                    chunks.append(Chunk(
                        content=fb_chunk.content,
                        start_index=chunk_start_index,
                        end_index=chunk_start_index + len(fb_chunk.content),
                        sequence_number=sequence_number,
                        source_document_id=source_document_id,
                        hierarchical_context=header_context.copy(),
                        chunking_strategy_used="markdown-fallback"
                    ))
                    sequence_number += 1
            else:
                chunks.append(Chunk(
                    content=section_text,
                    start_index=section_start_index,
                    end_index=section_start_index + len(section_text),
                    sequence_number=sequence_number,
                    source_document_id=source_document_id,
                    hierarchical_context=header_context.copy(),
                    chunking_strategy_used="markdown"
                ))
                sequence_number += 1

            current_pos = section_start_index + len(section_text)

        return self._enforce_minimum_chunk_size(chunks)
