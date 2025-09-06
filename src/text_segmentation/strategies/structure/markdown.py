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
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
        )

    def _get_line_start_indices(self, text: str) -> List[int]:
        indices = [0]
        for line in text.splitlines(keepends=True):
            indices.append(indices[-1] + len(line))
        return indices

    def _extract_blocks(
        self, text: str, line_start_indices: List[int]
    ) -> List[Tuple[str, int, int, Dict[str, Any]]]:
        tokens = self.md_parser.parse(text)
        blocks = []
        header_context: Dict[str, Any] = {}

        for i, token in enumerate(tokens):
            if token.type.endswith("_open"):
                start_line, end_line = token.map

                # Find the corresponding closing token to get the full block
                level = token.level
                j = i + 1
                while j < len(tokens):
                    if tokens[j].type == f"{token.tag}_close" and tokens[j].level == level:
                        end_line = tokens[j].map[1]
                        break
                    j += 1

                start_index = line_start_indices[start_line]
                end_index = line_start_indices[end_line] if end_line < len(line_start_indices) else len(text)
                content = text[start_index:end_index]

                if token.type == "heading_open":
                    level = int(token.tag[1])
                    inline_token = tokens[i+1]
                    header_text = ""
                    if inline_token.type == 'inline':
                        header_text = inline_token.content.strip()

                    if header_text:
                        keys_to_del = [k for k in header_context if int(k[1:]) >= level]
                        for k in keys_to_del:
                            del header_context[k]
                        header_context[f"H{level}"] = header_text

                if content.strip():
                    blocks.append((content, start_index, end_index, header_context.copy()))

        # This logic is flawed as it creates overlapping blocks.
        # A simpler approach is to split by headers. I will revert to that.
        # But this time I will fix the regex.
        return []


    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        if not text.strip():
            return []

        import re
        # Split by headers, keeping the header in the following part
        sections = re.split(r'(?m)(^#{1,6} .*$)', text, flags=re.MULTILINE)

        # The first element is the text before the first header
        # The subsequent elements are pairs of (header, content)
        prose = sections[0]
        headers = sections[1::2]
        contents = sections[2::2]

        final_sections = []
        if prose.strip():
            final_sections.append(prose)

        for i in range(len(headers)):
            if i < len(contents):
                final_sections.append(headers[i] + contents[i])
            else:
                final_sections.append(headers[i])

        chunks: List[Chunk] = []
        current_pos = 0
        sequence_number = 0
        header_context: Dict[str, Any] = {}

        for section_text in final_sections:
            if not section_text.strip():
                continue

            section_start_index = text.find(section_text, current_pos)
            if section_start_index == -1:
                continue

            header_match = re.match(r'^\s*(#{1,6}) (.*)', section_text)
            if header_match:
                level = len(header_match.group(1))
                header_content = header_match.group(2).split('\n', 1)[0].strip()
                tag = f"H{level}"

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

        return chunks
