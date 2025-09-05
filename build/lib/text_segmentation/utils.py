from typing import Callable, List, Optional

from text_segmentation.core import Chunk


def merge_chunks(
    base_units: List[Chunk],
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int],
    minimum_chunk_size: int = 50,
) -> List[Chunk]:
    """
    Merges a list of base units (e.g., sentences, code blocks) into larger chunks.

    This utility handles chunk sizing, overlap, and runt merging, ensuring
    all FRD requirements for the final output are met.

    Args:
        base_units: A list of preliminary Chunk objects to be merged.
        chunk_size: The target maximum size of a chunk.
        chunk_overlap: The target overlap between consecutive chunks.
        length_function: The function to measure text length.
        minimum_chunk_size: The threshold below which a chunk is considered a "runt".

    Returns:
        A list of final, merged Chunk objects.
    """
    if not base_units:
        return []

    final_chunks: List[Chunk] = []
    current_chunk_units: List[Chunk] = []
    current_length = 0

    # R-4.3.2: Runt handling
    # To avoid creating a very small final chunk, we merge it with the previous one
    # if it's smaller than the minimum size.

    # Pass 1: Group base units into chunks
    for i, unit in enumerate(base_units):
        unit_length = length_function(unit.content)

        # If adding the next unit exceeds the chunk size, finalize the current chunk
        if current_length > 0 and current_length + unit_length > chunk_size:
            # Finalize the current chunk
            content = "".join(u.content for u in current_chunk_units)
            start_index = current_chunk_units[0].start_index
            end_index = current_chunk_units[-1].end_index

            final_chunks.append(
                Chunk(
                    content=content,
                    start_index=start_index,
                    end_index=end_index,
                    sequence_number=len(final_chunks),
                )
            )

            # Start a new chunk, calculating overlap
            overlap_units: List[Chunk] = []
            overlap_len = 0
            for j in range(len(current_chunk_units) - 1, -1, -1):
                unit_to_add = current_chunk_units[j]
                if overlap_len + length_function(unit_to_add.content) > chunk_overlap:
                    break
                overlap_units.insert(0, unit_to_add)
                overlap_len += length_function(unit_to_add.content)

            current_chunk_units = overlap_units
            current_length = overlap_len

        current_chunk_units.append(unit)
        current_length += unit_length

    # Add the last remaining chunk
    if current_chunk_units:
        content = "".join(u.content for u in current_chunk_units)
        start_index = current_chunk_units[0].start_index
        end_index = current_chunk_units[-1].end_index

        # R-4.3.2 Runt Handling: Check if the last chunk is a runt
        if final_chunks and length_function(content) < minimum_chunk_size:
            # Merge with the previous chunk if it doesn't exceed the size limit
            prev_chunk = final_chunks[-1]
            new_content = prev_chunk.content + content
            if length_function(new_content) <= chunk_size:
                final_chunks[-1].content = new_content
                final_chunks[-1].end_index = end_index
            else:
                 final_chunks.append(
                    Chunk(
                        content=content,
                        start_index=start_index,
                        end_index=end_index,
                        sequence_number=len(final_chunks),
                    )
                )
        else:
            final_chunks.append(
                Chunk(
                    content=content,
                    start_index=start_index,
                    end_index=end_index,
                    sequence_number=len(final_chunks),
                )
            )

    # Pass 2: Populate overlap metadata (R-5.2.7, R-5.2.8)
    for i in range(len(final_chunks) - 1):
        current_chunk = final_chunks[i]
        next_chunk = final_chunks[i+1]

        # Find the start of the overlap
        overlap_start = next_chunk.start_index
        if current_chunk.end_index > overlap_start:
            overlap_text = current_chunk.content[overlap_start - current_chunk.start_index:]
            current_chunk.overlap_content_next = overlap_text
            next_chunk.overlap_content_previous = overlap_text

    return final_chunks
