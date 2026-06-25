"""Semantic chunking for academic document RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict


def _split_into_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def semantic_chunk(
    text: str,
    source: str,
    max_chunk_size: int = 600,
    overlap: int = 80,
) -> list[DocumentChunk]:
    """
    Chunk text semantically by paragraph boundaries with overlap.
    Keeps related sentences together while respecting size limits.
    """
    paragraphs = _split_into_paragraphs(text)
    chunks: list[DocumentChunk] = []
    current_parts: list[str] = []
    current_len = 0
    chunk_index = 0

    for paragraph in paragraphs:
        para_len = len(paragraph)
        if current_len + para_len > max_chunk_size and current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_index,
                    metadata={"type": "semantic", "char_count": len(chunk_text)},
                )
            )
            chunk_index += 1
            # Overlap: keep trailing content
            overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
            current_parts = [overlap_text, paragraph] if overlap_text else [paragraph]
            current_len = sum(len(p) for p in current_parts)
        else:
            current_parts.append(paragraph)
            current_len += para_len

    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        chunks.append(
            DocumentChunk(
                text=chunk_text,
                source=source,
                chunk_index=chunk_index,
                metadata={"type": "semantic", "char_count": len(chunk_text)},
            )
        )

    return chunks
