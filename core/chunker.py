"""
Text chunking — splits documents into smaller pieces for FAISS indexing.

WHY CHUNK: A 10-page document about France covers hundreds of topics.
If we embed the whole thing, the vector becomes a blurry average of all topics.
By splitting into ~500 character pieces, each vector represents ONE specific idea,
making FAISS retrieval much more precise.

WHY OVERLAP: If a fact spans two chunks (e.g., "The tower is 330 meters" split between
chunks), the overlap ensures both chunks contain the full fact.
"""

import hashlib
from app.config import settings
from models.schemas import DocumentChunk


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a deterministic unique ID for a chunk.

    WHY DETERMINISTIC: If we re-ingest the same document, we get the same chunk IDs.
    This lets us detect duplicates instead of creating redundant FAISS entries.
    """
    raw = f"{document_id}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[str]:
    """
    Split text into overlapping chunks by character count.

    Algorithm:
        1. Start at position 0
        2. Take `chunk_size` characters
        3. Find the last sentence boundary (period, newline) within that window
           so we don't cut mid-sentence
        4. Move forward by (chunk_size - overlap) characters
        5. Repeat until we've consumed the whole text

    Args:
        text: The document text to chunk
        chunk_size: Max characters per chunk (default from settings)
        chunk_overlap: Overlap between consecutive chunks (default from settings)

    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []

    # If text fits in one chunk, just return it
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0

    while start < len(text):
        # Take a window of chunk_size characters
        end = start + chunk_size

        # If we're not at the very end, try to break at a sentence boundary
        if end < len(text):
            # Look for the last sentence-ending punctuation in the window
            # Search backwards from `end` to find a clean break point
            best_break = -1
            for sep in ["\n\n", ".\n", ". ", "! ", "? ", "\n"]:
                pos = text.rfind(sep, start, end)
                if pos > start:
                    # +len(sep) so we include the separator in the current chunk
                    candidate = pos + len(sep)
                    if candidate > best_break:
                        best_break = candidate

            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, but step back by `overlap` to create overlap
        start = end - chunk_overlap
        if start >= len(text):
            break
        # Safety: ensure we always move forward
        if start <= (end - chunk_size):
            start = end

    return chunks


def create_chunks_from_document(
    text: str,
    document_id: str,
    source: str = "unknown",
    metadata: dict = None,
) -> list[DocumentChunk]:
    """
    Full pipeline: text → chunks → DocumentChunk objects with IDs.

    This is what the ingestion endpoint calls. Each chunk gets:
    - A deterministic ID (for dedup)
    - A reference back to its parent document
    - The source and any metadata
    """
    raw_chunks = chunk_text(text)
    doc_chunks = []

    for i, chunk_text_content in enumerate(raw_chunks):
        chunk = DocumentChunk(
            chunk_id=generate_chunk_id(document_id, i),
            document_id=document_id,
            text=chunk_text_content,
            source=source,
            metadata=metadata or {},
        )
        doc_chunks.append(chunk)

    return doc_chunks
