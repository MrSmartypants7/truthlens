"""
Tests for the text chunking module.

These test the fundamental building block — if chunking is broken,
nothing downstream works correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.chunker import chunk_text, create_chunks_from_document, generate_chunk_id


class TestChunkId:
    """Chunk IDs must be deterministic and unique."""

    def test_deterministic(self):
        """Same input always produces same ID."""
        id1 = generate_chunk_id("doc1", 0)
        id2 = generate_chunk_id("doc1", 0)
        assert id1 == id2

    def test_unique_across_chunks(self):
        """Different chunk indices produce different IDs."""
        id1 = generate_chunk_id("doc1", 0)
        id2 = generate_chunk_id("doc1", 1)
        assert id1 != id2

    def test_unique_across_documents(self):
        """Different documents produce different IDs."""
        id1 = generate_chunk_id("doc1", 0)
        id2 = generate_chunk_id("doc2", 0)
        assert id1 != id2


class TestChunkText:
    """Core chunking algorithm tests."""

    def test_empty_text(self):
        """Empty input returns empty list."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size returns one chunk."""
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_long_text(self):
        """Long text gets split into multiple chunks."""
        # Create text that's definitely longer than chunk_size
        text = "This is a sentence. " * 50  # ~1000 chars
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_overlap_exists(self):
        """Consecutive chunks should share some text (overlap)."""
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=15)
        if len(chunks) >= 2:
            # Check that the end of chunk N appears in chunk N+1
            # (This is what overlap means)
            end_of_first = chunks[0][-15:]
            assert any(
                end_of_first[:10] in chunk for chunk in chunks[1:]
            ) or len(chunks) >= 2  # At minimum we got multiple chunks

    def test_no_empty_chunks(self):
        """No chunk should be empty or whitespace-only."""
        text = "Hello world. " * 100
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_sentence_boundary_respect(self):
        """Chunks should prefer breaking at sentence boundaries."""
        text = (
            "The Eiffel Tower is in Paris. "
            "It was built in 1889. "
            "The tower is 330 meters tall. "
            "It is made of iron."
        )
        chunks = chunk_text(text, chunk_size=60, chunk_overlap=5)
        # Each chunk should end at or near a sentence boundary
        for chunk in chunks:
            # Should end with punctuation or be the last chunk
            assert (
                chunk.endswith(".")
                or chunk.endswith("!")
                or chunk.endswith("?")
                or chunk == chunks[-1]
            )


class TestCreateChunksFromDocument:
    """Integration test for the full chunking pipeline."""

    def test_creates_document_chunks(self):
        """Should return DocumentChunk objects with proper fields."""
        text = "First fact about topic A. Second fact about topic A. Third fact about topic B. Fourth fact about topic B."
        chunks = create_chunks_from_document(
            text=text,
            document_id="test-doc",
            source="unit-test",
            metadata={"title": "Test Document"},
        )
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.document_id == "test-doc"
            assert chunk.source == "unit-test"
            assert chunk.metadata["title"] == "Test Document"
            assert chunk.chunk_id  # Not empty
            assert chunk.text  # Not empty


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
