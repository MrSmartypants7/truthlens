"""
Tests for the FAISS vector store.

Tests that vectors can be added, searched, saved, and loaded correctly.
Uses small random vectors (no OpenAI calls needed).
"""

import sys
import os
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.vector_store import VectorStore
from models.schemas import DocumentChunk


def make_chunk(chunk_id: str, text: str = "test") -> DocumentChunk:
    """Helper to create a DocumentChunk for testing."""
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        source="test",
    )


class TestVectorStore:
    """FAISS vector store tests."""

    def setup_method(self):
        """Create a fresh store before each test."""
        self.dim = 64  # Small dimension for fast tests
        self.store = VectorStore(dimension=self.dim)

    def test_empty_store(self):
        assert self.store.size == 0

    def test_add_vectors(self):
        """Adding vectors increases the index size."""
        vectors = np.random.randn(3, self.dim).astype(np.float32)
        chunks = [make_chunk(f"c{i}") for i in range(3)]
        added = self.store.add_vectors(vectors, chunks)
        assert added == 3
        assert self.store.size == 3

    def test_search_returns_results(self):
        """Search should return the most similar vectors."""
        # Add a known vector
        target = np.ones((1, self.dim), dtype=np.float32)
        noise = np.random.randn(5, self.dim).astype(np.float32) * 0.1

        vectors = np.vstack([target, noise])
        chunks = [make_chunk(f"c{i}", f"text-{i}") for i in range(6)]
        self.store.add_vectors(vectors, chunks)

        # Search with a vector very close to the target
        query = np.ones(self.dim, dtype=np.float32) * 0.99
        results = self.store.search(query, top_k=3)

        assert len(results) == 3
        # First result should be the target (closest match)
        assert results[0].chunk.chunk_id == "c0"
        assert results[0].rank == 1
        # Scores should be in ascending order (lower = more similar)
        assert results[0].similarity_score <= results[1].similarity_score

    def test_search_empty_index(self):
        """Searching an empty index returns empty list."""
        query = np.random.randn(self.dim).astype(np.float32)
        results = self.store.search(query)
        assert results == []

    def test_batch_search(self):
        """Batch search returns results for each query."""
        vectors = np.random.randn(10, self.dim).astype(np.float32)
        chunks = [make_chunk(f"c{i}") for i in range(10)]
        self.store.add_vectors(vectors, chunks)

        queries = np.random.randn(3, self.dim).astype(np.float32)
        results = self.store.batch_search(queries, top_k=2)

        assert len(results) == 3
        for r in results:
            assert len(r) == 2

    def test_save_and_load(self):
        """Index should survive save/load cycle."""
        vectors = np.random.randn(5, self.dim).astype(np.float32)
        chunks = [make_chunk(f"c{i}", f"text-{i}") for i in range(5)]
        self.store.add_vectors(vectors, chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = f"{tmpdir}/test.bin"
            meta_path = f"{tmpdir}/test.json"

            self.store.save(idx_path, meta_path)

            # Create a new store and load
            new_store = VectorStore(dimension=self.dim)
            loaded = new_store.load(idx_path, meta_path)

            assert loaded is True
            assert new_store.size == 5

            # Search should still work
            query = vectors[0]
            results = new_store.search(query, top_k=1)
            assert len(results) == 1
            assert results[0].chunk.chunk_id == "c0"

    def test_clear(self):
        """Clear should reset the index."""
        vectors = np.random.randn(5, self.dim).astype(np.float32)
        chunks = [make_chunk(f"c{i}") for i in range(5)]
        self.store.add_vectors(vectors, chunks)
        assert self.store.size == 5

        self.store.clear()
        assert self.store.size == 0

    def test_score_threshold(self):
        """Score threshold should filter out distant results."""
        # One very close vector, one very far vector
        close = np.zeros((1, self.dim), dtype=np.float32)
        far = np.ones((1, self.dim), dtype=np.float32) * 100

        vectors = np.vstack([close, far])
        chunks = [make_chunk("close"), make_chunk("far")]
        self.store.add_vectors(vectors, chunks)

        query = np.zeros(self.dim, dtype=np.float32)
        results = self.store.search(query, top_k=2, score_threshold=1.0)

        # Only the close vector should pass the threshold
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "close"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
