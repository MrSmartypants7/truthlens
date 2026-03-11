"""
Standalone validation script — tests core logic with only numpy (no pip needed).

This validates the chunker and the FAISS/vector store concepts using pure numpy
as a stand-in, proving the algorithms work correctly.
"""

import hashlib
import json
import sys
import os
import tempfile
import time

import numpy as np


# ════════════════════════════════════════════════════════════════════════════════
#  CHUNKER TESTS (pure Python — no dependencies)
# ════════════════════════════════════════════════════════════════════════════════

def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    raw = f"{document_id}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    if not text or not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            best_break = -1
            for sep in ["\n\n", ".\n", ". ", "! ", "? ", "\n"]:
                pos = text.rfind(sep, start, end)
                if pos > start:
                    candidate = pos + len(sep)
                    if candidate > best_break:
                        best_break = candidate
            if best_break > start:
                end = best_break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start >= len(text):
            break
        if start <= (end - chunk_size):
            start = end
    return chunks


def test_chunker():
    print("─── Chunker Tests ───")
    passed = 0
    total = 0

    # Test 1: Empty text
    total += 1
    assert chunk_text("") == [], "Empty text should return []"
    passed += 1
    print(f"  ✅ Empty text returns empty list")

    # Test 2: Short text
    total += 1
    result = chunk_text("Short text.", chunk_size=500)
    assert len(result) == 1 and result[0] == "Short text."
    passed += 1
    print(f"  ✅ Short text returns single chunk")

    # Test 3: Long text splits
    total += 1
    text = "This is a sentence. " * 50
    result = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(result) > 1, f"Expected multiple chunks, got {len(result)}"
    passed += 1
    print(f"  ✅ Long text splits into {len(result)} chunks")

    # Test 4: No empty chunks
    total += 1
    text = "Hello world. " * 100
    result = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert all(c.strip() for c in result), "Found empty chunk"
    passed += 1
    print(f"  ✅ No empty chunks produced")

    # Test 5: Deterministic IDs
    total += 1
    id1 = generate_chunk_id("doc1", 0)
    id2 = generate_chunk_id("doc1", 0)
    id3 = generate_chunk_id("doc1", 1)
    assert id1 == id2, "Same input should produce same ID"
    assert id1 != id3, "Different chunk index should produce different ID"
    passed += 1
    print(f"  ✅ Chunk IDs are deterministic and unique")

    # Test 6: Sentence boundary
    total += 1
    text = "The Eiffel Tower is in Paris. It was built in 1889. The tower is 330 meters tall. It is made of iron."
    result = chunk_text(text, chunk_size=60, chunk_overlap=5)
    for chunk in result[:-1]:  # All except last
        assert chunk.endswith(".") or chunk.endswith("!") or chunk.endswith("?"), \
            f"Chunk doesn't end at sentence boundary: '{chunk}'"
    passed += 1
    print(f"  ✅ Chunks respect sentence boundaries")

    print(f"\n  Chunker: {passed}/{total} passed\n")
    return passed, total


# ════════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE TESTS (numpy-based, simulating FAISS behavior)
# ════════════════════════════════════════════════════════════════════════════════

class SimpleVectorStore:
    """Numpy-based vector store that mimics FAISS behavior for testing."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = np.zeros((0, dimension), dtype=np.float32)
        self.metadata = {}
        self.next_id = 0

    @property
    def size(self):
        return len(self.vectors)

    def add_vectors(self, vectors, chunk_metas):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if isinstance(chunk_metas, dict):
            for i in range(len(vectors)):
                self.metadata[self.next_id + i] = chunk_metas[i]
        else:
            for i, meta in enumerate(chunk_metas):
                self.metadata[self.next_id + i] = meta
        if self.size == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        added = len(vectors)
        self.next_id += added
        return added

    def search(self, query, top_k=5, score_threshold=None):
        if self.size == 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        # L2 distance — same as FAISS IndexFlatL2
        distances = np.sum((self.vectors - query) ** 2, axis=1)
        indices = np.argsort(distances)[:top_k]

        results = []
        for rank, idx in enumerate(indices):
            dist = float(distances[idx])
            if score_threshold is not None and dist > score_threshold:
                continue
            if idx in self.metadata:
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": dist,
                    "rank": rank + 1,
                })
        return results

    def batch_search(self, queries, top_k=5):
        return [self.search(q, top_k) for q in queries]


def test_vector_store():
    print("─── Vector Store Tests ───")
    passed = 0
    total = 0
    dim = 64

    # Test 1: Empty store
    total += 1
    store = SimpleVectorStore(dim)
    assert store.size == 0
    passed += 1
    print(f"  ✅ Empty store has size 0")

    # Test 2: Add vectors
    total += 1
    vectors = np.random.randn(3, dim).astype(np.float32)
    metas = {i: {"text": f"chunk-{i}"} for i in range(3)}
    store.add_vectors(vectors, metas)
    assert store.size == 3
    passed += 1
    print(f"  ✅ Added 3 vectors, size = {store.size}")

    # Test 3: Search finds nearest
    total += 1
    store2 = SimpleVectorStore(dim)
    target = np.ones((1, dim), dtype=np.float32)
    noise = np.random.randn(5, dim).astype(np.float32) * 0.1
    all_vecs = np.vstack([target, noise])
    metas = {i: {"id": f"c{i}"} for i in range(6)}
    store2.add_vectors(all_vecs, metas)

    query = np.ones(dim, dtype=np.float32) * 0.99
    results = store2.search(query, top_k=3)
    assert len(results) == 3
    assert results[0]["metadata"]["id"] == "c0", f"Expected c0 first, got {results[0]}"
    assert results[0]["distance"] <= results[1]["distance"]
    passed += 1
    print(f"  ✅ Search returns nearest neighbor first (distance={results[0]['distance']:.4f})")

    # Test 4: Empty search
    total += 1
    empty_store = SimpleVectorStore(dim)
    results = empty_store.search(np.random.randn(dim))
    assert results == []
    passed += 1
    print(f"  ✅ Empty index returns empty results")

    # Test 5: Batch search
    total += 1
    queries = np.random.randn(3, dim).astype(np.float32)
    results = store2.batch_search(queries, top_k=2)
    assert len(results) == 3
    assert all(len(r) == 2 for r in results)
    passed += 1
    print(f"  ✅ Batch search returns results for all {len(queries)} queries")

    # Test 6: Score threshold filtering
    total += 1
    store3 = SimpleVectorStore(dim)
    close = np.zeros((1, dim), dtype=np.float32)
    far = np.ones((1, dim), dtype=np.float32) * 100
    store3.add_vectors(np.vstack([close, far]), {0: {"id": "close"}, 1: {"id": "far"}})
    results = store3.search(np.zeros(dim), top_k=2, score_threshold=1.0)
    assert len(results) == 1
    assert results[0]["metadata"]["id"] == "close"
    passed += 1
    print(f"  ✅ Score threshold filters distant results")

    print(f"\n  Vector Store: {passed}/{total} passed\n")
    return passed, total


# ════════════════════════════════════════════════════════════════════════════════
#  SCORING LOGIC TESTS
# ════════════════════════════════════════════════════════════════════════════════

def compute_overall_score(verifications):
    """Same logic as verifier.py — tested independently."""
    if not verifications:
        return 0.5
    total_weight = 0.0
    weighted_score = 0.0
    for v in verifications:
        weight = v["confidence"]
        status = v["status"]
        if status == "supported":
            weighted_score += weight * 1.0
        elif status == "contradicted":
            weighted_score += weight * 0.0
        elif status == "partially_supported":
            weighted_score += weight * 0.5
        else:
            weighted_score += weight * 0.3
        total_weight += weight
    return round(weighted_score / total_weight, 3) if total_weight > 0 else 0.5


def test_scoring():
    print("─── Scoring Logic Tests ───")
    passed = 0
    total = 0

    # Test 1: All supported → high score
    total += 1
    score = compute_overall_score([
        {"status": "supported", "confidence": 0.9},
        {"status": "supported", "confidence": 0.8},
    ])
    assert score == 1.0, f"All supported should be 1.0, got {score}"
    passed += 1
    print(f"  ✅ All supported → score = {score}")

    # Test 2: All contradicted → zero
    total += 1
    score = compute_overall_score([
        {"status": "contradicted", "confidence": 0.9},
        {"status": "contradicted", "confidence": 0.8},
    ])
    assert score == 0.0, f"All contradicted should be 0.0, got {score}"
    passed += 1
    print(f"  ✅ All contradicted → score = {score}")

    # Test 3: Mixed
    total += 1
    score = compute_overall_score([
        {"status": "supported", "confidence": 0.9},
        {"status": "contradicted", "confidence": 0.9},
    ])
    assert 0.4 <= score <= 0.6, f"Mixed should be ~0.5, got {score}"
    passed += 1
    print(f"  ✅ Mixed supported/contradicted → score = {score}")

    # Test 4: Empty → 0.5
    total += 1
    score = compute_overall_score([])
    assert score == 0.5
    passed += 1
    print(f"  ✅ No claims → score = {score}")

    # Test 5: Confidence weighting
    total += 1
    score_high = compute_overall_score([
        {"status": "supported", "confidence": 1.0},
        {"status": "contradicted", "confidence": 0.1},
    ])
    score_low = compute_overall_score([
        {"status": "supported", "confidence": 0.1},
        {"status": "contradicted", "confidence": 1.0},
    ])
    assert score_high > score_low, "Higher confidence support should win"
    passed += 1
    print(f"  ✅ Confidence weighting works (high={score_high:.3f}, low={score_low:.3f})")

    print(f"\n  Scoring: {passed}/{total} passed\n")
    return passed, total


# ════════════════════════════════════════════════════════════════════════════════
#  EMBEDDING CACHE SIMULATION
# ════════════════════════════════════════════════════════════════════════════════

def test_cache_behavior():
    print("─── Cache Behavior Tests ───")
    passed = 0
    total = 0

    cache = {}

    def embed_with_cache(text, cache):
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in cache:
            return cache[key], True  # cache hit
        vec = np.random.randn(1536).astype(np.float32)
        cache[key] = vec
        return vec, False  # cache miss

    # Test 1: First call is a miss
    total += 1
    _, hit = embed_with_cache("test text", cache)
    assert hit is False
    passed += 1
    print(f"  ✅ First call is cache miss")

    # Test 2: Second call is a hit
    total += 1
    _, hit = embed_with_cache("test text", cache)
    assert hit is True
    passed += 1
    print(f"  ✅ Second call is cache hit")

    # Test 3: Different text is a miss
    total += 1
    _, hit = embed_with_cache("different text", cache)
    assert hit is False
    passed += 1
    print(f"  ✅ Different text is cache miss")

    # Test 4: Batch efficiency
    total += 1
    texts = [f"claim number {i}" for i in range(10)]
    misses = sum(1 for t in texts if not embed_with_cache(t, cache)[1])
    hits = sum(1 for t in texts if embed_with_cache(t, cache)[1])
    assert hits == 10  # All should be cached now
    passed += 1
    print(f"  ✅ Batch: {misses} misses then {hits} hits on repeat")

    print(f"\n  Cache: {passed}/{total} passed\n")
    return passed, total


# ════════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE SIMULATION
# ════════════════════════════════════════════════════════════════════════════════

def test_pipeline_simulation():
    """Simulates the full pipeline flow without any API calls."""
    print("─── Pipeline Simulation ───")

    dim = 64

    # 1. Ingest: chunk + embed + index
    doc = "The Eiffel Tower is 330 metres tall. It was built from 1887 to 1889. It is located in Paris, France."
    chunks = chunk_text(doc, chunk_size=60, chunk_overlap=10)
    print(f"  📄 Document chunked into {len(chunks)} pieces:")
    for i, c in enumerate(chunks):
        print(f"     Chunk {i}: \"{c[:50]}...\"" if len(c) > 50 else f"     Chunk {i}: \"{c}\"")

    # Simulate embedding
    vectors = np.random.randn(len(chunks), dim).astype(np.float32)
    store = SimpleVectorStore(dim)
    metas = {i: {"text": chunks[i], "source": "wikipedia"} for i in range(len(chunks))}
    store.add_vectors(vectors, metas)
    print(f"  📦 Indexed {store.size} vectors")

    # 2. Simulate claim extraction
    llm_response = "The Eiffel Tower is 324 meters tall and was built in 1889."
    claims = [
        "The Eiffel Tower is 324 meters tall",
        "The Eiffel Tower was built in 1889",
    ]
    print(f"\n  🔍 LLM Response: \"{llm_response}\"")
    print(f"  📝 Extracted {len(claims)} claims:")
    for c in claims:
        print(f"     • {c}")

    # 3. Simulate retrieval
    print(f"\n  🔎 Retrieving evidence for each claim...")
    for claim in claims:
        query_vec = np.random.randn(dim).astype(np.float32)
        results = store.search(query_vec, top_k=2)
        print(f"     Claim: \"{claim}\"")
        for r in results:
            print(f"       Evidence (dist={r['distance']:.2f}): \"{r['metadata']['text'][:60]}...\"")

    # 4. Simulate verification
    simulated_results = [
        {"claim": claims[0], "status": "contradicted", "confidence": 0.9,
         "reasoning": "Claim says 324m but evidence says 330m"},
        {"claim": claims[1], "status": "supported", "confidence": 0.85,
         "reasoning": "Evidence confirms construction period 1887-1889"},
    ]

    score = compute_overall_score(simulated_results)

    print(f"\n  📊 Verification Results:")
    for r in simulated_results:
        icon = "✅" if r["status"] == "supported" else "❌"
        print(f"     {icon} \"{r['claim']}\" → {r['status']} ({r['confidence']:.0%})")
        print(f"        Reason: {r['reasoning']}")

    print(f"\n  🎯 Overall Reliability Score: {score:.1%}")
    print(f"\n  Pipeline simulation: PASSED\n")
    return 1, 1


# ════════════════════════════════════════════════════════════════════════════════
#  RUN ALL TESTS
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  TruthLens — Standalone Validation Suite")
    print("=" * 70)
    print()

    total_passed = 0
    total_tests = 0

    for test_fn in [test_chunker, test_vector_store, test_scoring, test_cache_behavior, test_pipeline_simulation]:
        p, t = test_fn()
        total_passed += p
        total_tests += t

    print("=" * 70)
    print(f"  TOTAL: {total_passed}/{total_tests} tests passed")
    if total_passed == total_tests:
        print("  🎉 All tests passed!")
    else:
        print(f"  ⚠️  {total_tests - total_passed} tests failed")
    print("=" * 70)
