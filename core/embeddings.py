"""
Embedding service — converts text into vectors using Ollama (free, local).

CHANGED FROM OPENAI:
  - Uses Ollama's /api/embed endpoint instead of OpenAI's embeddings API
  - Model: nomic-embed-text (768 dimensions, runs locally)
  - No API key needed, no cost per token
  - Same caching and batching logic applies

The nomic-embed-text model is a high-quality open-source embedding model.
It produces 768-dimensional vectors (vs OpenAI's 1536), but works very
well for retrieval tasks.
"""

import hashlib

import numpy as np
import ollama
from cachetools import TTLCache

from app.config import settings
from app.logging_config import get_logger

logger = get_logger("embeddings")


class EmbeddingService:
    """Manages text → vector conversion with caching and batching."""

    def __init__(self):
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension

        # Create Ollama client pointing to local server
        self.client = ollama.Client(host=settings.ollama_base_url)

        # TTL cache: embeddings expire after cache_ttl_seconds
        # Key = hash of text, Value = numpy vector
        self._cache: TTLCache = TTLCache(
            maxsize=settings.cache_max_size,
            ttl=settings.cache_ttl_seconds,
        )

    def _cache_key(self, text: str) -> str:
        """Hash the text for use as a cache key. Keeps memory usage constant."""
        return hashlib.sha256(text.encode()).hexdigest()

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text string. Checks cache first.

        Returns:
            numpy array of shape (dimension,) — e.g., (768,)
        """
        key = self._cache_key(text)

        # Check cache
        if key in self._cache:
            logger.debug("cache_hit", text_length=len(text))
            return self._cache[key]

        # Call Ollama's embed endpoint
        response = self.client.embed(
            model=self.model,
            input=text,
        )

        vector = np.array(response["embeddings"][0], dtype=np.float32)
        self._cache[key] = vector

        logger.debug("embedded_single", text_length=len(text), dimension=len(vector))
        return vector

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed multiple texts. Ollama's embed endpoint supports batch input natively.

        WHY BATCH: Even locally, there's overhead per call (model loading, memory
        allocation). Batching N texts into 1 call is faster than N separate calls.

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        # Separate cached vs uncached texts
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch-embed only the uncached ones via Ollama
        if uncached_texts:
            response = self.client.embed(
                model=self.model,
                input=uncached_texts,
            )

            for j, embedding in enumerate(response["embeddings"]):
                vector = np.array(embedding, dtype=np.float32)
                original_index = uncached_indices[j]
                results[original_index] = vector

                # Store in cache
                key = self._cache_key(uncached_texts[j])
                self._cache[key] = vector

        logger.info(
            "batch_embedded",
            total=len(texts),
            cached=len(texts) - len(uncached_texts),
            api_calls=1 if uncached_texts else 0,
        )

        return np.vstack(results)

    def get_cache_stats(self) -> dict:
        """Return cache utilization stats for monitoring."""
        return {
            "current_size": len(self._cache),
            "max_size": self._cache.maxsize,
            "ttl_seconds": self._cache.ttl,
            "utilization_pct": round(len(self._cache) / self._cache.maxsize * 100, 1),
        }
