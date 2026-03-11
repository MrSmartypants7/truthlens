"""
Configuration management using Pydantic Settings.

WHY: Centralizes all config in one place. Pydantic validates types automatically,
so if someone sets EMBEDDING_DIMENSION="not_a_number", it fails fast with a clear error
instead of crashing deep in FAISS code.

CHANGED FROM OPENAI: Now uses Ollama (free, local). The Ollama server runs on
localhost:11434 by default. No API key needed.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """All application settings, loaded from environment variables or .env file."""

    # --- Ollama ---
    ollama_base_url: str = Field(default="http://localhost:11434")

    # --- Embedding (nomic-embed-text produces 768-dim vectors) ---
    embedding_model: str = Field(default="nomic-embed-text")
    embedding_dimension: int = Field(default=768)

    # --- LLM (for claim extraction + verification) ---
    llm_model: str = Field(default="llama3.2")
    llm_temperature: float = Field(default=0.0)

    # --- FAISS ---
    faiss_index_path: str = Field(default="data/faiss_index.bin")
    faiss_metadata_path: str = Field(default="data/faiss_metadata.json")
    faiss_nprobe: int = Field(default=10)
    retrieval_top_k: int = Field(default=5)

    # --- Chunking ---
    chunk_size: int = Field(default=200, description="Max characters per chunk")
    chunk_overlap: int = Field(default=30, description="Overlap between chunks")

    # --- Cache ---
    cache_ttl_seconds: int = Field(default=3600)
    cache_max_size: int = Field(default=1024)

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # --- Experiment tracking ---
    experiment_dir: str = Field(default="data/experiments")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        """Create necessary directories if they don't exist."""
        Path(self.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)


# Singleton instance — import this everywhere
settings = Settings()
