"""
FastAPI Application — the API layer for TruthLens.

CHANGED FROM OPENAI: Pipeline no longer needs an API key.
Everything else is identical.
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.logging_config import get_logger, setup_logging
from core.experiment_tracker import ExperimentTracker
from core.pipeline import TruthLensPipeline
from models.schemas import (
    IngestRequest,
    IngestResponse,
    VerifyRequest,
    VerifyResponse,
)

# Module-level references (set during lifespan)
pipeline: Optional[TruthLensPipeline] = None
tracker: Optional[ExperimentTracker] = None
logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — init on startup, cleanup on shutdown."""
    global pipeline, tracker, logger

    setup_logging()
    logger = get_logger("api")

    settings.ensure_dirs()
    pipeline = TruthLensPipeline()
    tracker = ExperimentTracker()

    logger.info(
        "server_started",
        host=settings.host,
        port=settings.port,
        index_size=pipeline.vector_store.size,
    )

    yield

    logger.info("server_shutting_down")


app = FastAPI(
    title="TruthLens",
    description="LLM Reliability & Hallucination Detection Engine (powered by Ollama)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing."""
    start_time = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start_time) * 1000

    if logger:
        logger.info(
            "request_processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            elapsed_ms=round(elapsed, 1),
        )

    response.headers["X-Process-Time-Ms"] = str(round(elapsed, 1))
    return response


@app.get("/health")
async def health_check():
    """Health check for load balancers and monitoring."""
    return {
        "status": "healthy",
        "index_size": pipeline.vector_store.size if pipeline else 0,
        "version": "1.0.0",
        "backend": "ollama",
    }


@app.post("/verify", response_model=VerifyResponse)
async def verify_response(request: VerifyRequest):
    """Verify an LLM-generated response for hallucinations."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.verify(request)
        return result
    except Exception as e:
        logger.error("verification_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Add documents to the knowledge base (FAISS index)."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.ingest_documents(request.documents)
        return result
    except Exception as e:
        logger.error("ingestion_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Return system statistics."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return {
        "index": {
            "num_vectors": pipeline.vector_store.size,
            "dimension": pipeline.vector_store.dimension,
        },
        "cache": pipeline.embedding_service.get_cache_stats(),
        "config": {
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "chunk_size": settings.chunk_size,
            "retrieval_top_k": settings.retrieval_top_k,
            "backend": "ollama",
        },
    }


@app.get("/experiments")
async def list_experiments():
    """List all benchmark experiment results."""
    if not tracker:
        raise HTTPException(status_code=503, detail="Tracker not initialized")
    return tracker.list_experiments()


@app.post("/benchmark")
async def run_benchmark(test_data: dict):
    """Run a benchmark evaluation."""
    if not pipeline or not tracker:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        model_name = test_data.get("model_name", "unknown")
        test_cases = test_data.get("test_cases", [])

        all_verifications = []
        all_ground_truth = []

        for case in test_cases:
            request = VerifyRequest(
                llm_response=case["llm_response"],
                model_name=model_name,
            )
            result = pipeline.verify(request)
            all_verifications.extend(result.claim_verifications)
            all_ground_truth.extend(case.get("expected_claims", []))

        benchmark = tracker.evaluate_verifications(
            verifications=all_verifications,
            ground_truth=all_ground_truth,
            model_name=model_name,
        )

        return benchmark.model_dump(mode="json")

    except Exception as e:
        logger.error("benchmark_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


# ── Serve Frontend ────────────────────────────────────────────────────────────
# Mount static files and serve index.html at root
import os

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the TruthLens web UI."""
        return FileResponse(os.path.join(static_dir, "index.html"))

