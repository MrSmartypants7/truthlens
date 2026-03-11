"""
TruthLens Pipeline — orchestrates the full verification flow.

CHANGED FROM OPENAI: Removed all api_key parameters since Ollama is local.
Everything else is identical — same batching, same flow, same performance design.
"""

import time
import uuid

import numpy as np

from app.config import settings
from app.logging_config import get_logger
from core.chunker import create_chunks_from_document
from core.claim_extractor import ClaimExtractor
from core.embeddings import EmbeddingService
from core.vector_store import VectorStore
from core.verifier import VerificationEngine
from models.schemas import (
    ClaimVerification,
    IngestResponse,
    VerifyRequest,
    VerifyResponse,
)

logger = get_logger("pipeline")


class TruthLensPipeline:
    """
    Main pipeline that orchestrates document ingestion and claim verification.

    Lifecycle:
        1. __init__: Create all components
        2. ingest_documents: Add ground-truth documents to FAISS
        3. verify: Run the full verification pipeline on an LLM response
    """

    def __init__(self):
        # Initialize all components (no API key needed — Ollama is local)
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.claim_extractor = ClaimExtractor()
        self.verifier = VerificationEngine()

        # Try to load existing FAISS index from disk
        self.vector_store.load()

        logger.info(
            "pipeline_initialized",
            index_size=self.vector_store.size,
        )

    def ingest_documents(
        self,
        documents: list[dict],
    ) -> IngestResponse:
        """
        Add documents to the knowledge base.

        Steps:
            1. Chunk each document into smaller pieces
            2. Embed all chunks in a batch
            3. Add vectors + metadata to FAISS
            4. Save index to disk
        """
        start_time = time.time()
        all_chunks = []

        # Step 1: Chunk all documents
        for doc in documents:
            doc_id = str(uuid.uuid4())[:8]
            chunks = create_chunks_from_document(
                text=doc["text"],
                document_id=doc_id,
                source=doc.get("source", "unknown"),
                metadata=doc.get("metadata", {}),
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            return IngestResponse(
                documents_processed=len(documents),
                chunks_created=0,
                index_size=self.vector_store.size,
            )

        # Step 2: Batch embed all chunks
        chunk_texts = [c.text for c in all_chunks]
        vectors = self.embedding_service.embed_batch(chunk_texts)

        # Step 3: Add to FAISS
        self.vector_store.add_vectors(vectors, all_chunks)

        # Step 4: Persist to disk
        self.vector_store.save()

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            "documents_ingested",
            num_documents=len(documents),
            num_chunks=len(all_chunks),
            index_size=self.vector_store.size,
            elapsed_ms=round(elapsed, 1),
        )

        return IngestResponse(
            documents_processed=len(documents),
            chunks_created=len(all_chunks),
            index_size=self.vector_store.size,
        )

    def verify(self, request: VerifyRequest) -> VerifyResponse:
        """
        Full verification pipeline for an LLM response.

        Steps:
            1. Extract claims from the LLM response
            2. Embed all claims in a single batch
            3. Batch search FAISS for evidence for each claim
            4. Verify each claim against its evidence
            5. Compute overall reliability score
            6. Return structured report
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]

        logger.info(
            "verification_started",
            request_id=request_id,
            response_length=len(request.llm_response),
        )

        # ── Step 1: Extract claims ────────────────────────────────────────
        claims = self.claim_extractor.extract(request.llm_response)

        if not claims:
            elapsed = (time.time() - start_time) * 1000
            return VerifyResponse(
                request_id=request_id,
                original_response=request.llm_response,
                overall_reliability_score=0.5,
                total_claims=0,
                supported_claims=0,
                contradicted_claims=0,
                unverifiable_claims=0,
                claim_verifications=[],
                verification_time_ms=round(elapsed, 1),
            )

        # ── Step 2: Batch embed all claims ────────────────────────────────
        claim_texts = [c.text for c in claims]
        claim_vectors = self.embedding_service.embed_batch(claim_texts)

        # ── Step 3: Batch FAISS search ────────────────────────────────────
        all_evidence = self.vector_store.batch_search(claim_vectors)

        # ── Step 4: Verify each claim ─────────────────────────────────────
        verifications: list[ClaimVerification] = []
        for claim, evidence in zip(claims, all_evidence):
            verification = self.verifier.verify_claim(claim, evidence)
            verifications.append(verification)

        # ── Step 5: Compute overall score ─────────────────────────────────
        overall_score = self.verifier.compute_overall_score(verifications)

        # ── Step 6: Build response ────────────────────────────────────────
        elapsed = (time.time() - start_time) * 1000

        supported = sum(1 for v in verifications if v.status.value == "supported")
        contradicted = sum(1 for v in verifications if v.status.value == "contradicted")
        unverifiable = sum(1 for v in verifications if v.status.value == "unverifiable")

        response = VerifyResponse(
            request_id=request_id,
            original_response=request.llm_response,
            overall_reliability_score=overall_score,
            total_claims=len(claims),
            supported_claims=supported,
            contradicted_claims=contradicted,
            unverifiable_claims=unverifiable,
            claim_verifications=verifications,
            verification_time_ms=round(elapsed, 1),
        )

        logger.info(
            "verification_complete",
            request_id=request_id,
            total_claims=len(claims),
            supported=supported,
            contradicted=contradicted,
            unverifiable=unverifiable,
            overall_score=overall_score,
            elapsed_ms=round(elapsed, 1),
        )

        return response
