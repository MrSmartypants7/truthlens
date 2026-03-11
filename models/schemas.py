"""
Data models for TruthLens.

WHY PYDANTIC MODELS: They enforce data contracts. If the verification engine
returns a confidence score of "high" instead of 0.85, Pydantic catches it
immediately. This prevents subtle bugs from propagating through the pipeline.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class VerificationStatus(str, Enum):
    """Possible outcomes of verifying a single claim."""
    SUPPORTED = "supported"           # Evidence confirms the claim
    CONTRADICTED = "contradicted"     # Evidence directly contradicts the claim
    UNVERIFIABLE = "unverifiable"     # No relevant evidence found
    PARTIALLY_SUPPORTED = "partially_supported"  # Some evidence, but incomplete


class SeverityLevel(str, Enum):
    """How serious is a hallucination?"""
    LOW = "low"           # Minor inaccuracy (e.g., off by a small amount)
    MEDIUM = "medium"     # Factually wrong but not dangerous
    HIGH = "high"         # Could cause real harm if believed
    CRITICAL = "critical" # Dangerous misinformation


# ─── Document / Chunk Models ─────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A single chunk of a source document stored in FAISS."""
    chunk_id: str = Field(description="Unique ID for this chunk")
    document_id: str = Field(description="ID of the parent document")
    text: str = Field(description="The actual text content")
    source: str = Field(default="unknown", description="Where this doc came from")
    metadata: dict = Field(default_factory=dict, description="Extra metadata")


class RetrievedEvidence(BaseModel):
    """A chunk retrieved from FAISS as potential evidence for a claim."""
    chunk: DocumentChunk
    similarity_score: float = Field(description="L2 distance (lower = more similar)")
    rank: int = Field(description="Rank in retrieval results (1 = most similar)")


# ─── Claim Models ────────────────────────────────────────────────────────────

class Claim(BaseModel):
    """A single verifiable claim extracted from an LLM response."""
    claim_id: str
    text: str = Field(description="The claim text itself")
    original_sentence: str = Field(description="The sentence it was extracted from")


class ClaimVerification(BaseModel):
    """Result of verifying a single claim against evidence."""
    claim: Claim
    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the verdict")
    severity: SeverityLevel = Field(default=SeverityLevel.LOW)
    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Why the verifier reached this verdict")


# ─── API Request/Response Models ─────────────────────────────────────────────

class VerifyRequest(BaseModel):
    """Incoming request to verify an LLM response."""
    llm_response: str = Field(description="The LLM-generated text to verify")
    query: Optional[str] = Field(default=None, description="Original user query (optional context)")
    model_name: Optional[str] = Field(default=None, description="Which LLM generated this")

    class Config:
        json_schema_extra = {
            "example": {
                "llm_response": "The Eiffel Tower is 324 meters tall and was built in 1887.",
                "query": "How tall is the Eiffel Tower?",
                "model_name": "gpt-4o"
            }
        }


class VerifyResponse(BaseModel):
    """Full verification report returned to the caller."""
    request_id: str
    original_response: str
    overall_reliability_score: float = Field(ge=0.0, le=1.0)
    total_claims: int
    supported_claims: int
    contradicted_claims: int
    unverifiable_claims: int
    claim_verifications: list[ClaimVerification]
    verification_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Document Ingestion ──────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Request to add documents to the knowledge base."""
    documents: list[dict] = Field(description="List of {text, source, metadata} dicts")

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "text": "The Eiffel Tower is 330 metres tall...",
                        "source": "wikipedia",
                        "metadata": {"title": "Eiffel Tower"}
                    }
                ]
            }
        }


class IngestResponse(BaseModel):
    """Result of document ingestion."""
    documents_processed: int
    chunks_created: int
    index_size: int


# ─── Experiment Tracking ─────────────────────────────────────────────────────

class BenchmarkResult(BaseModel):
    """Result of running a benchmark evaluation."""
    experiment_id: str
    model_name: str
    total_claims: int
    accuracy: float
    precision: float
    recall: float
    avg_verification_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: dict = Field(default_factory=dict)
