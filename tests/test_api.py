"""
Tests for the FastAPI endpoints.

Uses FastAPI's TestClient (backed by httpx) for synchronous testing.
These test the HTTP layer: correct status codes, request validation,
response schemas.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Health check should always work."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "index_size" in data
        assert "version" in data


class TestStatsEndpoint:
    """Stats should return index and cache info."""

    def test_stats_returns_200(self):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "index" in data
        assert "cache" in data
        assert "config" in data


class TestVerifyEndpoint:
    """Verify endpoint request validation."""

    def test_empty_body_returns_422(self):
        """Missing required fields should return validation error."""
        response = client.post("/verify", json={})
        assert response.status_code == 422

    def test_valid_request_format(self):
        """Valid request should be accepted (may fail on OpenAI call but validates)."""
        response = client.post("/verify", json={
            "llm_response": "Test claim.",
            "query": "test",
        })
        # Will either succeed or fail on OpenAI call, but should not be 422
        assert response.status_code in [200, 500]


class TestIngestEndpoint:
    """Ingest endpoint request validation."""

    def test_empty_documents(self):
        """Empty document list should work."""
        response = client.post("/ingest", json={"documents": []})
        assert response.status_code == 200
        data = response.json()
        assert data["documents_processed"] == 0


class TestExperimentsEndpoint:
    def test_list_experiments(self):
        response = client.get("/experiments")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestRequestLogging:
    """Verify that the middleware adds timing headers."""

    def test_timing_header_present(self):
        response = client.get("/health")
        assert "X-Process-Time-Ms" in response.headers


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
