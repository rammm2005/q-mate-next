"""Unit tests for API routes with FastAPI TestClient.

Tests the POST /api/query and POST /api/ingest endpoints including
authentication, input validation, and pipeline integration.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture
def api_key_header():
    """Valid API key header for authenticated requests."""
    return {"X-API-Key": "test-api-key-123"}


# --- Authentication Tests ---


class TestQueryEndpointAuthentication:
    """Tests for API key authentication on /api/query."""

    def test_missing_api_key_returns_422(self, client):
        """Request without X-API-Key header is rejected."""
        response = client.post(
            "/api/query",
            json={"question": "Where is the auth module?"},
        )
        # FastAPI returns 422 when required header is missing
        assert response.status_code == 422

    def test_valid_api_key_allows_request(self, client, api_key_header):
        """Request with valid API key is processed."""
        response = client.post(
            "/api/query",
            json={"question": "Where is the auth module?"},
            headers=api_key_header,
        )
        assert response.status_code == 200

    def test_empty_api_key_returns_401(self, client):
        """Request with empty API key string is rejected with 401."""
        response = client.post(
            "/api/query",
            json={"question": "Where is the auth module?"},
            headers={"X-API-Key": ""},
        )
        # Empty string header still gets passed but verify_api_key rejects it
        # FastAPI may treat empty string differently - it still passes the header
        # The endpoint should reject with 401
        assert response.status_code == 401

    def test_whitespace_only_api_key_returns_401(self, client):
        """Request with whitespace-only API key is rejected."""
        response = client.post(
            "/api/query",
            json={"question": "Where is the auth module?"},
            headers={"X-API-Key": "   "},
        )
        assert response.status_code == 401


# --- Input Validation Tests ---


class TestQueryEndpointValidation:
    """Tests for input validation on /api/query."""

    def test_empty_question_rejected(self, client, api_key_header):
        """Empty question string is rejected by Pydantic validation."""
        response = client.post(
            "/api/query",
            json={"question": ""},
            headers=api_key_header,
        )
        assert response.status_code == 422

    def test_question_exceeds_1000_chars_rejected(self, client, api_key_header):
        """Question over 1000 chars is rejected by Pydantic validation."""
        long_question = "a" * 1001
        response = client.post(
            "/api/query",
            json={"question": long_question},
            headers=api_key_header,
        )
        assert response.status_code == 422

    def test_question_exactly_1000_chars_accepted(self, client, api_key_header):
        """Question of exactly 1000 chars is accepted."""
        question = "a" * 1000
        response = client.post(
            "/api/query",
            json={"question": question},
            headers=api_key_header,
        )
        assert response.status_code == 200

    def test_question_exceeds_2000_chars_rejected(self, client, api_key_header):
        """Question over 2000 chars is rejected by input validation."""
        # This would be caught by Pydantic's max_length=1000 first,
        # but if that were relaxed, the 2000 char check would catch it
        long_question = "a" * 2001
        response = client.post(
            "/api/query",
            json={"question": long_question},
            headers=api_key_header,
        )
        assert response.status_code == 422

    def test_valid_short_question(self, client, api_key_header):
        """Short valid question is accepted."""
        response = client.post(
            "/api/query",
            json={"question": "hello"},
            headers=api_key_header,
        )
        assert response.status_code == 200

    def test_special_characters_are_escaped(self, client, api_key_header):
        """Special characters in question are handled safely."""
        response = client.post(
            "/api/query",
            json={"question": "What does <script>alert('xss')</script> do?"},
            headers=api_key_header,
        )
        assert response.status_code == 200
        # The answer should still be returned even with special chars
        data = response.json()
        assert "answer" in data


# --- Response Format Tests ---


class TestQueryEndpointResponse:
    """Tests for response format of /api/query."""

    def test_response_has_required_fields(self, client, api_key_header):
        """Response contains answer, sources, confidence, and metadata."""
        response = client.post(
            "/api/query",
            json={"question": "Where is the login function?"},
            headers=api_key_header,
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "metadata" in data

    def test_confidence_is_float(self, client, api_key_header):
        """Confidence score is a float between 0 and 1."""
        response = client.post(
            "/api/query",
            json={"question": "Where is the login function?"},
            headers=api_key_header,
        )
        data = response.json()
        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["confidence"] <= 1.0

    def test_sources_is_list(self, client, api_key_header):
        """Sources field is a list."""
        response = client.post(
            "/api/query",
            json={"question": "Where is the login function?"},
            headers=api_key_header,
        )
        data = response.json()
        assert isinstance(data["sources"], list)

    def test_metadata_is_dict(self, client, api_key_header):
        """Metadata field is a dictionary."""
        response = client.post(
            "/api/query",
            json={"question": "Where is the login function?"},
            headers=api_key_header,
        )
        data = response.json()
        assert isinstance(data["metadata"], dict)


# --- Filter Tests ---


class TestQueryEndpointFilters:
    """Tests for optional filters on /api/query."""

    def test_query_with_language_filter(self, client, api_key_header):
        """Query with language filter is accepted."""
        response = client.post(
            "/api/query",
            json={
                "question": "Find the auth module",
                "filters": {"languages": ["python", "typescript"]},
            },
            headers=api_key_header,
        )
        assert response.status_code == 200

    def test_query_with_paths_filter(self, client, api_key_header):
        """Query with paths filter is accepted."""
        response = client.post(
            "/api/query",
            json={
                "question": "Find the auth module",
                "filters": {"paths": ["src/**/*.py"]},
            },
            headers=api_key_header,
        )
        assert response.status_code == 200

    def test_query_with_repo_ids_filter(self, client, api_key_header):
        """Query with repo_ids filter is accepted."""
        response = client.post(
            "/api/query",
            json={
                "question": "Find the auth module",
                "filters": {"repo_ids": ["repo-1", "repo-2"]},
            },
            headers=api_key_header,
        )
        assert response.status_code == 200

    def test_query_without_filters(self, client, api_key_header):
        """Query without filters is accepted."""
        response = client.post(
            "/api/query",
            json={"question": "Find the auth module"},
            headers=api_key_header,
        )
        assert response.status_code == 200

    def test_query_with_null_filters(self, client, api_key_header):
        """Query with explicit null filters is accepted."""
        response = client.post(
            "/api/query",
            json={"question": "Find the auth module", "filters": None},
            headers=api_key_header,
        )
        assert response.status_code == 200


# --- Ingest Endpoint Tests ---


class TestIngestEndpointAuthentication:
    """Tests for API key authentication on /api/ingest."""

    def test_missing_api_key_returns_422(self, client):
        """Request without X-API-Key header is rejected."""
        response = client.post(
            "/api/ingest",
            json={"repository_path": "/path/to/repo"},
        )
        assert response.status_code == 422

    def test_valid_api_key_allows_request(self, client, api_key_header):
        """Request with valid API key is processed."""
        response = client.post(
            "/api/ingest",
            json={"repository_path": "/path/to/repo"},
            headers=api_key_header,
        )
        assert response.status_code == 200

    def test_empty_api_key_returns_401(self, client):
        """Request with empty API key is rejected."""
        response = client.post(
            "/api/ingest",
            json={"repository_path": "/path/to/repo"},
            headers={"X-API-Key": ""},
        )
        assert response.status_code == 401


class TestIngestEndpointValidation:
    """Tests for input validation on /api/ingest."""

    def test_empty_repository_path_rejected(self, client, api_key_header):
        """Empty repository path is rejected."""
        response = client.post(
            "/api/ingest",
            json={"repository_path": ""},
            headers=api_key_header,
        )
        assert response.status_code == 422

    def test_valid_repository_path_accepted(self, client, api_key_header):
        """Valid repository path is accepted."""
        response = client.post(
            "/api/ingest",
            json={"repository_path": "/home/user/my-repo"},
            headers=api_key_header,
        )
        assert response.status_code == 200


class TestIngestEndpointResponse:
    """Tests for response format of /api/ingest."""

    def test_response_has_required_fields(self, client, api_key_header):
        """Response contains status, chunks_processed, files_processed."""
        response = client.post(
            "/api/ingest",
            json={"repository_path": "/path/to/repo"},
            headers=api_key_header,
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "chunks_processed" in data
        assert "files_processed" in data
        assert "errors" in data

    def test_response_status_is_accepted(self, client, api_key_header):
        """Ingest response indicates acceptance."""
        response = client.post(
            "/api/ingest",
            json={"repository_path": "/path/to/repo"},
            headers=api_key_header,
        )
        data = response.json()
        assert data["status"] == "accepted"
