"""Unit tests for the SemanticRetriever service.

Tests cover:
- embed_query: single query embedding generation
- embed_chunks: batch embedding generation with batching logic
- search: pgvector cosine similarity search via Supabase RPC
- Error handling for embedding model failures
"""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from app.models.chunk import CodeChunk, ChunkType
from app.models.retrieval import ScoredChunk
from app.services.semantic_retriever import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    MAX_BATCH_SIZE,
    EmbeddingError,
    SemanticRetriever,
)


# ============================================================================
# Fixtures
# ============================================================================


def make_code_chunk(
    id: str = "chunk-1",
    repo_id: str = "repo-1",
    file_path: str = "src/main.py",
    language: str = "python",
    chunk_type: ChunkType = ChunkType.FUNCTION,
    content: str = "def hello():\n    return 'world'",
    start_line: int = 1,
    end_line: int = 2,
) -> CodeChunk:
    """Helper to create a CodeChunk for testing."""
    return CodeChunk(
        id=id,
        repo_id=repo_id,
        file_path=file_path,
        language=language,
        chunk_type=chunk_type,
        content=content,
        start_line=start_line,
        end_line=end_line,
    )


@pytest.fixture
def mock_model():
    """Create a mock SentenceTransformer model."""
    model = MagicMock()
    # Default: return a 384-dimensional vector
    model.encode.return_value = np.random.rand(EMBEDDING_DIMENSION).astype(np.float32)
    return model


@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    client = MagicMock()
    return client


@pytest.fixture
def retriever(mock_model, mock_supabase):
    """Create a SemanticRetriever with mocked dependencies."""
    return SemanticRetriever(supabase_client=mock_supabase, model=mock_model)


# ============================================================================
# Tests: embed_query
# ============================================================================


class TestEmbedQuery:
    """Tests for the embed_query method."""

    def test_embed_query_returns_list_of_floats(self, retriever, mock_model):
        """embed_query should return a list of floats with correct dimensionality."""
        expected_vector = np.ones(EMBEDDING_DIMENSION, dtype=np.float32)
        mock_model.encode.return_value = expected_vector

        result = retriever.embed_query("how does authentication work?")

        assert isinstance(result, list)
        assert len(result) == EMBEDDING_DIMENSION
        assert all(isinstance(v, float) for v in result)

    def test_embed_query_calls_model_with_query_text(self, retriever, mock_model):
        """embed_query should pass the query text to the model."""
        query = "find the login function"
        retriever.embed_query(query)

        mock_model.encode.assert_called_once_with(query, convert_to_numpy=True)

    def test_embed_query_raises_embedding_error_on_failure(self, retriever, mock_model):
        """embed_query should raise EmbeddingError if the model fails."""
        mock_model.encode.side_effect = RuntimeError("Model crashed")

        with pytest.raises(EmbeddingError) as exc_info:
            retriever.embed_query("test query")

        assert "Failed to generate embedding for query" in str(exc_info.value)
        assert exc_info.value.query == "test query"

    def test_embed_query_preserves_vector_values(self, retriever, mock_model):
        """embed_query should preserve the exact values from the model."""
        expected = np.array([0.1, 0.2, 0.3] + [0.0] * (EMBEDDING_DIMENSION - 3), dtype=np.float32)
        mock_model.encode.return_value = expected

        result = retriever.embed_query("test")

        assert result[:3] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)


# ============================================================================
# Tests: embed_chunks
# ============================================================================


class TestEmbedChunks:
    """Tests for the embed_chunks method."""

    def test_embed_chunks_returns_embeddings_for_all_chunks(self, retriever, mock_model):
        """embed_chunks should return one embedding per input chunk."""
        chunks = [make_code_chunk(id=f"chunk-{i}") for i in range(3)]
        mock_model.encode.return_value = np.random.rand(3, EMBEDDING_DIMENSION).astype(np.float32)

        result = retriever.embed_chunks(chunks)

        assert len(result) == 3
        assert all(len(emb) == EMBEDDING_DIMENSION for emb in result)

    def test_embed_chunks_empty_list_returns_empty(self, retriever, mock_model):
        """embed_chunks should return empty list for empty input."""
        result = retriever.embed_chunks([])

        assert result == []
        mock_model.encode.assert_not_called()

    def test_embed_chunks_passes_content_to_model(self, retriever, mock_model):
        """embed_chunks should pass chunk content texts to the model."""
        chunks = [
            make_code_chunk(id="c1", content="def foo(): pass"),
            make_code_chunk(id="c2", content="class Bar: pass"),
        ]
        mock_model.encode.return_value = np.random.rand(2, EMBEDDING_DIMENSION).astype(np.float32)

        retriever.embed_chunks(chunks)

        mock_model.encode.assert_called_once_with(
            ["def foo(): pass", "class Bar: pass"], convert_to_numpy=True
        )

    def test_embed_chunks_batches_large_input(self, retriever, mock_model):
        """embed_chunks should process in batches of MAX_BATCH_SIZE (256)."""
        # Create 300 chunks (should be split into 2 batches: 256 + 44)
        chunks = [make_code_chunk(id=f"chunk-{i}") for i in range(300)]

        # Return correct shapes for each batch call
        mock_model.encode.side_effect = [
            np.random.rand(256, EMBEDDING_DIMENSION).astype(np.float32),
            np.random.rand(44, EMBEDDING_DIMENSION).astype(np.float32),
        ]

        result = retriever.embed_chunks(chunks)

        assert len(result) == 300
        assert mock_model.encode.call_count == 2

    def test_embed_chunks_raises_embedding_error_on_failure(self, retriever, mock_model):
        """embed_chunks should raise EmbeddingError if the model fails."""
        chunks = [make_code_chunk(id="c1")]
        mock_model.encode.side_effect = RuntimeError("OOM error")

        with pytest.raises(EmbeddingError) as exc_info:
            retriever.embed_chunks(chunks)

        assert "Failed to generate embeddings for chunk batch" in str(exc_info.value)

    def test_embed_chunks_single_chunk(self, retriever, mock_model):
        """embed_chunks should handle a single chunk correctly."""
        chunks = [make_code_chunk(id="single")]
        mock_model.encode.return_value = np.random.rand(1, EMBEDDING_DIMENSION).astype(np.float32)

        result = retriever.embed_chunks(chunks)

        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIMENSION


# ============================================================================
# Tests: search
# ============================================================================


class TestSearch:
    """Tests for the search method."""

    def test_search_returns_scored_chunks(self, retriever, mock_supabase):
        """search should return ScoredChunk objects from pgvector results."""
        # Mock RPC response
        mock_response = MagicMock()
        mock_response.data = [
            {
                "id": "uuid-1",
                "repo_id": "repo-uuid-1",
                "file_path": "src/auth.py",
                "language": "python",
                "chunk_type": "function",
                "content": "def login(user): pass",
                "start_line": 10,
                "end_line": 15,
                "function_name": "login",
                "class_name": None,
                "similarity": 0.85,
            },
        ]
        mock_supabase.rpc.return_value.execute.return_value = mock_response

        query_embedding = [0.1] * EMBEDDING_DIMENSION
        results = retriever.search(query_embedding, top_k=10)

        assert len(results) == 1
        assert isinstance(results[0], ScoredChunk)
        assert results[0].score == 0.85
        assert results[0].source == "semantic"
        assert results[0].chunk.file_path == "src/auth.py"
        assert results[0].chunk.language == "python"

    def test_search_calls_rpc_with_correct_params(self, retriever, mock_supabase):
        """search should call match_chunks RPC with correct parameters."""
        mock_response = MagicMock()
        mock_response.data = []
        mock_supabase.rpc.return_value.execute.return_value = mock_response

        query_embedding = [0.5] * EMBEDDING_DIMENSION
        retriever.search(query_embedding, top_k=15)

        mock_supabase.rpc.assert_called_once_with(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": 15,
                "similarity_threshold": 0.0,
            },
        )

    def test_search_excludes_negative_similarity(self, retriever, mock_supabase):
        """search should exclude results with cosine similarity below 0.0."""
        mock_response = MagicMock()
        mock_response.data = [
            {
                "id": "uuid-1",
                "repo_id": "repo-uuid-1",
                "file_path": "src/good.py",
                "language": "python",
                "chunk_type": "function",
                "content": "def good(): pass",
                "start_line": 1,
                "end_line": 2,
                "function_name": "good",
                "class_name": None,
                "similarity": 0.7,
            },
            {
                "id": "uuid-2",
                "repo_id": "repo-uuid-2",
                "file_path": "src/bad.py",
                "language": "python",
                "chunk_type": "function",
                "content": "def bad(): pass",
                "start_line": 1,
                "end_line": 2,
                "function_name": "bad",
                "class_name": None,
                "similarity": -0.1,
            },
        ]
        mock_supabase.rpc.return_value.execute.return_value = mock_response

        query_embedding = [0.1] * EMBEDDING_DIMENSION
        results = retriever.search(query_embedding)

        assert len(results) == 1
        assert results[0].chunk.file_path == "src/good.py"

    def test_search_returns_empty_list_when_no_data(self, retriever, mock_supabase):
        """search should return empty list when no results from database."""
        mock_response = MagicMock()
        mock_response.data = []
        mock_supabase.rpc.return_value.execute.return_value = mock_response

        results = retriever.search([0.1] * EMBEDDING_DIMENSION)

        assert results == []

    def test_search_returns_empty_list_when_data_is_none(self, retriever, mock_supabase):
        """search should return empty list when response data is None."""
        mock_response = MagicMock()
        mock_response.data = None
        mock_supabase.rpc.return_value.execute.return_value = mock_response

        results = retriever.search([0.1] * EMBEDDING_DIMENSION)

        assert results == []

    def test_search_multiple_results_preserve_order(self, retriever, mock_supabase):
        """search should preserve the order of results from the database."""
        mock_response = MagicMock()
        mock_response.data = [
            {
                "id": "uuid-1",
                "repo_id": "repo-uuid-1",
                "file_path": "src/first.py",
                "language": "python",
                "chunk_type": "function",
                "content": "def first(): pass",
                "start_line": 1,
                "end_line": 2,
                "function_name": "first",
                "class_name": None,
                "similarity": 0.95,
            },
            {
                "id": "uuid-2",
                "repo_id": "repo-uuid-2",
                "file_path": "src/second.py",
                "language": "python",
                "chunk_type": "method",
                "content": "def second(): pass",
                "start_line": 5,
                "end_line": 8,
                "function_name": "second",
                "class_name": "MyClass",
                "similarity": 0.72,
            },
        ]
        mock_supabase.rpc.return_value.execute.return_value = mock_response

        results = retriever.search([0.1] * EMBEDDING_DIMENSION)

        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[0].chunk.file_path == "src/first.py"
        assert results[1].score == 0.72
        assert results[1].chunk.file_path == "src/second.py"

    def test_search_default_top_k_is_20(self, retriever, mock_supabase):
        """search should default to top_k=20."""
        mock_response = MagicMock()
        mock_response.data = []
        mock_supabase.rpc.return_value.execute.return_value = mock_response

        retriever.search([0.1] * EMBEDDING_DIMENSION)

        call_args = mock_supabase.rpc.call_args
        assert call_args[0][1]["match_count"] == 20


# ============================================================================
# Tests: Initialization and error handling
# ============================================================================


class TestInitialization:
    """Tests for SemanticRetriever initialization."""

    def test_init_with_provided_model(self, mock_supabase, mock_model):
        """Should use the provided model without loading a new one."""
        retriever = SemanticRetriever(supabase_client=mock_supabase, model=mock_model)

        assert retriever.model is mock_model
        assert retriever.supabase is mock_supabase

    @patch("app.services.semantic_retriever.SentenceTransformer")
    def test_init_loads_default_model_when_none_provided(
        self, mock_st_class, mock_supabase
    ):
        """Should load all-MiniLM-L6-v2 when no model is provided."""
        mock_st_class.return_value = MagicMock()

        retriever = SemanticRetriever(supabase_client=mock_supabase)

        mock_st_class.assert_called_once_with(EMBEDDING_MODEL_NAME)

    @patch("app.services.semantic_retriever.SentenceTransformer")
    def test_init_raises_embedding_error_when_model_load_fails(
        self, mock_st_class, mock_supabase
    ):
        """Should raise EmbeddingError if the model cannot be loaded."""
        mock_st_class.side_effect = OSError("Model not found")

        with pytest.raises(EmbeddingError) as exc_info:
            SemanticRetriever(supabase_client=mock_supabase)

        assert "Failed to load embedding model" in str(exc_info.value)
        assert EMBEDDING_MODEL_NAME in str(exc_info.value)
