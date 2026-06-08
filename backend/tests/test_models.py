"""Unit tests for core data models."""

import pytest
from pydantic import ValidationError

from app.models.chunk import (
    ChunkMetadata,
    ChunkType,
    CodeChunk,
    SUPPORTED_LANGUAGES,
)
from app.models.query import ProcessedQuery, QueryFilters, QueryIntent
from app.models.retrieval import RetrievalResult, ScoredChunk
from app.models.answer import GroundedAnswer, SourceReference


# --- Helper to create a valid CodeChunk ---


def make_chunk(**overrides) -> CodeChunk:
    """Create a valid CodeChunk with sensible defaults, allowing overrides."""
    defaults = {
        "id": "chunk-001",
        "repo_id": "repo-1",
        "file_path": "src/utils/helper.py",
        "language": "python",
        "chunk_type": ChunkType.FUNCTION,
        "content": "def hello():\n    return 'world'",
        "start_line": 1,
        "end_line": 2,
        "parent_id": None,
        "metadata": ChunkMetadata(),
    }
    defaults.update(overrides)
    return CodeChunk(**defaults)


# --- CodeChunk Validation Tests ---


class TestCodeChunkValidation:
    """Tests for CodeChunk Pydantic validation rules."""

    def test_valid_chunk_creation(self):
        chunk = make_chunk()
        assert chunk.id == "chunk-001"
        assert chunk.language == "python"
        assert chunk.start_line == 1
        assert chunk.end_line == 2

    def test_content_must_be_non_empty(self):
        with pytest.raises(ValidationError, match="content must be non-empty"):
            make_chunk(content="")

    def test_content_whitespace_only_rejected(self):
        with pytest.raises(ValidationError, match="content must be non-empty"):
            make_chunk(content="   \n\t  ")

    def test_start_line_must_be_at_least_1(self):
        with pytest.raises(ValidationError):
            make_chunk(start_line=0)

    def test_negative_start_line_rejected(self):
        with pytest.raises(ValidationError):
            make_chunk(start_line=-1)

    def test_start_line_must_be_lte_end_line(self):
        with pytest.raises(ValidationError, match="start_line.*must be <= end_line"):
            make_chunk(start_line=10, end_line=5)

    def test_start_line_equals_end_line_valid(self):
        chunk = make_chunk(start_line=5, end_line=5)
        assert chunk.start_line == chunk.end_line == 5

    def test_file_path_rejects_absolute_unix(self):
        with pytest.raises(ValidationError, match="relative path"):
            make_chunk(file_path="/etc/passwd")

    def test_file_path_rejects_absolute_windows(self):
        with pytest.raises(ValidationError, match="relative path"):
            make_chunk(file_path="C:\\Users\\test.py")

    def test_file_path_rejects_path_traversal(self):
        with pytest.raises(ValidationError, match="path traversal"):
            make_chunk(file_path="src/../../../etc/passwd")

    def test_file_path_rejects_leading_dotdot(self):
        with pytest.raises(ValidationError, match="path traversal"):
            make_chunk(file_path="../secret.py")

    def test_file_path_accepts_dots_in_filename(self):
        chunk = make_chunk(file_path="src/file.test.py")
        assert chunk.file_path == "src/file.test.py"

    def test_file_path_accepts_nested_relative_path(self):
        chunk = make_chunk(file_path="src/components/Button.tsx")
        assert chunk.file_path == "src/components/Button.tsx"

    def test_file_path_rejects_empty(self):
        with pytest.raises(ValidationError, match="file_path must be non-empty"):
            make_chunk(file_path="")

    def test_language_must_be_supported(self):
        with pytest.raises(ValidationError, match="language must be one of"):
            make_chunk(language="rust")

    def test_all_supported_languages_accepted(self):
        for lang in SUPPORTED_LANGUAGES:
            chunk = make_chunk(language=lang)
            assert chunk.language == lang

    def test_language_case_insensitive(self):
        chunk = make_chunk(language="Python")
        assert chunk.language == "python"

        chunk = make_chunk(language="TYPESCRIPT")
        assert chunk.language == "typescript"

    def test_chunk_type_enum_values(self):
        for ct in ChunkType:
            chunk = make_chunk(chunk_type=ct)
            assert chunk.chunk_type == ct


# --- ChunkMetadata Tests ---


class TestChunkMetadata:
    def test_defaults_are_empty(self):
        meta = ChunkMetadata()
        assert meta.function_name is None
        assert meta.class_name is None
        assert meta.module_name is None
        assert meta.imports == []
        assert meta.dependencies == []
        assert meta.docstring is None
        assert meta.signatures == []
        assert meta.tags == []

    def test_full_metadata(self):
        meta = ChunkMetadata(
            function_name="get_user",
            class_name="UserService",
            module_name="services.user",
            imports=["from models import User"],
            dependencies=["chunk-002"],
            docstring="Fetches user by ID.",
            signatures=["def get_user(user_id: str) -> User"],
            tags=["auth", "database"],
        )
        assert meta.function_name == "get_user"
        assert len(meta.imports) == 1
        assert len(meta.tags) == 2


# --- QueryIntent and ProcessedQuery Tests ---


class TestQueryModels:
    def test_query_intent_values(self):
        assert QueryIntent.CODE_LOOKUP.value == "code_lookup"
        assert QueryIntent.DOCUMENTATION.value == "documentation"
        assert QueryIntent.ARCHITECTURE.value == "architecture"
        assert QueryIntent.DEBUGGING.value == "debugging"
        assert QueryIntent.API_USAGE.value == "api_usage"

    def test_processed_query_minimal(self):
        q = ProcessedQuery(
            original_question="How does auth work?",
            intent=QueryIntent.ARCHITECTURE,
        )
        assert q.original_question == "How does auth work?"
        assert q.expanded_terms == []
        assert q.query_embedding == []

    def test_query_filters_defaults(self):
        f = QueryFilters()
        assert f.languages is None
        assert f.file_patterns is None
        assert f.chunk_types is None
        assert f.repo_ids is None

    def test_query_filters_with_values(self):
        f = QueryFilters(
            languages=["python", "typescript"],
            file_patterns=["src/**/*.py"],
            chunk_types=[ChunkType.FUNCTION, ChunkType.METHOD],
            repo_ids=["repo-1"],
        )
        assert len(f.languages) == 2
        assert len(f.chunk_types) == 2


# --- Retrieval Result Tests ---


class TestRetrievalModels:
    def test_scored_chunk_valid(self):
        chunk = make_chunk()
        sc = ScoredChunk(chunk=chunk, score=0.85, source="bm25")
        assert sc.score == 0.85
        assert sc.source == "bm25"

    def test_scored_chunk_rejects_negative_score(self):
        chunk = make_chunk()
        with pytest.raises(ValidationError):
            ScoredChunk(chunk=chunk, score=-0.1, source="semantic")

    def test_retrieval_result_valid(self):
        chunk = make_chunk()
        rr = RetrievalResult(
            chunk=chunk,
            fused_score=0.5,
            bm25_rank=3,
            semantic_rank=1,
            context_snippet="def hello():\n    return 'world'",
        )
        assert rr.fused_score == 0.5
        assert rr.bm25_rank == 3
        assert rr.semantic_rank == 1

    def test_retrieval_result_rejects_zero_fused_score(self):
        chunk = make_chunk()
        with pytest.raises(ValidationError):
            RetrievalResult(chunk=chunk, fused_score=0.0)

    def test_retrieval_result_rejects_negative_fused_score(self):
        chunk = make_chunk()
        with pytest.raises(ValidationError):
            RetrievalResult(chunk=chunk, fused_score=-0.1)

    def test_retrieval_result_null_ranks(self):
        chunk = make_chunk()
        rr = RetrievalResult(chunk=chunk, fused_score=0.1)
        assert rr.bm25_rank is None
        assert rr.semantic_rank is None


# --- Answer Model Tests ---


class TestAnswerModels:
    def test_source_reference_valid(self):
        sr = SourceReference(
            file_path="src/auth.py",
            function_name="login",
            start_line=10,
            end_line=25,
            snippet="def login(...):",
            relevance=0.9,
        )
        assert sr.file_path == "src/auth.py"
        assert sr.relevance == 0.9

    def test_source_reference_rejects_invalid_relevance(self):
        with pytest.raises(ValidationError):
            SourceReference(
                file_path="f.py",
                start_line=1,
                end_line=1,
                relevance=1.5,
            )

    def test_grounded_answer_valid(self):
        answer = GroundedAnswer(
            answer_text="The login function authenticates users.",
            sources=[],
            confidence=0.85,
            retrieval_metadata={"chunks_used": 3},
        )
        assert answer.confidence == 0.85
        assert answer.retrieval_metadata["chunks_used"] == 3

    def test_grounded_answer_confidence_bounds(self):
        with pytest.raises(ValidationError):
            GroundedAnswer(
                answer_text="test",
                confidence=1.5,
            )
        with pytest.raises(ValidationError):
            GroundedAnswer(
                answer_text="test",
                confidence=-0.1,
            )

    def test_grounded_answer_zero_confidence(self):
        answer = GroundedAnswer(
            answer_text="No relevant sources found.",
            confidence=0.0,
        )
        assert answer.confidence == 0.0
        assert answer.sources == []
