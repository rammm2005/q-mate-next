"""Unit tests for QueryProcessor.process_query full pipeline."""

from unittest.mock import MagicMock

import pytest

from app.models.query import (
    ProcessedQuery,
    QueryContext,
    QueryFilters,
    QueryIntent,
)
from app.services.query_processor import QueryProcessor


@pytest.fixture
def mock_semantic_retriever():
    """Create a mock SemanticRetriever that returns a fake embedding."""
    retriever = MagicMock()
    retriever.embed_query.return_value = [0.1] * 384
    return retriever


@pytest.fixture
def processor():
    """Create a QueryProcessor without a semantic retriever."""
    return QueryProcessor()


@pytest.fixture
def processor_with_retriever(mock_semantic_retriever):
    """Create a QueryProcessor with a mock semantic retriever."""
    return QueryProcessor(semantic_retriever=mock_semantic_retriever)


class TestProcessQueryValidation:
    """Tests for input validation in process_query."""

    def test_empty_string_raises_value_error(self, processor):
        with pytest.raises(ValueError, match="between 1 and 1000"):
            processor.process_query("")

    def test_whitespace_only_raises_value_error(self, processor):
        with pytest.raises(ValueError, match="between 1 and 1000"):
            processor.process_query("   \t\n  ")

    def test_exceeds_1000_chars_raises_value_error(self, processor):
        long_question = "a" * 1001
        with pytest.raises(ValueError, match="between 1 and 1000"):
            processor.process_query(long_question)

    def test_valid_question_does_not_raise(self, processor):
        result = processor.process_query("How does the auth module work?")
        assert isinstance(result, ProcessedQuery)


class TestProcessQueryNormalization:
    """Tests that process_query normalizes input properly."""

    def test_trims_leading_whitespace(self, processor):
        result = processor.process_query("  How does auth work?")
        assert result.original_question == "How does auth work?"

    def test_trims_trailing_whitespace(self, processor):
        result = processor.process_query("How does auth work?   ")
        assert result.original_question == "How does auth work?"

    def test_trims_both_sides(self, processor):
        result = processor.process_query("  How does auth work?  ")
        assert result.original_question == "How does auth work?"


class TestProcessQueryIntentClassification:
    """Tests that process_query correctly classifies intent."""

    def test_debugging_intent(self, processor):
        result = processor.process_query("There's a bug in the login")
        assert result.intent == QueryIntent.DEBUGGING

    def test_api_usage_intent(self, processor):
        result = processor.process_query("How to call the user API")
        assert result.intent == QueryIntent.API_USAGE

    def test_architecture_intent(self, processor):
        result = processor.process_query("What is the architecture?")
        assert result.intent == QueryIntent.ARCHITECTURE

    def test_documentation_intent(self, processor):
        result = processor.process_query("Explain the auth flow")
        assert result.intent == QueryIntent.DOCUMENTATION

    def test_code_lookup_intent(self, processor):
        result = processor.process_query("Find the getUserName function")
        assert result.intent == QueryIntent.CODE_LOOKUP

    def test_default_code_lookup(self, processor):
        result = processor.process_query("getUserName")
        assert result.intent == QueryIntent.CODE_LOOKUP


class TestProcessQueryExpansion:
    """Tests that process_query populates expanded_terms."""

    def test_expanded_terms_non_empty(self, processor):
        result = processor.process_query("getUserName")
        assert len(result.expanded_terms) > 0

    def test_expanded_terms_contains_base_tokens(self, processor):
        result = processor.process_query("getUserName")
        # code_aware_tokenize splits camelCase
        assert "get" in result.expanded_terms
        assert "user" in result.expanded_terms
        assert "name" in result.expanded_terms

    def test_code_lookup_adds_patterns(self, processor):
        result = processor.process_query("Find the parse function")
        assert any(item.startswith("def ") for item in result.expanded_terms)
        assert any(item.startswith("function ") for item in result.expanded_terms)

    def test_api_usage_adds_patterns(self, processor):
        result = processor.process_query("How to call the user API endpoint")
        assert any('@app.get("/' in item for item in result.expanded_terms)


class TestProcessQueryEmbedding:
    """Tests for semantic embedding generation."""

    def test_no_retriever_returns_empty_embedding(self, processor):
        result = processor.process_query("How does auth work?")
        assert result.query_embedding == []

    def test_with_retriever_returns_embedding(self, processor_with_retriever):
        result = processor_with_retriever.process_query("How does auth work?")
        assert len(result.query_embedding) == 384
        assert result.query_embedding == [0.1] * 384

    def test_retriever_called_with_normalized_question(
        self, processor_with_retriever, mock_semantic_retriever
    ):
        processor_with_retriever.process_query("  How does auth work?  ")
        mock_semantic_retriever.embed_query.assert_called_once_with(
            "How does auth work?"
        )


class TestProcessQueryLexicalQuery:
    """Tests for lexical query string generation."""

    def test_lexical_query_non_empty(self, processor):
        result = processor.process_query("getUserName")
        assert result.lexical_query != ""

    def test_lexical_query_is_string(self, processor):
        result = processor.process_query("getUserName")
        assert isinstance(result.lexical_query, str)

    def test_lexical_query_built_from_expanded_terms(self, processor):
        result = processor.process_query("auth flow")
        # lexical_query is a space-joined string of expanded_terms
        assert result.lexical_query == " ".join(result.expanded_terms)

    def test_lexical_query_contains_base_tokens(self, processor):
        result = processor.process_query("getUserName")
        assert "get" in result.lexical_query
        assert "user" in result.lexical_query
        assert "name" in result.lexical_query


class TestProcessQueryFilters:
    """Tests for filter handling via QueryContext."""

    def test_no_context_returns_empty_filters(self, processor):
        result = processor.process_query("getUserName")
        assert result.filters == QueryFilters()

    def test_none_context_returns_empty_filters(self, processor):
        result = processor.process_query("getUserName", context=None)
        assert result.filters == QueryFilters()

    def test_context_filters_passed_through(self, processor):
        ctx = QueryContext(
            filters=QueryFilters(languages=["python", "typescript"])
        )
        result = processor.process_query("getUserName", context=ctx)
        assert result.filters.languages == ["python", "typescript"]

    def test_context_with_repo_ids(self, processor):
        ctx = QueryContext(
            filters=QueryFilters(repo_ids=["repo-1", "repo-2"])
        )
        result = processor.process_query("getUserName", context=ctx)
        assert result.filters.repo_ids == ["repo-1", "repo-2"]

    def test_context_with_file_patterns(self, processor):
        ctx = QueryContext(
            filters=QueryFilters(file_patterns=["src/**/*.py"])
        )
        result = processor.process_query("getUserName", context=ctx)
        assert result.filters.file_patterns == ["src/**/*.py"]


class TestProcessQueryReturnType:
    """Tests that process_query returns properly structured ProcessedQuery."""

    def test_returns_processed_query_instance(self, processor):
        result = processor.process_query("How does auth work?")
        assert isinstance(result, ProcessedQuery)

    def test_all_fields_populated(self, processor_with_retriever):
        result = processor_with_retriever.process_query("Find the bug in API")
        assert result.original_question == "Find the bug in API"
        assert result.intent is not None
        assert len(result.expanded_terms) > 0
        assert len(result.query_embedding) == 384
        assert result.lexical_query != ""
        assert result.filters is not None

    def test_original_question_preserved(self, processor):
        result = processor.process_query("  Hello world  ")
        assert result.original_question == "Hello world"
