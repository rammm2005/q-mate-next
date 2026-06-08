"""Query processor service for intent classification and query handling."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.models.query import ProcessedQuery, QueryContext, QueryFilters, QueryIntent
from app.services.tokenizer import code_aware_tokenize

if TYPE_CHECKING:
    from app.services.semantic_retriever import SemanticRetriever


# Keyword patterns for intent classification (ordered by specificity)
_DEBUGGING_KEYWORDS = [
    "bug", "error", "fix", "crash", "debug", "issue",
    "traceback", "exception", "failure", "broken",
]

_API_USAGE_KEYWORDS = [
    "api", "endpoint", "request", "route", "http", "rest",
    "response", "status code", "url", "webhook",
]

_ARCHITECTURE_KEYWORDS = [
    "architecture", "structure", "module", "dependency", "design",
    "pattern", "layer", "component", "diagram",
]

_DOCUMENTATION_KEYWORDS = [
    "document", "explain", "how does", "what is", "describe",
    "overview", "summary", "readme", "guide",
]

_CODE_LOOKUP_KEYWORDS = [
    "find", "where is", "show me", "locate", "definition",
    "implementation", "function", "class", "method",
]

# Maximum allowed question length
MAX_QUESTION_LENGTH = 1000


class QueryProcessor:
    """Processes developer questions with intent classification and query expansion."""

    def __init__(self, semantic_retriever: SemanticRetriever | None = None):
        """Initialize the QueryProcessor.

        Args:
            semantic_retriever: Optional SemanticRetriever instance for embedding
                generation. If None, process_query will return an empty embedding.
        """
        self._semantic_retriever = semantic_retriever

    def process_query(
        self, question: str, context: QueryContext | None = None
    ) -> ProcessedQuery:
        """Process a raw developer question into a structured ProcessedQuery.

        Normalizes the input, classifies intent, expands the query with
        intent-specific terms, generates a semantic embedding, and builds
        a lexical query string for BM25 search.

        Args:
            question: The developer's natural language question (1-1000 chars).
            context: Optional query context with filters or session info.

        Returns:
            A ProcessedQuery with all fields populated.

        Raises:
            ValueError: If the question is empty or exceeds 1000 characters.
        """
        # Step 1: Validate question (reuse existing validation)
        self._validate_question(question)

        # Normalize input: trim whitespace
        normalized_question = question.strip()

        # Step 2: Classify intent (uses lowercased text internally)
        intent = self.classify_intent(normalized_question)

        # Step 3: Expand query based on intent
        expanded_terms = self.expand_query(normalized_question, intent)

        # Step 4: Generate semantic embedding
        query_embedding: list[float] = []
        if self._semantic_retriever is not None:
            query_embedding = self._semantic_retriever.embed_query(normalized_question)

        # Step 5: Build lexical query string from expanded terms
        lexical_query = " ".join(expanded_terms)

        # Step 6: Build filters from context if provided
        filters = context.filters if context is not None else QueryFilters()

        return ProcessedQuery(
            original_question=normalized_question,
            intent=intent,
            expanded_terms=expanded_terms,
            query_embedding=query_embedding,
            lexical_query=lexical_query,
            filters=filters,
        )

    def classify_intent(self, question: str) -> QueryIntent:
        """Classify the intent of a developer question.

        Uses deterministic keyword-based classification. The same input
        always produces the same output.

        Args:
            question: The developer's natural language question.

        Returns:
            A QueryIntent enum value indicating the classified intent.

        Raises:
            ValueError: If the question is empty or exceeds 1000 characters.
        """
        self._validate_question(question)

        normalized = question.lower()

        # Check intents in priority order (most specific first)
        if self._matches_keywords(normalized, _DEBUGGING_KEYWORDS):
            return QueryIntent.DEBUGGING

        if self._matches_keywords(normalized, _API_USAGE_KEYWORDS):
            return QueryIntent.API_USAGE

        if self._matches_keywords(normalized, _ARCHITECTURE_KEYWORDS):
            return QueryIntent.ARCHITECTURE

        if self._matches_keywords(normalized, _DOCUMENTATION_KEYWORDS):
            return QueryIntent.DOCUMENTATION

        if self._matches_keywords(normalized, _CODE_LOOKUP_KEYWORDS):
            return QueryIntent.CODE_LOOKUP

        # Default to CODE_LOOKUP if no specific pattern matches
        return QueryIntent.CODE_LOOKUP

    def expand_query(self, question: str, intent: QueryIntent) -> list[str]:
        """Expand a developer question into additional search terms based on intent.

        Uses code-aware tokenization to extract base tokens, then adds
        intent-specific patterns:
        - CODE_LOOKUP: function signature patterns (def, function, func, class, call)
        - API_USAGE: HTTP method/endpoint patterns (@app.get, @app.post, router, endpoint)
        - DOCUMENTATION: doc-related keywords (readme, docs, guide, tutorial, example)
        - ARCHITECTURE: structural terms (import, module, package, dependency, config)

        Args:
            question: The developer's natural language question.
            intent: The classified QueryIntent for the question.

        Returns:
            A list of expanded query terms including base tokens and
            intent-specific patterns.
        """
        base_tokens = code_aware_tokenize(question)
        expanded = list(base_tokens)

        if intent == QueryIntent.CODE_LOOKUP:
            for token in base_tokens:
                expanded.extend([
                    f"def {token}",
                    f"function {token}",
                    f"func {token}",
                    f"class {token}",
                    f"{token}()",
                ])
        elif intent == QueryIntent.API_USAGE:
            for token in base_tokens:
                expanded.extend([
                    f'@app.get("/{token}")',
                    f'@app.post("/{token}")',
                    f"router.{token}",
                    f"endpoint {token}",
                ])
        elif intent == QueryIntent.DOCUMENTATION:
            expanded.extend(["readme", "docs", "guide", "tutorial", "example"])
        elif intent == QueryIntent.ARCHITECTURE:
            expanded.extend(["import", "module", "package", "dependency", "config"])

        return expanded

    def _validate_question(self, question: str) -> None:
        """Validate that the question meets length requirements.

        Args:
            question: The question to validate.

        Raises:
            ValueError: If the question is empty or exceeds 1000 characters.
        """
        if not question or not question.strip():
            raise ValueError(
                "Question must be between 1 and 1000 characters"
            )
        if len(question) > MAX_QUESTION_LENGTH:
            raise ValueError(
                "Question must be between 1 and 1000 characters"
            )

    def _matches_keywords(self, text: str, keywords: list[str]) -> bool:
        """Check if text contains any of the given keywords.

        Args:
            text: Lowercased text to search in.
            keywords: List of keyword patterns to look for.

        Returns:
            True if any keyword is found in the text.
        """
        for keyword in keywords:
            if keyword in text:
                return True
        return False
