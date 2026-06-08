"""Data models for processed queries."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.models.chunk import ChunkType


class QueryIntent(str, Enum):
    """Classification of developer question type."""

    CODE_LOOKUP = "code_lookup"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    DEBUGGING = "debugging"
    API_USAGE = "api_usage"


class QueryFilters(BaseModel):
    """Optional filters to narrow retrieval results."""

    languages: Optional[list[str]] = None
    file_patterns: Optional[list[str]] = None
    chunk_types: Optional[list[ChunkType]] = None
    repo_ids: Optional[list[str]] = None


class QueryContext(BaseModel):
    """Optional context for query processing.

    Provides additional context such as filters or session information
    that can influence how a query is processed.
    """

    filters: QueryFilters = Field(default_factory=QueryFilters)


class ProcessedQuery(BaseModel):
    """A developer question that has been processed into structured form."""

    original_question: str
    intent: QueryIntent
    expanded_terms: list[str] = Field(default_factory=list)
    query_embedding: list[float] = Field(default_factory=list)
    lexical_query: str = ""
    filters: QueryFilters = Field(default_factory=QueryFilters)
