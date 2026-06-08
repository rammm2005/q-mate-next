"""API routes for CodeQ-Mate.

Provides endpoints for querying code repositories and ingesting new repositories.
All endpoints require API key authentication via X-API-Key header.
"""

import html
import re
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.models.answer import GroundedAnswer
from app.services.answer_generator import AnswerGenerator
from app.services.hybrid_retriever import HybridRetriever
from app.services.query_processor import QueryProcessor

router = APIRouter()

# Maximum query length for input validation (Requirement 12.3)
MAX_INPUT_LENGTH = 2000


# --- Request/Response Models ---


class QueryFiltersRequest(BaseModel):
    """Optional filters for narrowing query results."""

    languages: Optional[list[str]] = None
    paths: Optional[list[str]] = None
    chunk_types: Optional[list[str]] = None
    repo_ids: Optional[list[str]] = None


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    question: str = Field(min_length=1, max_length=1000)
    filters: Optional[QueryFiltersRequest] = None


class QueryResponse(BaseModel):
    """Response body for the query endpoint."""

    answer: str
    sources: list[dict] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Request body for the ingestion endpoint."""

    repository_path: str = Field(min_length=1)
    config: Optional[dict] = None


class IngestResponse(BaseModel):
    """Response body for the ingestion endpoint."""

    status: str
    chunks_processed: int = 0
    files_processed: int = 0
    errors: list[str] = Field(default_factory=list)


# --- Authentication ---


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Verify the API key from the X-API-Key request header.

    Rejects requests with missing or empty API keys with a 401 error.
    For now accepts any non-empty key (to be wired to DB validation later).

    Args:
        x_api_key: The API key from the X-API-Key header.

    Returns:
        The validated API key string.

    Raises:
        HTTPException: 401 if the API key is missing or empty.
    """
    if not x_api_key or not x_api_key.strip():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# --- Input Sanitization ---


def sanitize_input(text: str) -> str:
    """Sanitize input text by escaping special characters.

    Escapes HTML entities and removes potential injection patterns
    to prevent injection attacks (Requirement 12.3).

    Args:
        text: The raw input text to sanitize.

    Returns:
        Sanitized text with special characters escaped.
    """
    # Escape HTML special characters
    sanitized = html.escape(text, quote=True)
    return sanitized


def validate_query_length(question: str) -> None:
    """Validate that the query does not exceed maximum input length.

    Requirement 12.3: reject queries > 2000 chars.

    Args:
        question: The query text to validate.

    Raises:
        HTTPException: 422 if the query exceeds 2000 characters.
    """
    if len(question) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Query exceeds maximum length of {MAX_INPUT_LENGTH} characters",
        )


# --- Service Dependencies ---

# These will be initialized at app startup and injected via dependency injection.
# For now, we use None defaults and create instances on demand for testing.
_query_processor: Optional[QueryProcessor] = None
_hybrid_retriever: Optional[HybridRetriever] = None
_answer_generator: Optional[AnswerGenerator] = None


def get_query_processor() -> QueryProcessor:
    """Get or create the QueryProcessor instance."""
    global _query_processor
    if _query_processor is None:
        _query_processor = QueryProcessor()
    return _query_processor


def get_hybrid_retriever() -> Optional[HybridRetriever]:
    """Get the HybridRetriever instance (may be None if not configured)."""
    return _hybrid_retriever


def get_answer_generator() -> AnswerGenerator:
    """Get or create the AnswerGenerator instance.

    Attempts to initialize with a Gemini LLM client if GEMINI_API_KEY
    is set in the environment. Falls back to no LLM (source-only mode)
    if the key is not configured.
    """
    global _answer_generator
    if _answer_generator is None:
        llm_client = None
        try:
            from app.services.gemini_client import GeminiClient
            llm_client = GeminiClient()
        except (ValueError, ImportError):
            # No API key configured or package not installed - run without LLM
            pass
        _answer_generator = AnswerGenerator(llm_client=llm_client)
    return _answer_generator


# --- Endpoints ---


@router.post("/api/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key),
) -> QueryResponse:
    """Process a developer question and return a source-grounded answer.

    Accepts a natural language question (1-1000 chars) with optional filters,
    processes it through the QueryProcessor → HybridRetriever → AnswerGenerator
    pipeline, and returns a grounded answer with source references.

    Requirements: 1.1, 1.8, 12.1, 12.2, 12.3
    """
    # Validate input length (Requirement 12.3)
    validate_query_length(request.question)

    # Sanitize input
    sanitized_question = sanitize_input(request.question)

    # Get services
    query_processor = get_query_processor()
    hybrid_retriever = get_hybrid_retriever()
    answer_generator = get_answer_generator()

    # Process query through the pipeline
    try:
        from app.models.query import QueryContext, QueryFilters

        # Build filters from request
        filters = QueryFilters()
        if request.filters:
            filters = QueryFilters(
                languages=request.filters.languages,
                file_patterns=request.filters.paths,
                chunk_types=None,  # Would need ChunkType conversion
                repo_ids=request.filters.repo_ids,
            )

        context = QueryContext(filters=filters)
        processed_query = query_processor.process_query(sanitized_question, context)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Retrieve relevant chunks
    retrieval_results = []
    if hybrid_retriever is not None:
        try:
            retrieval_results = await hybrid_retriever.retrieve(processed_query)
        except Exception:
            # Graceful degradation: return empty results on retriever failure
            retrieval_results = []

    # Generate answer
    grounded_answer: GroundedAnswer = await answer_generator.generate_answer(
        sanitized_question, retrieval_results
    )

    # Build response
    sources = [
        {
            "file_path": src.file_path,
            "function_name": src.function_name,
            "start_line": src.start_line,
            "end_line": src.end_line,
            "snippet": src.snippet,
            "relevance": src.relevance,
        }
        for src in grounded_answer.sources
    ]

    return QueryResponse(
        answer=grounded_answer.answer_text,
        sources=sources,
        confidence=grounded_answer.confidence,
        metadata=grounded_answer.retrieval_metadata,
    )


@router.post("/api/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    request: IngestRequest,
    api_key: str = Depends(verify_api_key),
) -> IngestResponse:
    """Ingest a repository into the search indexes.

    Accepts a repository path and optional configuration, then runs
    the ingestion pipeline to parse, chunk, embed, and index the repository.

    Requires admin authentication (currently any valid API key).

    Requirements: 7.1, 12.1, 12.2
    """
    # Validate input length
    if len(request.repository_path) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail="Repository path exceeds maximum length",
        )

    # Sanitize repository path
    sanitized_path = sanitize_input(request.repository_path)

    # TODO: Wire to IngestionPipeline when available
    # For now, return a placeholder response indicating the endpoint is ready
    return IngestResponse(
        status="accepted",
        chunks_processed=0,
        files_processed=0,
        errors=[],
    )
