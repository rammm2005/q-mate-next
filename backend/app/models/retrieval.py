"""Data models for retrieval results."""

from typing import Optional

from pydantic import BaseModel, Field

from app.models.chunk import CodeChunk


class ScoredChunk(BaseModel):
    """A code chunk with a relevance score from a single retrieval source."""

    chunk: CodeChunk
    score: float = Field(ge=0.0)
    source: str  # "bm25" or "semantic"


class RetrievalResult(BaseModel):
    """A fused retrieval result combining BM25 and semantic search scores."""

    chunk: CodeChunk
    fused_score: float = Field(gt=0.0)
    bm25_rank: Optional[int] = None  # 1-based rank in BM25 results
    semantic_rank: Optional[int] = None  # 1-based rank in semantic results
    context_snippet: str = ""
