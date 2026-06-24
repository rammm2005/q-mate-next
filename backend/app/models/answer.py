"""Data models for grounded answers."""

from typing import Optional

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """A reference to a specific source location backing an answer."""

    file_path: str
    function_name: Optional[str] = None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    snippet: str = ""
    relevance: float = Field(ge=0.0)  # RRF fusion can produce scores > 1.0


class GroundedAnswer(BaseModel):
    """A generated answer grounded in retrieved source code and documentation."""

    answer_text: str
    sources: list[SourceReference] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    retrieval_metadata: dict = Field(default_factory=dict)
