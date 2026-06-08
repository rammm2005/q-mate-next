"""Data models for code chunks with Pydantic v2 validation."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

import tiktoken


class ChunkType(str, Enum):
    """Types of code elements that can be chunked."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    DOCUMENTATION = "documentation"
    MODULE = "module"
    CONFIG = "config"
    API_ENDPOINT = "api_endpoint"


SUPPORTED_LANGUAGES = {"typescript", "javascript", "python", "php", "go"}

MAX_CONTENT_TOKENS = 8192


def _estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text using tiktoken cl100k_base encoding."""
    if not text:
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


class ChunkMetadata(BaseModel):
    """Additional structural metadata for a code chunk."""

    function_name: Optional[str] = None
    class_name: Optional[str] = None
    module_name: Optional[str] = None
    imports: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    docstring: Optional[str] = None
    signatures: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class CodeChunk(BaseModel):
    """A semantically meaningful segment of source code or documentation.

    Validation Rules:
    - content must be non-empty and under 8192 tokens
    - start_line must be >= 1 and <= end_line
    - file_path must be a relative path without path traversal sequences (..)
    - language must be one of: typescript, javascript, python, php, go
    """

    id: str
    repo_id: str
    file_path: str
    language: str
    chunk_type: ChunkType
    content: str
    start_line: int = Field(ge=1)
    end_line: int
    parent_id: Optional[str] = None
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)

    @field_validator("content")
    @classmethod
    def content_must_be_non_empty(cls, v: str) -> str:
        """Validate that content is non-empty."""
        if not v or not v.strip():
            raise ValueError("content must be non-empty")
        return v

    @field_validator("file_path")
    @classmethod
    def file_path_must_be_valid_relative(cls, v: str) -> str:
        """Validate file_path is relative and has no path traversal."""
        if not v or not v.strip():
            raise ValueError("file_path must be non-empty")

        # Reject absolute paths (Unix or Windows style)
        if v.startswith("/") or v.startswith("\\"):
            raise ValueError("file_path must be a relative path, not absolute")

        # Reject Windows drive letters (e.g., C:\, D:/)
        if len(v) >= 2 and v[1] == ":":
            raise ValueError("file_path must be a relative path, not absolute")

        # Reject path traversal sequences
        # Check for '..' as a path component
        parts = v.replace("\\", "/").split("/")
        for part in parts:
            if part == "..":
                raise ValueError(
                    "file_path must not contain path traversal sequences (..)"
                )

        return v

    @field_validator("language")
    @classmethod
    def language_must_be_supported(cls, v: str) -> str:
        """Validate language is one of the supported languages."""
        normalized = v.lower()
        if normalized not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"language must be one of {sorted(SUPPORTED_LANGUAGES)}, "
                f"got '{v}'"
            )
        return normalized

    @model_validator(mode="after")
    def validate_line_range(self) -> "CodeChunk":
        """Validate that start_line <= end_line."""
        if self.start_line > self.end_line:
            raise ValueError(
                f"start_line ({self.start_line}) must be <= end_line ({self.end_line})"
            )
        return self

    @model_validator(mode="after")
    def validate_content_tokens(self) -> "CodeChunk":
        """Validate that content does not exceed MAX_CONTENT_TOKENS."""
        token_count = _estimate_tokens(self.content)
        if token_count > MAX_CONTENT_TOKENS:
            raise ValueError(
                f"content exceeds maximum of {MAX_CONTENT_TOKENS} tokens "
                f"(has {token_count} tokens)"
            )
        return self
