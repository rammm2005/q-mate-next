"""Repository ingestion pipeline for walking and parsing source files.

Handles directory traversal, file type identification, binary detection,
exclusion pattern matching, secret detection, embedding generation,
and index building for repository ingestion.
"""

import os
import re
import fnmatch
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.chunker import CodeChunker
    from app.services.semantic_retriever import SemanticRetriever
    from app.services.bm25_engine import BM25Engine


# Supported language extensions mapping
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".php": "php",
    ".go": "go",
    # Additional text-based files (indexed as their closest supported language)
    ".html": "javascript",
    ".css": "javascript",
    ".scss": "javascript",
    ".json": "javascript",
    ".yaml": "python",
    ".yml": "python",
    ".md": "python",
    ".txt": "python",
    ".sql": "python",
    ".sh": "python",
    ".bat": "python",
    ".xml": "javascript",
    ".env.example": "python",
    ".toml": "python",
    ".cfg": "python",
    ".ini": "python",
    ".rst": "python",
    ".vue": "typescript",
    ".svelte": "typescript",
    ".dart": "python",
    ".rb": "python",
    ".java": "python",
    ".kt": "python",
    ".swift": "python",
    ".rs": "python",
    ".c": "python",
    ".cpp": "python",
    ".h": "python",
    ".cs": "python",
}

# Patterns indicating secrets in file content
SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"(?i)(api[_\-]?key|secret[_\-]?key|password|token)\s*[=:]\s*[\"']?[a-zA-Z0-9_\-]{16,}"
    ),
    re.compile(r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?i)(aws_access_key_id|aws_secret_access_key)\s*="),
    re.compile(
        r"(?i)(database_url|db_password|mysql_password|postgres_password)\s*[=:]\s*\S+"
    ),
]

# File names that typically contain secrets
SECRET_FILENAMES: set[str] = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.staging",
    "credentials",
    "credentials.json",
    "credentials.yaml",
    "credentials.yml",
    "secrets.yaml",
    "secrets.yml",
    "secrets.json",
    ".npmrc",
    ".pypirc",
}

# Default exclusion patterns for common non-source directories and files
DEFAULT_EXCLUSIONS: list[str] = [
    "node_modules/**",
    "__pycache__/**",
    ".git/**",
    "*.pyc",
    "*.min.js",
    "dist/**",
    "build/**",
    ".venv/**",
    "venv/**",
    "*.lock",
    "*.log",
    ".idea/**",
    ".vscode/**",
]


@dataclass
class IngestionConfig:
    """Configuration for the repository ingestion pipeline."""

    languages: list[str] = field(default_factory=lambda: list(set(SUPPORTED_EXTENSIONS.values())))
    exclude_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUSIONS))
    max_chunk_tokens: int = 512
    overlap_tokens: int = 50


@dataclass
class IngestionStats:
    """Statistics returned after repository ingestion."""

    total_files: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    repo_id: str = ""


@dataclass
class ParsedFile:
    """Represents a parsed file from the repository."""

    file_path: str  # Relative path within the repository
    absolute_path: str  # Absolute path on disk
    language: Optional[str]  # Detected language or None for unsupported
    content: str  # File content
    is_supported_language: bool  # Whether the language is in supported set


class IngestionPipeline:
    """Pipeline for ingesting repository files into the search index.

    Walks the directory tree, identifies file types, skips binary files
    and exclusion patterns, detects secrets, and categorizes files by
    supported/unsupported language for downstream processing.

    Supports full ingestion: walk → chunk → embed → index with
    idempotent behavior (re-ingesting the same repo produces identical state).
    """

    def __init__(
        self,
        chunker: "Optional[CodeChunker]" = None,
        semantic_retriever: "Optional[SemanticRetriever]" = None,
        bm25_engine: "Optional[BM25Engine]" = None,
        supabase_client=None,
        repo_id: str = "default",
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            chunker: CodeChunker instance for splitting files into chunks.
            semantic_retriever: SemanticRetriever for embedding generation.
            bm25_engine: BM25Engine for lexical index building.
            supabase_client: Supabase client for storing chunks + embeddings.
            repo_id: Repository identifier for this ingestion.
        """
        self.chunker = chunker
        self.semantic_retriever = semantic_retriever
        self.bm25_engine = bm25_engine
        self.supabase = supabase_client
        self.repo_id = repo_id

    async def ingest_repository(
        self, repo_path: str, config: Optional[IngestionConfig] = None
    ) -> IngestionStats:
        """Full ingestion pipeline: walk → chunk → embed → index.

        Performs idempotent ingestion by deleting existing chunks for the
        same repo_id before inserting new data.

        Args:
            repo_path: Absolute path to the repository root.
            config: Ingestion configuration. Uses defaults if None.

        Returns:
            IngestionStats with counts of processed files, chunks, and embeddings.

        Raises:
            FileNotFoundError: If repo_path does not exist.
            NotADirectoryError: If repo_path is not a directory.
            RuntimeError: If required dependencies (chunker, semantic_retriever,
                         bm25_engine, supabase) are not configured.
        """
        # Validate dependencies
        if self.chunker is None:
            raise RuntimeError("chunker is required for ingest_repository")
        if self.semantic_retriever is None:
            raise RuntimeError("semantic_retriever is required for ingest_repository")
        if self.bm25_engine is None:
            raise RuntimeError("bm25_engine is required for ingest_repository")
        if self.supabase is None:
            raise RuntimeError("supabase_client is required for ingest_repository")

        # 1. Walk repository
        parsed_files = self.walk_repository(repo_path, config)

        # 2. Chunk each file
        all_chunks = []
        for parsed_file in parsed_files:
            language = parsed_file.language or "python"
            file_chunks = self.chunker.chunk_file(
                parsed_file.content, parsed_file.file_path, language
            )
            all_chunks.extend(file_chunks)

        if not all_chunks:
            return IngestionStats(
                total_files=len(parsed_files),
                total_chunks=0,
                total_embeddings=0,
                repo_id=self.repo_id,
            )

        # 3. Generate embeddings in batches
        embeddings = self.semantic_retriever.embed_chunks(all_chunks)

        # 4. Store in Supabase (idempotent: delete existing, then insert)
        await self._store_chunks(all_chunks, embeddings)

        # 5. Build BM25 index
        self.bm25_engine.build_index(all_chunks)

        return IngestionStats(
            total_files=len(parsed_files),
            total_chunks=len(all_chunks),
            total_embeddings=len(embeddings),
            repo_id=self.repo_id,
        )

    async def _store_chunks(
        self, chunks: list, embeddings: list[list[float]]
    ) -> None:
        """Store chunks and their embeddings in Supabase/pgvector.

        Implements idempotent ingestion by first deleting any existing
        chunks for the same repo_id, then inserting the new chunks.

        Args:
            chunks: List of CodeChunk objects to store.
            embeddings: Corresponding embedding vectors for each chunk.
        """
        # Delete existing chunks for this repo (idempotent ingestion)
        self.supabase.table("code_chunks").delete().eq(
            "repo_id", self.repo_id
        ).execute()

        # Insert chunks with embeddings in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            rows = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                row = {
                    "id": chunk.id,
                    "repo_id": chunk.repo_id,
                    "file_path": chunk.file_path,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else chunk.chunk_type,
                    "content": chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "embedding": embedding,
                    "function_name": chunk.metadata.function_name if chunk.metadata else None,
                    "class_name": chunk.metadata.class_name if chunk.metadata else None,
                }
                rows.append(row)

            self.supabase.table("code_chunks").insert(rows).execute()

    def walk_repository(
        self, repo_path: str, config: Optional[IngestionConfig] = None
    ) -> list[ParsedFile]:
        """Walk the directory tree and identify parseable files.

        Args:
            repo_path: Absolute path to the repository root.
            config: Ingestion configuration. Uses defaults if None.

        Returns:
            List of ParsedFile objects for all eligible files.

        Raises:
            FileNotFoundError: If repo_path does not exist.
            NotADirectoryError: If repo_path is not a directory.
        """
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        if not os.path.isdir(repo_path):
            raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

        if config is None:
            config = IngestionConfig()

        parsed_files: list[ParsedFile] = []

        for root, dirs, files in os.walk(repo_path):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [
                d
                for d in dirs
                if not self._matches_exclusion(
                    os.path.relpath(os.path.join(root, d), repo_path),
                    config.exclude_patterns,
                )
            ]

            for filename in files:
                absolute_path = os.path.join(root, filename)
                relative_path = os.path.relpath(absolute_path, repo_path)
                # Normalize to forward slashes for consistency
                relative_path = relative_path.replace("\\", "/")

                # Skip files matching exclusion patterns
                if self._matches_exclusion(relative_path, config.exclude_patterns):
                    continue

                # Skip binary files
                if self._is_binary(absolute_path):
                    continue

                # Skip files with secret filenames
                if os.path.basename(absolute_path).lower() in {
                    s.lower() for s in SECRET_FILENAMES
                }:
                    continue

                # Skip files containing secrets
                if self._contains_secrets(absolute_path):
                    continue

                # Detect language
                language = self._detect_language(absolute_path)
                is_supported = language is not None

                # Read file content
                try:
                    with open(absolute_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except (OSError, IOError):
                    continue

                # Skip empty files
                if not content.strip():
                    continue

                parsed_files.append(
                    ParsedFile(
                        file_path=relative_path,
                        absolute_path=absolute_path,
                        language=language,
                        content=content,
                        is_supported_language=is_supported,
                    )
                )

        return parsed_files

    def _is_binary(self, file_path: str) -> bool:
        """Check if a file is binary by reading initial bytes.

        Reads up to 8192 bytes and checks for null bytes which indicate
        binary content.

        Args:
            file_path: Absolute path to the file.

        Returns:
            True if the file appears to be binary, False otherwise.
        """
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(8192)
            # Check for null bytes which indicate binary
            if b"\x00" in chunk:
                return True
            return False
        except (OSError, IOError):
            # If we can't read the file, treat it as binary (skip it)
            return True

    def _matches_exclusion(self, relative_path: str, patterns: list[str]) -> bool:
        """Check if a relative path matches any exclusion pattern.

        Supports glob-style patterns including ** for recursive matching.

        Args:
            relative_path: Path relative to repository root.
            patterns: List of glob patterns to match against.

        Returns:
            True if the path matches any exclusion pattern.
        """
        # Normalize path separators
        normalized_path = relative_path.replace("\\", "/")

        for pattern in patterns:
            normalized_pattern = pattern.replace("\\", "/")

            # Handle ** patterns for recursive directory matching
            if "**" in normalized_pattern:
                # Convert ** pattern to regex
                # e.g., "node_modules/**" should match "node_modules/foo/bar.js"
                regex_pattern = normalized_pattern.replace(".", r"\.")
                regex_pattern = regex_pattern.replace("**", ".*")
                regex_pattern = regex_pattern.replace("*", "[^/]*")
                # Fix: after replacing ** with .*, the second replace would
                # corrupt it. Let's use a different approach.
                regex_pattern = self._glob_to_regex(normalized_pattern)
                if re.match(regex_pattern, normalized_path):
                    return True
            else:
                # Simple glob matching against the full path
                if fnmatch.fnmatch(normalized_path, normalized_pattern):
                    return True
                # Also match against just the filename for patterns like "*.pyc"
                filename = os.path.basename(normalized_path)
                if fnmatch.fnmatch(filename, normalized_pattern):
                    return True

        return False

    def _glob_to_regex(self, pattern: str) -> str:
        """Convert a glob pattern to a regex pattern.

        Args:
            pattern: Glob pattern with support for ** and *.

        Returns:
            Regex pattern string.
        """
        # Escape regex special chars except * and ?
        result = ""
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if c == "*":
                if i + 1 < len(pattern) and pattern[i + 1] == "*":
                    # ** matches any path
                    result += ".*"
                    i += 2
                    # Skip trailing slash after **
                    if i < len(pattern) and pattern[i] == "/":
                        i += 1
                        result += "(?:/)?"
                    continue
                else:
                    # * matches anything except /
                    result += "[^/]*"
            elif c == "?":
                result += "[^/]"
            elif c in r"\.[]{}()+^$|":
                result += "\\" + c
            else:
                result += c
            i += 1

        return "^" + result + "$"

    def _contains_secrets(self, file_path: str) -> bool:
        """Detect if a file contains secret patterns.

        Reads the file and scans for patterns that indicate secrets
        such as API keys, private keys, AWS credentials, etc.

        Args:
            file_path: Absolute path to the file.

        Returns:
            True if the file appears to contain secrets.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (OSError, IOError):
            return False

        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                return True

        return False

    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension.

        Args:
            file_path: Path to the file (absolute or relative).

        Returns:
            Language string if supported, None otherwise.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        return SUPPORTED_EXTENSIONS.get(ext)
