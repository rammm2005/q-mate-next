"""In-memory search engine combining BM25 + chunking for demo mode.

This module provides a self-contained search engine that works without
Supabase. It uses BM25 for retrieval and the code chunker for parsing.
Perfect for local demo and testing.
"""

from dataclasses import dataclass, field

from app.models.chunk import CodeChunk
from app.models.retrieval import RetrievalResult, ScoredChunk
from app.services.bm25_engine import BM25Engine
from app.services.chunker import CodeChunker
from app.services.ingestion import IngestionPipeline, IngestionConfig
from app.services.tokenizer import code_aware_tokenize


@dataclass
class IngestStats:
    """Statistics from a repository ingestion."""
    repo_name: str
    total_files: int
    total_chunks: int
    languages_found: list[str] = field(default_factory=list)


class SearchEngine:
    """Self-contained search engine for CodeQ-Mate demo.

    Combines repository parsing, code chunking, and BM25 search
    in memory. No database required.
    """

    def __init__(self) -> None:
        self.bm25 = BM25Engine()
        self.chunker = CodeChunker(max_chunk_size=512, overlap=50)
        self.ingestion = IngestionPipeline()
        self.chunks: list[CodeChunk] = []
        self.is_indexed: bool = False
        self.repo_name: str = ""
        self.repo_path: str = ""  # Store absolute path to repo

    def ingest_local_repo(self, repo_path: str, repo_name: str = "default") -> IngestStats:
        """Ingest a local repository into the in-memory search index.

        Args:
            repo_path: Path to the local repository folder.
            repo_name: Name identifier for the repository.

        Returns:
            IngestStats with counts of processed files and chunks.
        """
        self.repo_name = repo_name
        self.repo_path = repo_path  # Store the path

        # Walk the repository
        config = IngestionConfig()
        parsed_files = self.ingestion.walk_repository(repo_path, config)

        # Chunk each file
        all_chunks: list[CodeChunk] = []
        languages_found: set[str] = set()

        for parsed_file in parsed_files:
            lang = parsed_file.language or "python"
            if parsed_file.language:
                languages_found.add(parsed_file.language)

            try:
                file_chunks = self.chunker.chunk_file(
                    content=parsed_file.content,
                    file_path=parsed_file.file_path,
                    language=lang,
                )
                all_chunks.extend(file_chunks)
            except Exception:
                # Skip files that fail to chunk
                continue

        # Build BM25 index
        self.chunks = all_chunks
        self.bm25.build_index(all_chunks)
        self.is_indexed = True

        return IngestStats(
            repo_name=repo_name,
            total_files=len(parsed_files),
            total_chunks=len(all_chunks),
            languages_found=sorted(languages_found),
        )

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Search the indexed repository using BM25.

        Args:
            query: Natural language query from the user.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievalResult sorted by relevance score.
        """
        if not self.is_indexed or not self.chunks:
            return []

        # Tokenize the query using code-aware tokenization
        query_tokens = code_aware_tokenize(query)

        if not query_tokens:
            return []

        # Search BM25
        scored_chunks = self.bm25.search(query_tokens, top_k=top_k)

        # Convert to RetrievalResult
        results: list[RetrievalResult] = []
        for i, sc in enumerate(scored_chunks):
            results.append(RetrievalResult(
                chunk=sc.chunk,
                fused_score=sc.score if sc.score > 0 else 0.001,
                bm25_rank=i + 1,
                semantic_rank=None,
                context_snippet=sc.chunk.content,
            ))

        return results

    def get_status(self) -> dict:
        """Get current engine status."""
        return {
            "is_indexed": self.is_indexed,
            "repo_name": self.repo_name,
            "total_chunks": len(self.chunks),
        }


# Global singleton instance
_engine: SearchEngine | None = None


def get_search_engine() -> SearchEngine:
    """Get or create the global SearchEngine singleton."""
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine
