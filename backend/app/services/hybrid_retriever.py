"""Hybrid retriever combining BM25 and semantic search with Reciprocal Rank Fusion.

This module implements the HybridRetriever class which orchestrates parallel
BM25 lexical search and semantic vector search, then fuses the results using
the Reciprocal Rank Fusion (RRF) algorithm with configurable weighting.

RRF Formula:
    score(d) = alpha * (1 / (k + rank_bm25(d))) + (1 - alpha) * (1 / (k + rank_semantic(d)))

where:
    - alpha controls the weight between BM25 (alpha) and semantic (1 - alpha) results
    - k is a constant that prevents high scores for top-ranked items (default 60, min 1)
    - rank is the 1-based position in each result list
"""

import asyncio
import fnmatch
import logging
from typing import Any

from app.models.chunk import CodeChunk
from app.models.query import ProcessedQuery, QueryFilters
from app.models.retrieval import RetrievalResult, ScoredChunk
from app.services.bm25_engine import BM25Engine
from app.services.semantic_retriever import SemanticRetriever


logger = logging.getLogger(__name__)


class HybridRetriever:
    """Orchestrates parallel BM25 and semantic searches with RRF fusion.

    Executes both retrieval strategies in parallel, then merges results
    using Reciprocal Rank Fusion with configurable alpha weighting and
    k constant.

    Attributes:
        bm25_engine: The BM25 lexical search engine.
        semantic_retriever: The semantic vector search retriever.
        rrf_k: The RRF constant k (default 60, minimum 1).
    """

    def __init__(
        self,
        bm25_engine: BM25Engine,
        semantic_retriever: SemanticRetriever,
        rrf_k: int = 60,
    ) -> None:
        """Initialize the HybridRetriever.

        Args:
            bm25_engine: BM25Engine instance for lexical search.
            semantic_retriever: SemanticRetriever instance for vector search.
            rrf_k: RRF constant k. Must be >= 1. Default 60.

        Raises:
            ValueError: If rrf_k is less than 1.
        """
        if rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")

        self.bm25_engine = bm25_engine
        self.semantic_retriever = semantic_retriever
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        processed_query: ProcessedQuery,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> list[RetrievalResult]:
        """Execute hybrid retrieval with configurable BM25/semantic weighting.

        Runs BM25 and semantic search in parallel, each fetching up to
        3 * top_k candidates, then fuses using RRF.

        Args:
            processed_query: The processed query containing lexical query
                and embedding vector.
            top_k: Maximum number of results to return (default 10).
            alpha: Weight for BM25 results (0.0 to 1.0, default 0.5).
                   Semantic weight is (1 - alpha).

        Returns:
            A list of at most top_k RetrievalResult objects sorted by
            fused_score in strictly descending order.

        Raises:
            ValueError: If alpha is not in [0.0, 1.0] or top_k < 1.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        candidate_count = top_k * 3

        # Execute BM25 and semantic search in parallel with graceful degradation
        bm25_results, semantic_results = await self._parallel_search(
            processed_query, candidate_count
        )

        # Fuse results using RRF
        fused_results = self.reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=alpha
        )

        # Apply post-fusion filters if specified
        if processed_query.filters:
            fused_results = self._apply_filters(fused_results, processed_query.filters)

        # Return at most top_k results
        return fused_results[:top_k]

    async def _parallel_search(
        self,
        processed_query: ProcessedQuery,
        candidate_count: int,
    ) -> tuple[list[ScoredChunk], list[ScoredChunk]]:
        """Execute BM25 and semantic search in parallel with error handling.

        If one retriever fails, the other's results are still returned.

        Args:
            processed_query: The structured query.
            candidate_count: Number of candidates to fetch from each retriever.

        Returns:
            A tuple of (bm25_results, semantic_results). Either list may be
            empty if the corresponding retriever failed.
        """

        async def run_bm25() -> list[ScoredChunk]:
            """Run BM25 search using expanded terms or lexical query."""
            tokens = processed_query.expanded_terms
            if not tokens and processed_query.lexical_query:
                tokens = processed_query.lexical_query.split()
            return self.bm25_engine.search(tokens, top_k=candidate_count)

        async def run_semantic() -> list[ScoredChunk]:
            """Run semantic search using query embedding."""
            if not processed_query.query_embedding:
                return []
            return self.semantic_retriever.search(
                processed_query.query_embedding, top_k=candidate_count
            )

        # Run both searches in parallel, handle failures gracefully
        results = await asyncio.gather(
            run_bm25(), run_semantic(), return_exceptions=True
        )

        bm25_results: list[ScoredChunk] = []
        semantic_results: list[ScoredChunk] = []

        if isinstance(results[0], BaseException):
            logger.warning(f"BM25 search failed: {results[0]}")
        else:
            bm25_results = results[0]

        if isinstance(results[1], BaseException):
            logger.warning(f"Semantic search failed: {results[1]}")
        else:
            semantic_results = results[1]

        return bm25_results, semantic_results

    def reciprocal_rank_fusion(
        self,
        bm25_results: list[ScoredChunk],
        semantic_results: list[ScoredChunk],
        alpha: float = 0.5,
    ) -> list[RetrievalResult]:
        """Fuse two ranked lists using the RRF algorithm.

        RRF formula:
            score(d) = alpha * (1 / (k + rank_bm25(d))) + (1 - alpha) * (1 / (k + rank_semantic(d)))

        Args:
            bm25_results: Ranked list of BM25 search results.
            semantic_results: Ranked list of semantic search results.
            alpha: Weight for BM25 contribution (0.0 to 1.0, default 0.5).

        Returns:
            List of RetrievalResult objects sorted by fused_score in strictly
            descending order with provenance metadata.
        """
        k = self.rrf_k

        fused_scores: dict[str, float] = {}
        chunk_map: dict[str, Any] = {}
        bm25_rank_map: dict[str, int] = {}
        semantic_rank_map: dict[str, int] = {}

        # Process BM25 results (1-based ranking)
        for rank_idx, scored_chunk in enumerate(bm25_results):
            chunk_id = scored_chunk.chunk.id
            rank = rank_idx + 1  # 1-based rank
            rrf_contribution = alpha * (1.0 / (k + rank))

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + rrf_contribution
            chunk_map[chunk_id] = scored_chunk.chunk
            bm25_rank_map[chunk_id] = rank

        # Process semantic results (1-based ranking)
        for rank_idx, scored_chunk in enumerate(semantic_results):
            chunk_id = scored_chunk.chunk.id
            rank = rank_idx + 1  # 1-based rank
            rrf_contribution = (1.0 - alpha) * (1.0 / (k + rank))

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + rrf_contribution
            chunk_map[chunk_id] = scored_chunk.chunk
            semantic_rank_map[chunk_id] = rank

        # Sort by fused score in strictly descending order
        sorted_ids = sorted(
            fused_scores.keys(),
            key=lambda cid: fused_scores[cid],
            reverse=True,
        )

        # Build RetrievalResult objects with provenance metadata
        results: list[RetrievalResult] = []
        for chunk_id in sorted_ids:
            score = fused_scores[chunk_id]
            # Only include results with positive fused_score (per Requirement 8.5)
            if score <= 0.0:
                continue

            result = RetrievalResult(
                chunk=chunk_map[chunk_id],
                fused_score=score,
                bm25_rank=bm25_rank_map.get(chunk_id),
                semantic_rank=semantic_rank_map.get(chunk_id),
                context_snippet=chunk_map[chunk_id].content,
            )
            results.append(result)

        return results

    def _apply_filters(
        self, results: list[RetrievalResult], filters: QueryFilters
    ) -> list[RetrievalResult]:
        """Apply post-fusion filters to retrieval results.

        Filters combine with AND logic — all specified filters must be
        satisfied simultaneously for a result to be included.

        Args:
            results: The fused retrieval results to filter.
            filters: The query filters to apply.

        Returns:
            Filtered list of RetrievalResult. Returns empty list if all
            chunks are excluded by filters.
        """
        # If no filters are specified, return results unchanged
        if (
            filters.languages is None
            and filters.file_patterns is None
            and filters.chunk_types is None
            and filters.repo_ids is None
        ):
            return results

        filtered: list[RetrievalResult] = []

        for result in results:
            chunk = result.chunk

            # Language filter: case-insensitive comparison
            if filters.languages is not None:
                languages_lower = [lang.lower() for lang in filters.languages]
                if chunk.language.lower() not in languages_lower:
                    continue

            # File pattern filter: glob matching using fnmatch
            if filters.file_patterns is not None:
                matches_any_pattern = any(
                    fnmatch.fnmatch(chunk.file_path, pattern)
                    for pattern in filters.file_patterns
                )
                if not matches_any_pattern:
                    continue

            # Chunk type filter: exact match against ChunkType enum values
            if filters.chunk_types is not None:
                if chunk.chunk_type not in filters.chunk_types:
                    continue

            # Repo ID filter: exact match
            if filters.repo_ids is not None:
                if chunk.repo_id not in filters.repo_ids:
                    continue

            filtered.append(result)

        return filtered
