"""Unit tests for HybridRetriever with Reciprocal Rank Fusion.

Tests cover:
- RRF algorithm correctness (scoring, ranking, provenance)
- Parallel execution of BM25 and semantic search
- Alpha weighting between BM25 and semantic results
- Graceful degradation when one retriever fails
- Edge cases (empty results, single source, top_k limits)
- Parameter validation (alpha, top_k, rrf_k)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.chunk import CodeChunk, ChunkMetadata, ChunkType
from app.models.query import ProcessedQuery, QueryIntent, QueryFilters
from app.models.retrieval import RetrievalResult, ScoredChunk
from app.services.bm25_engine import BM25Engine
from app.services.hybrid_retriever import HybridRetriever
from app.services.semantic_retriever import SemanticRetriever


def make_chunk(
    chunk_id: str,
    content: str = "def example(): pass",
    language: str = "python",
    file_path: str = "src/main.py",
    repo_id: str = "repo-1",
) -> CodeChunk:
    """Helper to create a CodeChunk for testing."""
    return CodeChunk(
        id=chunk_id,
        repo_id=repo_id,
        file_path=file_path,
        language=language,
        chunk_type=ChunkType.FUNCTION,
        content=content,
        start_line=1,
        end_line=10,
        metadata=ChunkMetadata(),
    )


def make_scored_chunk(
    chunk_id: str, score: float, source: str = "bm25"
) -> ScoredChunk:
    """Helper to create a ScoredChunk for testing."""
    return ScoredChunk(
        chunk=make_chunk(chunk_id),
        score=score,
        source=source,
    )


def make_processed_query(
    question: str = "how does get_user work?",
    lexical_query: str = "get user",
    expanded_terms: list[str] | None = None,
    query_embedding: list[float] | None = None,
) -> ProcessedQuery:
    """Helper to create a ProcessedQuery for testing."""
    return ProcessedQuery(
        original_question=question,
        intent=QueryIntent.CODE_LOOKUP,
        expanded_terms=expanded_terms or ["get", "user"],
        query_embedding=query_embedding or [0.1] * 384,
        lexical_query=lexical_query,
        filters=QueryFilters(),
    )


class TestHybridRetrieverInit:
    """Tests for HybridRetriever initialization."""

    def test_default_rrf_k(self):
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        retriever = HybridRetriever(bm25, semantic)
        assert retriever.rrf_k == 60

    def test_custom_rrf_k(self):
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        retriever = HybridRetriever(bm25, semantic, rrf_k=30)
        assert retriever.rrf_k == 30

    def test_rrf_k_minimum_one(self):
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        retriever = HybridRetriever(bm25, semantic, rrf_k=1)
        assert retriever.rrf_k == 1

    def test_rrf_k_below_minimum_raises(self):
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        with pytest.raises(ValueError, match="rrf_k must be >= 1"):
            HybridRetriever(bm25, semantic, rrf_k=0)

    def test_rrf_k_negative_raises(self):
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        with pytest.raises(ValueError, match="rrf_k must be >= 1"):
            HybridRetriever(bm25, semantic, rrf_k=-5)


class TestReciprocalRankFusion:
    """Tests for the RRF algorithm implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bm25 = MagicMock(spec=BM25Engine)
        self.semantic = MagicMock(spec=SemanticRetriever)
        self.retriever = HybridRetriever(self.bm25, self.semantic, rrf_k=60)

    def test_empty_both_lists(self):
        results = self.retriever.reciprocal_rank_fusion([], [])
        assert results == []

    def test_empty_bm25_list(self):
        semantic_results = [
            make_scored_chunk("c1", 0.9, "semantic"),
            make_scored_chunk("c2", 0.8, "semantic"),
        ]
        results = self.retriever.reciprocal_rank_fusion([], semantic_results)
        assert len(results) == 2
        # All results should have bm25_rank=None
        for r in results:
            assert r.bm25_rank is None
            assert r.semantic_rank is not None

    def test_empty_semantic_list(self):
        bm25_results = [
            make_scored_chunk("c1", 5.0, "bm25"),
            make_scored_chunk("c2", 3.0, "bm25"),
        ]
        results = self.retriever.reciprocal_rank_fusion(bm25_results, [])
        assert len(results) == 2
        # All results should have semantic_rank=None
        for r in results:
            assert r.semantic_rank is None
            assert r.bm25_rank is not None

    def test_rrf_score_formula(self):
        """Verify RRF score calculation matches the formula."""
        retriever = HybridRetriever(self.bm25, self.semantic, rrf_k=60)
        bm25_results = [make_scored_chunk("c1", 5.0, "bm25")]
        semantic_results = [make_scored_chunk("c1", 0.9, "semantic")]

        results = retriever.reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.5
        )

        assert len(results) == 1
        # score = alpha * (1/(k+1)) + (1-alpha) * (1/(k+1))
        # = 0.5 * (1/61) + 0.5 * (1/61) = 1/61
        expected_score = 0.5 * (1.0 / 61) + 0.5 * (1.0 / 61)
        assert abs(results[0].fused_score - expected_score) < 1e-10

    def test_rrf_score_different_ranks(self):
        """Verify scores differ based on rank position."""
        bm25_results = [
            make_scored_chunk("c1", 5.0, "bm25"),
            make_scored_chunk("c2", 3.0, "bm25"),
        ]
        semantic_results = [
            make_scored_chunk("c2", 0.9, "semantic"),
            make_scored_chunk("c1", 0.8, "semantic"),
        ]

        results = self.retriever.reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.5
        )

        # c1: bm25 rank 1, semantic rank 2
        # score = 0.5*(1/61) + 0.5*(1/62)
        c1_expected = 0.5 * (1.0 / 61) + 0.5 * (1.0 / 62)

        # c2: bm25 rank 2, semantic rank 1
        # score = 0.5*(1/62) + 0.5*(1/61)
        c2_expected = 0.5 * (1.0 / 62) + 0.5 * (1.0 / 61)

        # Both should have equal scores (symmetric case)
        assert abs(c1_expected - c2_expected) < 1e-10

        # Find results
        c1_result = next(r for r in results if r.chunk.id == "c1")
        c2_result = next(r for r in results if r.chunk.id == "c2")
        assert abs(c1_result.fused_score - c1_expected) < 1e-10
        assert abs(c2_result.fused_score - c2_expected) < 1e-10

    def test_results_sorted_descending(self):
        """Results must be sorted by fused_score in strictly descending order."""
        bm25_results = [
            make_scored_chunk("c1", 5.0, "bm25"),
            make_scored_chunk("c2", 4.0, "bm25"),
            make_scored_chunk("c3", 3.0, "bm25"),
        ]
        semantic_results = [
            make_scored_chunk("c3", 0.9, "semantic"),
            make_scored_chunk("c1", 0.7, "semantic"),
        ]

        results = self.retriever.reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.5
        )

        for i in range(len(results) - 1):
            assert results[i].fused_score >= results[i + 1].fused_score

    def test_provenance_metadata_bm25_rank(self):
        """BM25 rank should be 1-based and reflect position in BM25 results."""
        bm25_results = [
            make_scored_chunk("c1", 5.0, "bm25"),
            make_scored_chunk("c2", 4.0, "bm25"),
            make_scored_chunk("c3", 3.0, "bm25"),
        ]

        results = self.retriever.reciprocal_rank_fusion(bm25_results, [])
        c1 = next(r for r in results if r.chunk.id == "c1")
        c2 = next(r for r in results if r.chunk.id == "c2")
        c3 = next(r for r in results if r.chunk.id == "c3")

        assert c1.bm25_rank == 1
        assert c2.bm25_rank == 2
        assert c3.bm25_rank == 3
        assert c1.semantic_rank is None
        assert c2.semantic_rank is None
        assert c3.semantic_rank is None

    def test_provenance_metadata_semantic_rank(self):
        """Semantic rank should be 1-based and reflect position in semantic results."""
        semantic_results = [
            make_scored_chunk("c1", 0.95, "semantic"),
            make_scored_chunk("c2", 0.80, "semantic"),
        ]

        results = self.retriever.reciprocal_rank_fusion([], semantic_results)
        c1 = next(r for r in results if r.chunk.id == "c1")
        c2 = next(r for r in results if r.chunk.id == "c2")

        assert c1.semantic_rank == 1
        assert c2.semantic_rank == 2
        assert c1.bm25_rank is None
        assert c2.bm25_rank is None

    def test_provenance_metadata_both_ranks(self):
        """Document in both lists should have both ranks populated."""
        bm25_results = [
            make_scored_chunk("c1", 5.0, "bm25"),
            make_scored_chunk("c2", 4.0, "bm25"),
        ]
        semantic_results = [
            make_scored_chunk("c2", 0.9, "semantic"),
            make_scored_chunk("c1", 0.8, "semantic"),
        ]

        results = self.retriever.reciprocal_rank_fusion(bm25_results, semantic_results)
        c1 = next(r for r in results if r.chunk.id == "c1")
        c2 = next(r for r in results if r.chunk.id == "c2")

        assert c1.bm25_rank == 1
        assert c1.semantic_rank == 2
        assert c2.bm25_rank == 2
        assert c2.semantic_rank == 1

    def test_alpha_one_uses_only_bm25(self):
        """alpha=1.0 means only BM25 contributes to scores."""
        bm25_results = [make_scored_chunk("c1", 5.0, "bm25")]
        semantic_results = [make_scored_chunk("c2", 0.9, "semantic")]

        results = self.retriever.reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=1.0
        )

        # c1 should have BM25 contribution only
        c1 = next(r for r in results if r.chunk.id == "c1")
        assert c1.fused_score == pytest.approx(1.0 / 61)

        # c2 has (1-1.0) * 1/61 = 0.0 score, so it's excluded (fused_score must be > 0)
        c2_results = [r for r in results if r.chunk.id == "c2"]
        assert len(c2_results) == 0

    def test_alpha_zero_uses_only_semantic(self):
        """alpha=0.0 means only semantic contributes to scores."""
        bm25_results = [make_scored_chunk("c1", 5.0, "bm25")]
        semantic_results = [make_scored_chunk("c2", 0.9, "semantic")]

        results = self.retriever.reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.0
        )

        # c1 should have 0 BM25 contribution (alpha=0)
        c1_results = [r for r in results if r.chunk.id == "c1"]
        c2 = next(r for r in results if r.chunk.id == "c2")

        # c2 should have semantic contribution only
        assert c2.fused_score == pytest.approx(1.0 / 61)
        # c1 has zero score from alpha=0 * BM25, so it may be excluded
        if c1_results:
            assert c1_results[0].fused_score == 0.0

    def test_document_in_both_lists_scores_higher(self):
        """A document appearing in both lists should score higher than one in only one."""
        bm25_results = [
            make_scored_chunk("c1", 5.0, "bm25"),
            make_scored_chunk("c2", 4.0, "bm25"),
        ]
        semantic_results = [
            make_scored_chunk("c1", 0.9, "semantic"),
        ]

        results = self.retriever.reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.5
        )

        c1 = next(r for r in results if r.chunk.id == "c1")
        c2 = next(r for r in results if r.chunk.id == "c2")

        # c1 in both lists should have higher score
        assert c1.fused_score > c2.fused_score

    def test_context_snippet_populated(self):
        """Each result should have context_snippet set to chunk content."""
        bm25_results = [make_scored_chunk("c1", 5.0, "bm25")]

        results = self.retriever.reciprocal_rank_fusion(bm25_results, [])
        assert results[0].context_snippet == results[0].chunk.content

    def test_custom_rrf_k_affects_scores(self):
        """Different k values should produce different scores."""
        retriever_k10 = HybridRetriever(self.bm25, self.semantic, rrf_k=10)
        retriever_k60 = HybridRetriever(self.bm25, self.semantic, rrf_k=60)

        bm25_results = [make_scored_chunk("c1", 5.0, "bm25")]

        results_k10 = retriever_k10.reciprocal_rank_fusion(bm25_results, [], alpha=0.5)
        results_k60 = retriever_k60.reciprocal_rank_fusion(bm25_results, [], alpha=0.5)

        # k=10: score = 0.5 * (1/11) ≈ 0.0455
        # k=60: score = 0.5 * (1/61) ≈ 0.0082
        assert results_k10[0].fused_score > results_k60[0].fused_score

    def test_all_scores_positive(self):
        """All fused scores should be positive (> 0.0)."""
        bm25_results = [
            make_scored_chunk(f"c{i}", float(10 - i), "bm25") for i in range(10)
        ]
        semantic_results = [
            make_scored_chunk(f"c{i}", 0.9 - i * 0.05, "semantic") for i in range(10)
        ]

        results = self.retriever.reciprocal_rank_fusion(bm25_results, semantic_results)
        for r in results:
            assert r.fused_score > 0.0


class TestHybridRetrieverRetrieve:
    """Tests for the retrieve() method with parallel execution."""

    @pytest.fixture
    def setup_retriever(self):
        """Create a retriever with mocked engines."""
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        retriever = HybridRetriever(bm25, semantic, rrf_k=60)
        return retriever, bm25, semantic

    @pytest.mark.asyncio
    async def test_basic_retrieve(self, setup_retriever):
        retriever, bm25, semantic = setup_retriever

        bm25.search.return_value = [
            make_scored_chunk("c1", 5.0, "bm25"),
            make_scored_chunk("c2", 3.0, "bm25"),
        ]
        semantic.search.return_value = [
            make_scored_chunk("c1", 0.9, "semantic"),
            make_scored_chunk("c3", 0.7, "semantic"),
        ]

        query = make_processed_query()
        results = await retriever.retrieve(query, top_k=10, alpha=0.5)

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self, setup_retriever):
        retriever, bm25, semantic = setup_retriever

        bm25.search.return_value = [
            make_scored_chunk(f"c{i}", float(15 - i), "bm25") for i in range(15)
        ]
        semantic.search.return_value = [
            make_scored_chunk(f"c{i}", 0.9 - i * 0.05, "semantic") for i in range(10)
        ]

        query = make_processed_query()
        results = await retriever.retrieve(query, top_k=5, alpha=0.5)

        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_retrieve_calls_with_3x_top_k(self, setup_retriever):
        """BM25 and semantic should be called with 3 * top_k candidates."""
        retriever, bm25, semantic = setup_retriever

        bm25.search.return_value = []
        semantic.search.return_value = []

        query = make_processed_query()
        await retriever.retrieve(query, top_k=10)

        bm25.search.assert_called_once()
        # Verify the top_k passed to BM25 is 3 * 10 = 30
        call_args = bm25.search.call_args
        assert call_args[1]["top_k"] == 30

        semantic.search.assert_called_once()
        call_args = semantic.search.call_args
        assert call_args[1]["top_k"] == 30

    @pytest.mark.asyncio
    async def test_retrieve_invalid_alpha_high(self, setup_retriever):
        retriever, bm25, semantic = setup_retriever
        bm25.search.return_value = []
        semantic.search.return_value = []
        query = make_processed_query()
        with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
            await retriever.retrieve(query, alpha=1.5)

    @pytest.mark.asyncio
    async def test_retrieve_invalid_alpha_low(self, setup_retriever):
        retriever, bm25, semantic = setup_retriever
        bm25.search.return_value = []
        semantic.search.return_value = []
        query = make_processed_query()
        with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
            await retriever.retrieve(query, alpha=-0.1)

    @pytest.mark.asyncio
    async def test_retrieve_invalid_top_k(self, setup_retriever):
        retriever, bm25, semantic = setup_retriever
        bm25.search.return_value = []
        semantic.search.return_value = []
        query = make_processed_query()
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            await retriever.retrieve(query, top_k=0)

    @pytest.mark.asyncio
    async def test_retrieve_sorted_descending(self, setup_retriever):
        retriever, bm25, semantic = setup_retriever

        bm25.search.return_value = [
            make_scored_chunk("c1", 5.0, "bm25"),
            make_scored_chunk("c2", 4.0, "bm25"),
            make_scored_chunk("c3", 3.0, "bm25"),
        ]
        semantic.search.return_value = [
            make_scored_chunk("c3", 0.9, "semantic"),
            make_scored_chunk("c2", 0.8, "semantic"),
            make_scored_chunk("c1", 0.7, "semantic"),
        ]

        query = make_processed_query()
        results = await retriever.retrieve(query, top_k=10, alpha=0.5)

        for i in range(len(results) - 1):
            assert results[i].fused_score >= results[i + 1].fused_score

    @pytest.mark.asyncio
    async def test_retrieve_uses_expanded_terms_for_bm25(self, setup_retriever):
        """BM25 search should use expanded_terms from the processed query."""
        retriever, bm25, semantic = setup_retriever

        bm25.search.return_value = []
        semantic.search.return_value = []

        custom_terms = ["get", "user", "name"]
        query = ProcessedQuery(
            original_question="how does get_user_name work?",
            intent=QueryIntent.CODE_LOOKUP,
            expanded_terms=custom_terms,
            query_embedding=[0.1] * 384,
            lexical_query="get user name",
            filters=QueryFilters(),
        )
        await retriever.retrieve(query)

        bm25.search.assert_called_once()
        call_args = bm25.search.call_args
        assert call_args[0][0] == custom_terms

    @pytest.mark.asyncio
    async def test_retrieve_uses_embedding_for_semantic(self, setup_retriever):
        """Semantic search should use query_embedding from the processed query."""
        retriever, bm25, semantic = setup_retriever

        bm25.search.return_value = []
        semantic.search.return_value = []

        embedding = [0.5] * 384
        query = make_processed_query(query_embedding=embedding)
        await retriever.retrieve(query)

        semantic.search.assert_called_once()
        call_args = semantic.search.call_args
        assert call_args[0][0] == embedding


class TestGracefulDegradation:
    """Tests for graceful degradation when one retriever fails."""

    @pytest.mark.asyncio
    async def test_bm25_failure_uses_semantic_only(self):
        """When BM25 fails, semantic results should still be returned."""
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        retriever = HybridRetriever(bm25, semantic, rrf_k=60)

        bm25.search.side_effect = RuntimeError("BM25 index corrupted")
        semantic.search.return_value = [
            make_scored_chunk("c1", 0.9, "semantic"),
            make_scored_chunk("c2", 0.8, "semantic"),
        ]

        query = make_processed_query()
        results = await retriever.retrieve(query, alpha=0.5)

        assert len(results) == 2
        # All results should have bm25_rank=None
        for r in results:
            assert r.bm25_rank is None
            assert r.semantic_rank is not None

    @pytest.mark.asyncio
    async def test_semantic_failure_uses_bm25_only(self):
        """When semantic fails, BM25 results should still be returned."""
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        retriever = HybridRetriever(bm25, semantic, rrf_k=60)

        bm25.search.return_value = [
            make_scored_chunk("c1", 5.0, "bm25"),
            make_scored_chunk("c2", 3.0, "bm25"),
        ]
        semantic.search.side_effect = ConnectionError("Vector DB unavailable")

        query = make_processed_query()
        results = await retriever.retrieve(query, alpha=0.5)

        assert len(results) == 2
        # All results should have semantic_rank=None
        for r in results:
            assert r.semantic_rank is None
            assert r.bm25_rank is not None

    @pytest.mark.asyncio
    async def test_both_fail_returns_empty(self):
        """When both retrievers fail, return empty results."""
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        retriever = HybridRetriever(bm25, semantic, rrf_k=60)

        bm25.search.side_effect = RuntimeError("BM25 error")
        semantic.search.side_effect = ConnectionError("Semantic error")

        query = make_processed_query()
        results = await retriever.retrieve(query, alpha=0.5)

        assert results == []

    @pytest.mark.asyncio
    async def test_empty_embedding_skips_semantic(self):
        """When query has no embedding, semantic search is skipped."""
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        retriever = HybridRetriever(bm25, semantic, rrf_k=60)

        bm25.search.return_value = [make_scored_chunk("c1", 5.0, "bm25")]
        semantic.search.return_value = []

        # Explicitly create query with empty embedding
        query = ProcessedQuery(
            original_question="how does get_user work?",
            intent=QueryIntent.CODE_LOOKUP,
            expanded_terms=["get", "user"],
            query_embedding=[],
            lexical_query="get user",
            filters=QueryFilters(),
        )
        results = await retriever.retrieve(query, alpha=0.5)

        assert len(results) == 1
        # Semantic search should not be called with empty embedding
        semantic.search.assert_not_called()


class TestRRFMonotonicity:
    """Tests verifying RRF monotonicity property.

    If document A ranks higher than document B in BOTH the BM25 and
    semantic result lists, then A's fused_score must be higher than B's.
    """

    def test_monotonicity_both_lists(self):
        """Doc ranked higher in both lists must have higher fused score."""
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        retriever = HybridRetriever(bm25, semantic, rrf_k=60)

        # c1 is rank 1 in both lists, c2 is rank 2 in both
        bm25_results = [
            make_scored_chunk("c1", 5.0, "bm25"),
            make_scored_chunk("c2", 3.0, "bm25"),
        ]
        semantic_results = [
            make_scored_chunk("c1", 0.9, "semantic"),
            make_scored_chunk("c2", 0.7, "semantic"),
        ]

        results = retriever.reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.5
        )

        c1 = next(r for r in results if r.chunk.id == "c1")
        c2 = next(r for r in results if r.chunk.id == "c2")

        assert c1.fused_score > c2.fused_score

    def test_monotonicity_various_alphas(self):
        """Monotonicity holds for different alpha values."""
        bm25 = MagicMock(spec=BM25Engine)
        semantic = MagicMock(spec=SemanticRetriever)
        retriever = HybridRetriever(bm25, semantic, rrf_k=60)

        bm25_results = [
            make_scored_chunk("c1", 5.0, "bm25"),
            make_scored_chunk("c2", 3.0, "bm25"),
            make_scored_chunk("c3", 2.0, "bm25"),
        ]
        semantic_results = [
            make_scored_chunk("c1", 0.9, "semantic"),
            make_scored_chunk("c2", 0.7, "semantic"),
            make_scored_chunk("c3", 0.5, "semantic"),
        ]

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            results = retriever.reciprocal_rank_fusion(
                bm25_results, semantic_results, alpha=alpha
            )
            if len(results) >= 2:
                # When alpha is 0 or 1, some docs may have 0 score and be excluded
                scored_results = [r for r in results if r.fused_score > 0]
                for i in range(len(scored_results) - 1):
                    assert scored_results[i].fused_score >= scored_results[i + 1].fused_score


class TestAlphaWeighting:
    """Tests for alpha weighting between BM25 and semantic results."""

    def setup_method(self):
        self.bm25 = MagicMock(spec=BM25Engine)
        self.semantic = MagicMock(spec=SemanticRetriever)
        self.retriever = HybridRetriever(self.bm25, self.semantic, rrf_k=60)

    def test_alpha_0_5_equal_weight(self):
        """alpha=0.5 gives equal weight to both sources."""
        bm25_results = [make_scored_chunk("c1", 5.0, "bm25")]
        semantic_results = [make_scored_chunk("c2", 0.9, "semantic")]

        results = self.retriever.reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.5
        )

        c1 = next(r for r in results if r.chunk.id == "c1")
        c2 = next(r for r in results if r.chunk.id == "c2")

        # Both at rank 1 in their respective lists with equal weight
        assert abs(c1.fused_score - c2.fused_score) < 1e-10

    def test_alpha_high_favors_bm25(self):
        """Higher alpha gives more weight to BM25 results."""
        # c1 only in BM25, c2 only in semantic
        bm25_results = [make_scored_chunk("c1", 5.0, "bm25")]
        semantic_results = [make_scored_chunk("c2", 0.9, "semantic")]

        results = self.retriever.reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.9
        )

        c1 = next(r for r in results if r.chunk.id == "c1")
        c2 = next(r for r in results if r.chunk.id == "c2")

        # BM25-only doc should score higher with high alpha
        assert c1.fused_score > c2.fused_score

    def test_alpha_low_favors_semantic(self):
        """Lower alpha gives more weight to semantic results."""
        bm25_results = [make_scored_chunk("c1", 5.0, "bm25")]
        semantic_results = [make_scored_chunk("c2", 0.9, "semantic")]

        results = self.retriever.reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.1
        )

        c1 = next(r for r in results if r.chunk.id == "c1")
        c2 = next(r for r in results if r.chunk.id == "c2")

        # Semantic-only doc should score higher with low alpha
        assert c2.fused_score > c1.fused_score
