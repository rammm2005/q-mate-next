"""Unit tests for retrieval evaluation metrics.

Tests cover:
- Precision@K at K = 1, 3, 5, 10
- Recall@K at K = 1, 3, 5, 10
- Mean Reciprocal Rank (MRR)
- Reciprocal Rank for single query
- Latency measurement
- Single query evaluation
- Multi-query evaluation (averaged metrics)
- Edge cases (empty inputs, no relevant docs, perfect retrieval)
"""

import time

import pytest

from app.services.evaluation import (
    EvaluationResult,
    K_VALUES,
    evaluate_retrieval,
    evaluate_single_query,
    mean_reciprocal_rank,
    measure_latency_ms,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPrecisionAtK:
    """Tests for Precision@K computation."""

    def test_perfect_precision_at_1(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2"}
        assert precision_at_k(retrieved, relevant, 1) == 1.0

    def test_zero_precision_at_1(self):
        retrieved = ["doc3", "doc1", "doc2"]
        relevant = {"doc1", "doc2"}
        assert precision_at_k(retrieved, relevant, 1) == 0.0

    def test_precision_at_3_partial(self):
        retrieved = ["doc1", "doc3", "doc2", "doc4", "doc5"]
        relevant = {"doc1", "doc2"}
        # top-3: doc1, doc3, doc2 → 2 relevant out of 3
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)

    def test_precision_at_5(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc5"}
        # top-5: 3 relevant out of 5
        assert precision_at_k(retrieved, relevant, 5) == pytest.approx(3 / 5)

    def test_precision_at_10_with_fewer_results(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2"}
        # Only 3 retrieved, so top-10 is same as all 3; 2 relevant / 3
        assert precision_at_k(retrieved, relevant, 10) == pytest.approx(2 / 3)

    def test_precision_at_10_all_relevant(self):
        retrieved = [f"doc{i}" for i in range(10)]
        relevant = set(retrieved)
        assert precision_at_k(retrieved, relevant, 10) == 1.0

    def test_precision_at_10_none_relevant(self):
        retrieved = [f"doc{i}" for i in range(10)]
        relevant = {"other1", "other2"}
        assert precision_at_k(retrieved, relevant, 10) == 0.0

    def test_empty_retrieved_list(self):
        assert precision_at_k([], {"doc1"}, 5) == 0.0

    def test_empty_relevant_set(self):
        retrieved = ["doc1", "doc2"]
        assert precision_at_k(retrieved, set(), 3) == 0.0

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["doc1"], {"doc1"}, 0) == 0.0

    def test_k_negative_returns_zero(self):
        assert precision_at_k(["doc1"], {"doc1"}, -1) == 0.0


class TestRecallAtK:
    """Tests for Recall@K computation."""

    def test_perfect_recall_at_1_single_relevant(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        assert recall_at_k(retrieved, relevant, 1) == 1.0

    def test_zero_recall_at_1(self):
        retrieved = ["doc3", "doc1", "doc2"]
        relevant = {"doc1", "doc2"}
        # top-1: doc3 → 0 out of 2 relevant
        assert recall_at_k(retrieved, relevant, 1) == 0.0

    def test_recall_at_3(self):
        retrieved = ["doc1", "doc3", "doc2", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc6"}
        # top-3: doc1, doc3, doc2 → 2 relevant out of 3 total relevant
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)

    def test_recall_at_5(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc5", "doc7"}
        # top-5: doc1, doc3, doc5 → 3 relevant out of 4 total relevant
        assert recall_at_k(retrieved, relevant, 5) == pytest.approx(3 / 4)

    def test_recall_at_10_all_relevant_found(self):
        retrieved = [f"doc{i}" for i in range(10)]
        relevant = {"doc0", "doc3", "doc7"}
        # All 3 relevant found in top-10
        assert recall_at_k(retrieved, relevant, 10) == 1.0

    def test_recall_at_10_partial(self):
        retrieved = [f"doc{i}" for i in range(10)]
        relevant = {"doc0", "doc3", "doc15", "doc20"}
        # 2 out of 4 relevant found in top-10
        assert recall_at_k(retrieved, relevant, 10) == pytest.approx(2 / 4)

    def test_empty_relevant_returns_zero(self):
        retrieved = ["doc1", "doc2"]
        assert recall_at_k(retrieved, set(), 5) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert recall_at_k([], {"doc1", "doc2"}, 5) == 0.0

    def test_k_zero_returns_zero(self):
        assert recall_at_k(["doc1"], {"doc1"}, 0) == 0.0

    def test_k_negative_returns_zero(self):
        assert recall_at_k(["doc1"], {"doc1"}, -1) == 0.0

    def test_recall_increases_with_k(self):
        retrieved = ["doc1", "doc4", "doc2", "doc5", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        r1 = recall_at_k(retrieved, relevant, 1)
        r3 = recall_at_k(retrieved, relevant, 3)
        r5 = recall_at_k(retrieved, relevant, 5)
        assert r1 <= r3 <= r5


class TestReciprocalRank:
    """Tests for single-query Reciprocal Rank."""

    def test_first_result_relevant(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        assert reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_result_relevant(self):
        retrieved = ["doc2", "doc1", "doc3"]
        relevant = {"doc1"}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 2)

    def test_third_result_relevant(self):
        retrieved = ["doc3", "doc2", "doc1"]
        relevant = {"doc1"}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_found(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4"}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank([], {"doc1"}) == 0.0

    def test_empty_relevant(self):
        retrieved = ["doc1", "doc2"]
        assert reciprocal_rank(retrieved, set()) == 0.0

    def test_multiple_relevant_uses_first(self):
        """RR only considers the first relevant result."""
        retrieved = ["doc3", "doc1", "doc2"]
        relevant = {"doc1", "doc2"}
        # doc1 is at rank 2, doc2 is at rank 3; first relevant is rank 2
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 2)


class TestMeanReciprocalRank:
    """Tests for Mean Reciprocal Rank across multiple queries."""

    def test_single_query_perfect(self):
        queries = [(["doc1", "doc2"], {"doc1"})]
        assert mean_reciprocal_rank(queries) == 1.0

    def test_single_query_second_rank(self):
        queries = [(["doc2", "doc1"], {"doc1"})]
        assert mean_reciprocal_rank(queries) == pytest.approx(0.5)

    def test_multiple_queries(self):
        queries = [
            (["doc1", "doc2", "doc3"], {"doc1"}),  # RR = 1.0
            (["doc3", "doc1", "doc2"], {"doc1"}),  # RR = 0.5
            (["doc2", "doc3", "doc1"], {"doc1"}),  # RR = 1/3
        ]
        expected = (1.0 + 0.5 + 1 / 3) / 3
        assert mean_reciprocal_rank(queries) == pytest.approx(expected)

    def test_no_queries(self):
        assert mean_reciprocal_rank([]) == 0.0

    def test_all_queries_no_relevant(self):
        queries = [
            (["doc1", "doc2"], {"doc5"}),
            (["doc3", "doc4"], {"doc6"}),
        ]
        assert mean_reciprocal_rank(queries) == 0.0

    def test_mixed_found_and_not_found(self):
        queries = [
            (["doc1", "doc2"], {"doc1"}),  # RR = 1.0
            (["doc3", "doc4"], {"doc5"}),  # RR = 0.0
        ]
        expected = (1.0 + 0.0) / 2
        assert mean_reciprocal_rank(queries) == pytest.approx(expected)


class TestMeasureLatency:
    """Tests for latency measurement utility."""

    def test_positive_latency(self):
        start = time.perf_counter()
        # Small computation
        _ = sum(range(1000))
        end = time.perf_counter()
        latency = measure_latency_ms(start, end)
        assert latency >= 0.0

    def test_zero_duration(self):
        t = 1000.0
        assert measure_latency_ms(t, t) == 0.0

    def test_known_duration(self):
        # 0.5 seconds = 500 ms
        start = 0.0
        end = 0.5
        assert measure_latency_ms(start, end) == pytest.approx(500.0)

    def test_small_duration(self):
        # 1 ms = 0.001 seconds
        start = 0.0
        end = 0.001
        assert measure_latency_ms(start, end) == pytest.approx(1.0)


class TestEvaluateSingleQuery:
    """Tests for single-query evaluation wrapper."""

    def test_perfect_retrieval(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        result = evaluate_single_query(retrieved, relevant, latency_ms=5.0)

        assert result.precision_at_k[1] == 1.0
        assert result.precision_at_k[3] == 1.0
        assert result.recall_at_k[1] == pytest.approx(1 / 3)
        assert result.recall_at_k[3] == 1.0
        assert result.mrr == 1.0
        assert result.latency_ms == 5.0

    def test_no_relevant_retrieval(self):
        retrieved = ["doc4", "doc5", "doc6"]
        relevant = {"doc1", "doc2"}
        result = evaluate_single_query(retrieved, relevant, latency_ms=10.0)

        assert result.precision_at_k[1] == 0.0
        assert result.precision_at_k[3] == 0.0
        assert result.recall_at_k[1] == 0.0
        assert result.recall_at_k[3] == 0.0
        assert result.mrr == 0.0
        assert result.latency_ms == 10.0

    def test_all_k_values_present(self):
        retrieved = [f"doc{i}" for i in range(15)]
        relevant = {"doc0", "doc5", "doc9"}
        result = evaluate_single_query(retrieved, relevant)

        for k in K_VALUES:
            assert k in result.precision_at_k
            assert k in result.recall_at_k

    def test_default_latency_zero(self):
        result = evaluate_single_query(["doc1"], {"doc1"})
        assert result.latency_ms == 0.0


class TestEvaluateRetrieval:
    """Tests for multi-query evaluation."""

    def test_empty_queries(self):
        result = evaluate_retrieval([])
        assert result.precision_at_k == {}
        assert result.recall_at_k == {}
        assert result.mrr == 0.0
        assert result.latency_ms == 0.0

    def test_single_query(self):
        queries = [(["doc1", "doc2", "doc3"], {"doc1", "doc2"})]
        result = evaluate_retrieval(queries)

        # P@1 = 1.0 (doc1 is relevant)
        assert result.precision_at_k[1] == 1.0
        # P@3 = 2/3
        assert result.precision_at_k[3] == pytest.approx(2 / 3)
        # R@1 = 1/2
        assert result.recall_at_k[1] == pytest.approx(1 / 2)
        # R@3 = 2/2 = 1.0
        assert result.recall_at_k[3] == 1.0
        # MRR = 1.0 (first result relevant)
        assert result.mrr == 1.0

    def test_multiple_queries_averaged(self):
        queries = [
            (["doc1", "doc2", "doc3"], {"doc1"}),  # P@1=1.0, RR=1.0
            (["doc2", "doc1", "doc3"], {"doc1"}),  # P@1=0.0, RR=0.5
        ]
        result = evaluate_retrieval(queries)

        # Averaged P@1 = (1.0 + 0.0) / 2 = 0.5
        assert result.precision_at_k[1] == pytest.approx(0.5)
        # MRR = (1.0 + 0.5) / 2 = 0.75
        assert result.mrr == pytest.approx(0.75)

    def test_latencies_averaged(self):
        queries = [(["doc1"], {"doc1"}), (["doc2"], {"doc2"})]
        latencies = [10.0, 20.0]
        result = evaluate_retrieval(queries, latencies_ms=latencies)
        assert result.latency_ms == pytest.approx(15.0)

    def test_no_latencies_provided(self):
        queries = [(["doc1"], {"doc1"})]
        result = evaluate_retrieval(queries)
        assert result.latency_ms == 0.0

    def test_result_values_bounded(self):
        """All metric values should be between 0.0 and 1.0."""
        queries = [
            (["doc1", "doc3", "doc5", "doc7", "doc9"], {"doc1", "doc2", "doc3"}),
            (["doc2", "doc4", "doc6", "doc8", "doc10"], {"doc1", "doc5"}),
            ([f"doc{i}" for i in range(12)], {"doc3", "doc7", "doc11"}),
        ]
        result = evaluate_retrieval(queries)

        for k in K_VALUES:
            assert 0.0 <= result.precision_at_k[k] <= 1.0
            assert 0.0 <= result.recall_at_k[k] <= 1.0
        assert 0.0 <= result.mrr <= 1.0


class TestEvaluationResult:
    """Tests for the EvaluationResult dataclass."""

    def test_default_values(self):
        result = EvaluationResult()
        assert result.precision_at_k == {}
        assert result.recall_at_k == {}
        assert result.mrr == 0.0
        assert result.latency_ms == 0.0

    def test_custom_values(self):
        result = EvaluationResult(
            precision_at_k={1: 0.85, 3: 0.72, 5: 0.68, 10: 0.55},
            recall_at_k={1: 0.12, 3: 0.35, 5: 0.52, 10: 0.78},
            mrr=0.91,
            latency_ms=42.5,
        )
        assert result.precision_at_k[1] == 0.85
        assert result.recall_at_k[10] == 0.78
        assert result.mrr == 0.91
        assert result.latency_ms == 42.5
