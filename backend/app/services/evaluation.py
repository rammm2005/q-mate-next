"""Retrieval evaluation metrics for CodeQ-Mate.

Implements standard IR evaluation metrics based on Arwan et al. (SIET 2023):
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Retrieval latency measurement

Requirements: 14.1, 14.2, 14.3, 14.5
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from app.models.retrieval import ScoredChunk


K_VALUES = [1, 3, 5, 10]


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics from a single evaluation run."""

    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    latency_ms: float = 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@K.

    Precision@K = |relevant ∩ retrieved[:k]| / k

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Precision value between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@K.

    Recall@K = |relevant ∩ retrieved[:k]| / |relevant|

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Recall value between 0.0 and 1.0.
    """
    if not relevant_ids:
        return 0.0
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Reciprocal Rank for a single query.

    RR = 1 / rank of the first relevant document (0.0 if none found).

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: Set of ground-truth relevant document IDs.

    Returns:
        Reciprocal rank value between 0.0 and 1.0.
    """
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(queries_results: list[tuple[list[str], set[str]]]) -> float:
    """Compute Mean Reciprocal Rank (MRR) across multiple queries.

    MRR = (1/|Q|) * Σ (1 / rank_i) for i in queries

    Args:
        queries_results: List of (retrieved_ids, relevant_ids) tuples,
            one per evaluation query.

    Returns:
        MRR value between 0.0 and 1.0.
    """
    if not queries_results:
        return 0.0
    total_rr = 0.0
    for retrieved_ids, relevant_ids in queries_results:
        total_rr += reciprocal_rank(retrieved_ids, relevant_ids)
    return total_rr / len(queries_results)


def measure_latency_ms(start_time: float, end_time: float) -> float:
    """Compute elapsed time in milliseconds.

    Args:
        start_time: Start timestamp from time.perf_counter().
        end_time: End timestamp from time.perf_counter().

    Returns:
        Elapsed time in milliseconds with at least 2 decimal places precision.
    """
    return (end_time - start_time) * 1000.0


def evaluate_single_query(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    latency_ms: float = 0.0,
) -> EvaluationResult:
    """Evaluate retrieval quality for a single query.

    Computes Precision@K, Recall@K for K in [1, 3, 5, 10] and the
    reciprocal rank.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: Set of ground-truth relevant document IDs.
        latency_ms: Measured retrieval latency in milliseconds.

    Returns:
        EvaluationResult with all computed metrics.
    """
    p_at_k = {k: precision_at_k(retrieved_ids, relevant_ids, k) for k in K_VALUES}
    r_at_k = {k: recall_at_k(retrieved_ids, relevant_ids, k) for k in K_VALUES}
    rr = reciprocal_rank(retrieved_ids, relevant_ids)

    return EvaluationResult(
        precision_at_k=p_at_k,
        recall_at_k=r_at_k,
        mrr=rr,
        latency_ms=latency_ms,
    )


def evaluate_retrieval(
    queries_results: list[tuple[list[str], set[str]]],
    latencies_ms: list[float] | None = None,
) -> EvaluationResult:
    """Evaluate retrieval quality across multiple queries.

    Computes averaged Precision@K, Recall@K, MRR, and average latency
    across all evaluation queries.

    Args:
        queries_results: List of (retrieved_ids, relevant_ids) tuples.
        latencies_ms: Optional list of per-query latencies in milliseconds.

    Returns:
        EvaluationResult with averaged metrics across all queries.
    """
    if not queries_results:
        return EvaluationResult()

    n = len(queries_results)

    # Compute averaged Precision@K
    avg_precision = {}
    for k in K_VALUES:
        total = sum(
            precision_at_k(retrieved, relevant, k)
            for retrieved, relevant in queries_results
        )
        avg_precision[k] = total / n

    # Compute averaged Recall@K
    avg_recall = {}
    for k in K_VALUES:
        total = sum(
            recall_at_k(retrieved, relevant, k)
            for retrieved, relevant in queries_results
        )
        avg_recall[k] = total / n

    # Compute MRR
    mrr = mean_reciprocal_rank(queries_results)

    # Compute average latency
    avg_latency = 0.0
    if latencies_ms:
        avg_latency = sum(latencies_ms) / len(latencies_ms)

    return EvaluationResult(
        precision_at_k=avg_precision,
        recall_at_k=avg_recall,
        mrr=mrr,
        latency_ms=avg_latency,
    )
