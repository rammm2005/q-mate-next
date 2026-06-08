"""BM25 Engine with inverted index for lexical code search.

This module implements the BM25 (Best Matching 25) ranking algorithm with an
inverted index for efficient lexical search over code chunks. It supports
configurable k1 and b parameters, incremental index updates, and uses
code-aware tokenization for indexing.

BM25 formula:
    score(Q, D) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

where:
    IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
    f(qi, D) = frequency of term qi in document D
    |D| = document length in tokens
    avgdl = average document length across corpus
    N = total number of documents
    n(qi) = number of documents containing qi
"""

import math
from collections import defaultdict

from app.models.chunk import CodeChunk
from app.models.retrieval import ScoredChunk
from app.services.tokenizer import code_aware_tokenize


class BM25Engine:
    """BM25 search engine with inverted index for code chunks.

    Provides exact lexical matching against indexed code and documentation
    tokens with configurable BM25 parameters.

    Attributes:
        k1: Term frequency saturation parameter (default 1.5, must be > 0).
        b: Document length normalization parameter (default 0.75, range [0, 1]).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize the BM25 engine with scoring parameters.

        Args:
            k1: Term frequency saturation parameter. Higher values increase
                the impact of term frequency. Must be > 0.
            b: Document length normalization parameter. 0 means no length
                normalization, 1 means full normalization. Must be in [0, 1].
        """
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if b < 0.0 or b > 1.0:
            raise ValueError("b must be between 0.0 and 1.0")

        self.k1 = k1
        self.b = b

        # Inverted index: term -> set of chunk_ids containing the term
        self._inverted_index: dict[str, set[str]] = defaultdict(set)

        # Document store: chunk_id -> CodeChunk
        self._chunks: dict[str, CodeChunk] = {}

        # Document token lists: chunk_id -> list of tokens
        self._doc_tokens: dict[str, list[str]] = {}

        # Document term frequencies: chunk_id -> {term: frequency}
        self._doc_term_freqs: dict[str, dict[str, int]] = {}

        # Document lengths (in tokens): chunk_id -> length
        self._doc_lengths: dict[str, int] = {}

        # Corpus statistics
        self._total_docs: int = 0
        self._total_doc_length: int = 0
        self._avg_doc_length: float = 0.0

    @property
    def total_documents(self) -> int:
        """Return the total number of indexed documents."""
        return self._total_docs

    @property
    def avg_doc_length(self) -> float:
        """Return the average document length in the corpus."""
        return self._avg_doc_length

    def build_index(self, chunks: list[CodeChunk]) -> None:
        """Build the inverted index from scratch using the provided chunks.

        This replaces any existing index data.

        Args:
            chunks: List of CodeChunk objects to index.
        """
        # Clear existing index
        self._inverted_index = defaultdict(set)
        self._chunks = {}
        self._doc_tokens = {}
        self._doc_term_freqs = {}
        self._doc_lengths = {}
        self._total_docs = 0
        self._total_doc_length = 0
        self._avg_doc_length = 0.0

        # Add all chunks
        self._index_chunks(chunks)

    def add_to_index(self, chunks: list[CodeChunk]) -> None:
        """Add chunks to the existing index without full rebuild.

        Incrementally updates the inverted index, document frequencies,
        and corpus statistics.

        Args:
            chunks: List of CodeChunk objects to add to the index.
        """
        self._index_chunks(chunks)

    def search(self, query_tokens: list[str], top_k: int = 20) -> list[ScoredChunk]:
        """Search the index for chunks matching the query tokens.

        Computes BM25 scores for all documents containing at least one query
        token, then returns the top-k results sorted by score descending.

        Args:
            query_tokens: List of query tokens to search for.
            top_k: Maximum number of results to return (default 20).

        Returns:
            List of ScoredChunk objects sorted by BM25 score in descending order.
            Returns empty list if query_tokens is empty or index is empty.
        """
        if not query_tokens:
            return []

        if self._total_docs == 0:
            return []

        # Compute BM25 scores for candidate documents
        scores: dict[str, float] = {}

        for token in query_tokens:
            if token not in self._inverted_index:
                continue

            # Get document frequency for IDF calculation
            df = len(self._inverted_index[token])

            # IDF: log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
            idf = math.log(
                (self._total_docs - df + 0.5) / (df + 0.5) + 1.0
            )

            # Score each document containing this token
            for chunk_id in self._inverted_index[token]:
                tf = self._doc_term_freqs[chunk_id].get(token, 0)
                doc_len = self._doc_lengths[chunk_id]

                # BM25 TF normalization
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_length)
                )

                contribution = idf * tf_norm

                # Only add non-negative contributions
                if contribution > 0:
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution

        # Filter to only non-negative scores and sort
        results: list[tuple[str, float]] = [
            (chunk_id, score) for chunk_id, score in scores.items() if score > 0.0
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results as ScoredChunk objects
        scored_chunks: list[ScoredChunk] = []
        for chunk_id, score in results[:top_k]:
            scored_chunks.append(
                ScoredChunk(
                    chunk=self._chunks[chunk_id],
                    score=score,
                    source="bm25",
                )
            )

        return scored_chunks

    def _index_chunks(self, chunks: list[CodeChunk]) -> None:
        """Internal method to index a list of chunks.

        Updates the inverted index, term frequencies, document lengths,
        and corpus statistics.

        Args:
            chunks: List of CodeChunk objects to index.
        """
        for chunk in chunks:
            chunk_id = chunk.id

            # Skip if already indexed (avoid duplicates)
            if chunk_id in self._chunks:
                continue

            # Tokenize the chunk content using code-aware tokenizer
            tokens = code_aware_tokenize(chunk.content, chunk.language)

            # Store chunk and tokens
            self._chunks[chunk_id] = chunk
            self._doc_tokens[chunk_id] = tokens

            # Compute term frequencies
            term_freqs: dict[str, int] = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1
            self._doc_term_freqs[chunk_id] = term_freqs

            # Store document length
            doc_len = len(tokens)
            self._doc_lengths[chunk_id] = doc_len

            # Update inverted index
            for token in term_freqs:
                self._inverted_index[token].add(chunk_id)

            # Update corpus statistics
            self._total_docs += 1
            self._total_doc_length += doc_len

        # Recompute average document length
        if self._total_docs > 0:
            self._avg_doc_length = self._total_doc_length / self._total_docs
        else:
            self._avg_doc_length = 0.0
