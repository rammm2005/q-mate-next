"""IndoBERT retriever using sentence-transformers for local/in-memory semantic search.

Generates embeddings using an Indonesian semantic model ('firqaaa/indo-sentence-bert-base')
and computes cosine similarity scores in memory.
"""

import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from app.models.chunk import CodeChunk
from app.models.retrieval import ScoredChunk

logger = logging.getLogger(__name__)

INDOBERT_MODEL_NAME = "firqaaa/indo-sentence-bert-base"
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"

class IndoBERTRetriever:
    """Performs semantic vector search using sentence-transformers in memory."""

    def __init__(self, model_name: str = INDOBERT_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}. Falling back to {FALLBACK_MODEL_NAME}")
            try:
                self.model = SentenceTransformer(FALLBACK_MODEL_NAME)
                self.model_name = FALLBACK_MODEL_NAME
            except Exception as ex:
                logger.error(f"Failed to load fallback model: {ex}")
                raise ex

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        return self.model.encode(query, convert_to_numpy=True)

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        """Embed a list of text chunks."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        if not texts:
            return np.empty((0, 0))
        return self.model.encode(texts, convert_to_numpy=True)

    def search(
        self,
        query: str,
        chunks: list[CodeChunk],
        embeddings: np.ndarray,
        top_k: int = 20
    ) -> list[ScoredChunk]:
        """Search the in-memory chunks using cosine similarity."""
        if not chunks or embeddings.size == 0:
            return []

        # Generate query embedding
        query_vector = self.embed_query(query)

        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        query_vector = query_vector / query_norm

        # Normalize chunk embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        norm_embeddings = embeddings / norms

        # Cosine similarity (dot product of normalized vectors)
        similarities = np.dot(norm_embeddings, query_vector)

        # Build ScoredChunk list
        scored_chunks = []
        for idx, similarity in enumerate(similarities):
            # cosine similarity is in [-1, 1], filter out <= 0.0 values
            score = float(similarity)
            if score <= 0.0:
                continue
            
            scored_chunks.append(
                ScoredChunk(
                    chunk=chunks[idx],
                    score=score,
                    source="semantic"
                )
            )

        # Sort by similarity descending
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        return scored_chunks[:top_k]
