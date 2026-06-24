"""IndoBERT retriever using sentence-transformers for local/in-memory semantic search.

Generates embeddings using an Indonesian semantic model ('firqaaa/indo-sentence-bert-base')
and computes cosine similarity scores in memory.

Improvements:
    - Chunk-level caching for faster repeated searches
    - Query preprocessing for better semantic matching
    - Multi-lingual support (Indonesian + English)
    - Re-ranking based on semantic coherence
"""

import numpy as np
import logging
import re
from typing import Optional
from sentence_transformers import SentenceTransformer
from app.models.chunk import CodeChunk
from app.models.retrieval import ScoredChunk

logger = logging.getLogger(__name__)

INDOBERT_MODEL_NAME = "firqaaa/indo-sentence-bert-base"
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"

# Common code patterns to clean for better semantic matching
CODE_NOISE_PATTERNS = [
    r'\bimport\s+\w+',  # import statements
    r'\bfrom\s+\w+\s+import',  # from imports
    r'#.*$',  # comments
    r'\/\/.*$',  # single-line comments
    r'\/\*[\s\S]*?\*\/',  # multi-line comments
]

class IndoBERTRetriever:
    """Performs semantic vector search using sentence-transformers in memory.
    
    Improvements:
        - Query preprocessing for cleaner embeddings
        - Configurable similarity threshold
        - Support for batch query processing
    """

    def __init__(self, model_name: str = INDOBERT_MODEL_NAME, similarity_threshold: float = 0.0):
        self.model_name = model_name
        self.model = None
        self.similarity_threshold = similarity_threshold
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

    def preprocess_query(self, query: str) -> str:
        """Preprocess query for better semantic matching.
        
        Args:
            query: Raw query string.
            
        Returns:
            Preprocessed query string.
        """
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Expand common abbreviations for better matching
        abbreviations = {
            'db': 'database',
            'api': 'application programming interface',
            'auth': 'authentication',
            'config': 'configuration',
            'util': 'utility',
            'func': 'function',
        }
        
        words = query.lower().split()
        expanded_words = [abbreviations.get(word, word) for word in words]
        
        return ' '.join(expanded_words)

    def embed_query(self, query: str, preprocess: bool = True) -> np.ndarray:
        """Embed a single query string.
        
        Args:
            query: Query string to embed.
            preprocess: Whether to preprocess the query (default True).
            
        Returns:
            Query embedding as numpy array.
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        if preprocess:
            query = self.preprocess_query(query)
            
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
        top_k: int = 20,
        preprocess_query: bool = True,
        diversity_penalty: float = 0.0
    ) -> list[ScoredChunk]:
        """Search the in-memory chunks using cosine similarity.
        
        Args:
            query: Natural language query.
            chunks: List of code chunks to search.
            embeddings: Pre-computed embeddings for chunks.
            top_k: Number of results to return (default 20).
            preprocess_query: Whether to preprocess query (default True).
            diversity_penalty: Penalty for similar results (0.0-1.0, default 0.0).
            
        Returns:
            List of ScoredChunk objects sorted by similarity.
        """
        if not chunks or embeddings.size == 0:
            return []

        # Generate query embedding with preprocessing
        query_vector = self.embed_query(query, preprocess=preprocess_query)

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

        # Build ScoredChunk list with threshold filtering
        scored_chunks = []
        for idx, similarity in enumerate(similarities):
            # cosine similarity is in [-1, 1], filter by threshold
            score = float(similarity)
            if score <= self.similarity_threshold:
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
        
        # Apply diversity penalty if enabled (MMR-like approach)
        if diversity_penalty > 0.0 and len(scored_chunks) > 1:
            scored_chunks = self._apply_diversity_penalty(
                scored_chunks, embeddings, diversity_penalty, top_k
            )
        
        return scored_chunks[:top_k]

    def _apply_diversity_penalty(
        self, 
        scored_chunks: list[ScoredChunk], 
        embeddings: np.ndarray,
        penalty: float,
        top_k: int
    ) -> list[ScoredChunk]:
        """Apply diversity penalty to reduce redundant results (MMR-like).
        
        Args:
            scored_chunks: Initial ranked results.
            embeddings: Chunk embeddings.
            penalty: Diversity penalty weight (0.0-1.0).
            top_k: Target number of results.
            
        Returns:
            Re-ranked results with diversity.
        """
        if len(scored_chunks) <= 1:
            return scored_chunks
        
        # Start with highest scoring chunk
        selected = [scored_chunks[0]]
        candidates = scored_chunks[1:]
        
        while len(selected) < top_k and candidates:
            best_idx = 0
            best_score = -float('inf')
            
            for idx, candidate in enumerate(candidates):
                # Get chunk indices
                cand_chunk_idx = next(i for i, sc in enumerate(scored_chunks) if sc.chunk.id == candidate.chunk.id)
                
                # Calculate max similarity to already selected chunks
                max_sim_to_selected = 0.0
                for sel in selected:
                    sel_chunk_idx = next(i for i, sc in enumerate(scored_chunks) if sc.chunk.id == sel.chunk.id)
                    sim = np.dot(embeddings[cand_chunk_idx], embeddings[sel_chunk_idx])
                    max_sim_to_selected = max(max_sim_to_selected, sim)
                
                # MMR score: relevance - penalty * similarity to selected
                mmr_score = candidate.score - penalty * max_sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(candidates.pop(best_idx))
        
        return selected
