"""Semantic retriever using sentence-transformers and pgvector.

This module provides vector-based semantic search over code chunks stored
in Supabase with pgvector. It uses the all-MiniLM-L6-v2 sentence transformer
model to generate 384-dimensional embeddings for queries and code chunks,
then performs cosine similarity search via the Supabase `match_chunks` RPC function.
"""

from sentence_transformers import SentenceTransformer
from supabase import Client as SupabaseClient

from app.models.chunk import CodeChunk
from app.models.retrieval import ScoredChunk


# Model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
MAX_BATCH_SIZE = 256


class EmbeddingError(Exception):
    """Raised when the embedding model fails to generate a vector."""

    def __init__(self, message: str, query: str | None = None):
        self.query = query
        super().__init__(message)


class SemanticRetriever:
    """Performs semantic vector search using sentence-transformers and pgvector.

    This class handles:
    - Embedding generation for queries and code chunks
    - Vector similarity search via Supabase pgvector (match_chunks function)
    - Batch embedding for ingestion pipeline

    Attributes:
        model: The sentence-transformers model instance.
        supabase: The Supabase client for database interaction.
    """

    def __init__(
        self,
        supabase_client: SupabaseClient,
        model: SentenceTransformer | None = None,
    ):
        """Initialize the SemanticRetriever.

        Args:
            supabase_client: Supabase client for pgvector search.
            model: Optional pre-loaded SentenceTransformer model.
                   If None, loads the default all-MiniLM-L6-v2 model.
        """
        self.supabase = supabase_client
        if model is not None:
            self.model = model
        else:
            try:
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}"
                )

    def embed_query(self, query: str) -> list[float]:
        """Generate a 384-dimensional embedding vector for a query string.

        Args:
            query: The query text to embed.

        Returns:
            A list of 384 floats representing the embedding vector.

        Raises:
            EmbeddingError: If the embedding model fails to generate a vector.
        """
        try:
            embedding = self.model.encode(query, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embedding for query: {e}",
                query=query,
            )

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """Batch embed code chunks for the ingestion pipeline.

        Processes chunks in batches of up to 256 to manage memory usage.

        Args:
            chunks: List of CodeChunk objects to embed.

        Returns:
            A list of embedding vectors (each a list of 384 floats),
            in the same order as the input chunks.

        Raises:
            EmbeddingError: If the embedding model fails during batch processing.
        """
        if not chunks:
            return []

        all_embeddings: list[list[float]] = []

        # Process in batches of MAX_BATCH_SIZE
        for i in range(0, len(chunks), MAX_BATCH_SIZE):
            batch = chunks[i : i + MAX_BATCH_SIZE]
            texts = [chunk.content for chunk in batch]

            try:
                batch_embeddings = self.model.encode(texts, convert_to_numpy=True)
                all_embeddings.extend(batch_embeddings.tolist())
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to generate embeddings for chunk batch "
                    f"(batch starting at index {i}): {e}"
                )

        return all_embeddings

    def search(
        self, query_embedding: list[float], top_k: int = 20
    ) -> list[ScoredChunk]:
        """Search for similar chunks using pgvector cosine similarity.

        Calls the Supabase `match_chunks` RPC function to perform approximate
        nearest neighbor search and returns results with cosine similarity >= 0.0.

        Args:
            query_embedding: The 384-dimensional query embedding vector.
            top_k: Maximum number of results to return (default 20).

        Returns:
            A list of ScoredChunk objects sorted by cosine similarity
            in descending order, excluding results with similarity < 0.0.
        """
        # Call the match_chunks RPC function in Supabase
        response = self.supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
                "similarity_threshold": 0.0,
            },
        ).execute()

        results: list[ScoredChunk] = []

        if not response.data:
            return results

        for row in response.data:
            # Only include results with cosine similarity >= 0.0
            similarity = row.get("similarity", 0.0)
            if similarity < 0.0:
                continue

            chunk = CodeChunk(
                id=str(row["id"]),
                repo_id=str(row["repo_id"]),
                file_path=row["file_path"],
                language=row["language"],
                chunk_type=row["chunk_type"],
                content=row["content"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                metadata={
                    "function_name": row.get("function_name"),
                    "class_name": row.get("class_name"),
                },
            )

            scored = ScoredChunk(
                chunk=chunk,
                score=similarity,
                source="semantic",
            )
            results.append(scored)

        return results
