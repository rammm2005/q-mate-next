"""Unit tests for the embedding generation and indexing pipeline (ingest_repository).

Tests the full ingestion pipeline: walk → chunk → embed → store → index,
with mocked dependencies for chunker, semantic_retriever, bm25_engine, and supabase.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.models.chunk import ChunkMetadata, ChunkType, CodeChunk
from app.services.ingestion import (
    IngestionConfig,
    IngestionPipeline,
    IngestionStats,
    ParsedFile,
)


# --- Fixtures ---


def _make_chunk(
    file_path: str = "src/main.py",
    content: str = "def hello(): pass",
    chunk_id: str | None = None,
    repo_id: str = "test-repo",
    language: str = "python",
    chunk_type: ChunkType = ChunkType.FUNCTION,
    start_line: int = 1,
    end_line: int = 1,
) -> CodeChunk:
    """Helper to create a CodeChunk for testing."""
    return CodeChunk(
        id=chunk_id or str(uuid.uuid4()),
        repo_id=repo_id,
        file_path=file_path,
        language=language,
        chunk_type=chunk_type,
        content=content,
        start_line=start_line,
        end_line=end_line,
        metadata=ChunkMetadata(function_name="hello"),
    )


@pytest.fixture
def mock_chunker():
    """Create a mock CodeChunker."""
    chunker = MagicMock()
    # Default: returns one chunk per file
    chunker.chunk_file.return_value = [
        _make_chunk(content="def hello(): pass", chunk_id="chunk-1")
    ]
    return chunker


@pytest.fixture
def mock_semantic_retriever():
    """Create a mock SemanticRetriever."""
    retriever = MagicMock()
    # Default: returns one embedding per chunk
    retriever.embed_chunks.return_value = [[0.1] * 384]
    return retriever


@pytest.fixture
def mock_bm25_engine():
    """Create a mock BM25Engine."""
    engine = MagicMock()
    engine.build_index.return_value = None
    return engine


@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client with chained method calls."""
    client = MagicMock()

    # Setup chained calls for delete: table().delete().eq().execute()
    delete_chain = MagicMock()
    delete_chain.eq.return_value = delete_chain
    delete_chain.execute.return_value = MagicMock(data=[])

    # Setup chained calls for insert: table().insert().execute()
    insert_chain = MagicMock()
    insert_chain.execute.return_value = MagicMock(data=[])

    # table() returns an object with both .delete() and .insert()
    table_mock = MagicMock()
    table_mock.delete.return_value = delete_chain
    table_mock.insert.return_value = insert_chain

    client.table.return_value = table_mock
    return client


@pytest.fixture
def pipeline(mock_chunker, mock_semantic_retriever, mock_bm25_engine, mock_supabase):
    """Create an IngestionPipeline with all mocked dependencies."""
    return IngestionPipeline(
        chunker=mock_chunker,
        semantic_retriever=mock_semantic_retriever,
        bm25_engine=mock_bm25_engine,
        supabase_client=mock_supabase,
        repo_id="test-repo",
    )


@pytest.fixture
def temp_repo(tmp_path):
    """Create a minimal temporary repository for ingestion tests."""
    py_file = tmp_path / "src" / "main.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("def hello():\n    return 'world'\n")

    ts_file = tmp_path / "src" / "app.ts"
    ts_file.write_text("export function greet(): string { return 'hi'; }\n")

    return tmp_path


# --- Tests ---


class TestIngestRepositoryBasic:
    """Tests for the basic ingest_repository flow."""

    @pytest.mark.asyncio
    async def test_returns_ingestion_stats(self, pipeline, temp_repo):
        """ingest_repository returns IngestionStats with correct counts."""
        stats = await pipeline.ingest_repository(str(temp_repo))

        assert isinstance(stats, IngestionStats)
        assert stats.total_files > 0
        assert stats.total_chunks > 0
        assert stats.total_embeddings > 0
        assert stats.repo_id == "test-repo"

    @pytest.mark.asyncio
    async def test_walks_repository(self, pipeline, temp_repo):
        """ingest_repository walks the directory tree and finds files."""
        stats = await pipeline.ingest_repository(str(temp_repo))

        # Should find at least the 2 source files
        assert stats.total_files >= 2

    @pytest.mark.asyncio
    async def test_chunks_each_file(
        self, pipeline, temp_repo, mock_chunker
    ):
        """ingest_repository calls chunker for each parsed file."""
        await pipeline.ingest_repository(str(temp_repo))

        # chunker.chunk_file should be called for each file found
        assert mock_chunker.chunk_file.call_count >= 2

    @pytest.mark.asyncio
    async def test_generates_embeddings(
        self, pipeline, temp_repo, mock_chunker, mock_semantic_retriever
    ):
        """ingest_repository generates embeddings for all chunks."""
        # Setup: two chunks per file call
        chunks = [
            _make_chunk(content="def hello(): pass", chunk_id="chunk-1"),
            _make_chunk(content="def world(): pass", chunk_id="chunk-2"),
        ]
        mock_chunker.chunk_file.return_value = chunks
        mock_semantic_retriever.embed_chunks.return_value = [
            [0.1] * 384,
            [0.2] * 384,
        ]

        await pipeline.ingest_repository(str(temp_repo))

        # embed_chunks should be called with all collected chunks
        mock_semantic_retriever.embed_chunks.assert_called_once()
        call_args = mock_semantic_retriever.embed_chunks.call_args[0][0]
        # Should contain chunks from multiple files
        assert len(call_args) >= 2

    @pytest.mark.asyncio
    async def test_builds_bm25_index(
        self, pipeline, temp_repo, mock_bm25_engine
    ):
        """ingest_repository builds the BM25 index with all chunks."""
        await pipeline.ingest_repository(str(temp_repo))

        mock_bm25_engine.build_index.assert_called_once()
        call_args = mock_bm25_engine.build_index.call_args[0][0]
        assert len(call_args) >= 1


class TestIngestRepositoryIdempotent:
    """Tests for idempotent ingestion behavior."""

    @pytest.mark.asyncio
    async def test_deletes_existing_chunks_before_insert(
        self, pipeline, temp_repo, mock_supabase
    ):
        """Idempotent ingestion deletes existing repo chunks before inserting."""
        await pipeline.ingest_repository(str(temp_repo))

        # Should call table("code_chunks").delete().eq("repo_id", ...).execute()
        mock_supabase.table.assert_called_with("code_chunks")
        table_mock = mock_supabase.table.return_value
        table_mock.delete.assert_called()
        delete_chain = table_mock.delete.return_value
        delete_chain.eq.assert_called_with("repo_id", "test-repo")
        delete_chain.eq.return_value.execute.assert_called()

    @pytest.mark.asyncio
    async def test_inserts_chunks_with_embeddings(
        self, pipeline, temp_repo, mock_supabase, mock_chunker, mock_semantic_retriever
    ):
        """Stores chunks with their embeddings in Supabase."""
        chunk = _make_chunk(content="def hello(): pass", chunk_id="chunk-abc")
        mock_chunker.chunk_file.return_value = [chunk]
        mock_semantic_retriever.embed_chunks.return_value = [[0.5] * 384]

        await pipeline.ingest_repository(str(temp_repo))

        # Should call table("code_chunks").insert(rows).execute()
        table_mock = mock_supabase.table.return_value
        table_mock.insert.assert_called()
        insert_args = table_mock.insert.call_args[0][0]

        # Verify the inserted row structure
        assert len(insert_args) >= 1
        row = insert_args[0]
        assert row["id"] == "chunk-abc"
        assert row["repo_id"] == "test-repo"
        assert row["content"] == "def hello(): pass"
        assert row["embedding"] == [0.5] * 384
        assert row["language"] == "python"
        assert row["file_path"] == "src/main.py"

    @pytest.mark.asyncio
    async def test_same_content_produces_same_result(
        self, pipeline, temp_repo, mock_chunker, mock_semantic_retriever, mock_bm25_engine, mock_supabase
    ):
        """Running ingestion twice on same repo produces identical behavior."""
        chunk = _make_chunk(content="def hello(): pass", chunk_id="chunk-1")
        mock_chunker.chunk_file.return_value = [chunk]
        mock_semantic_retriever.embed_chunks.return_value = [[0.1] * 384]

        stats1 = await pipeline.ingest_repository(str(temp_repo))

        # Reset mocks
        mock_supabase.reset_mock()
        # Re-setup the chained calls after reset
        delete_chain = MagicMock()
        delete_chain.eq.return_value = delete_chain
        delete_chain.execute.return_value = MagicMock(data=[])
        insert_chain = MagicMock()
        insert_chain.execute.return_value = MagicMock(data=[])
        table_mock = MagicMock()
        table_mock.delete.return_value = delete_chain
        table_mock.insert.return_value = insert_chain
        mock_supabase.table.return_value = table_mock

        stats2 = await pipeline.ingest_repository(str(temp_repo))

        # Same stats both times
        assert stats1.total_files == stats2.total_files
        assert stats1.total_chunks == stats2.total_chunks
        assert stats1.total_embeddings == stats2.total_embeddings


class TestIngestRepositoryEdgeCases:
    """Tests for edge cases in the ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_empty_repository(
        self, pipeline, tmp_path, mock_chunker, mock_semantic_retriever, mock_bm25_engine
    ):
        """An empty repository returns zero stats without errors."""
        stats = await pipeline.ingest_repository(str(tmp_path))

        assert stats.total_files == 0
        assert stats.total_chunks == 0
        assert stats.total_embeddings == 0
        # Should not call embed or build_index for empty repos
        mock_semantic_retriever.embed_chunks.assert_not_called()
        mock_bm25_engine.build_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_chunks_from_files(
        self, pipeline, temp_repo, mock_chunker, mock_semantic_retriever, mock_bm25_engine
    ):
        """When chunker produces no chunks, skip embedding and indexing."""
        mock_chunker.chunk_file.return_value = []

        stats = await pipeline.ingest_repository(str(temp_repo))

        assert stats.total_files >= 2
        assert stats.total_chunks == 0
        assert stats.total_embeddings == 0
        mock_semantic_retriever.embed_chunks.assert_not_called()
        mock_bm25_engine.build_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonexistent_path_raises_error(self, pipeline):
        """ingest_repository raises FileNotFoundError for missing paths."""
        with pytest.raises(FileNotFoundError):
            await pipeline.ingest_repository("/nonexistent/repo/path")

    @pytest.mark.asyncio
    async def test_file_path_raises_error(self, pipeline, tmp_path):
        """ingest_repository raises NotADirectoryError for file paths."""
        f = tmp_path / "file.py"
        f.write_text("content")
        with pytest.raises(NotADirectoryError):
            await pipeline.ingest_repository(str(f))

    @pytest.mark.asyncio
    async def test_uses_custom_config(
        self, pipeline, temp_repo, mock_chunker
    ):
        """ingest_repository respects custom IngestionConfig."""
        config = IngestionConfig(exclude_patterns=["src/**"])

        stats = await pipeline.ingest_repository(str(temp_repo), config)

        # With src/** excluded, should find no files
        assert stats.total_files == 0

    @pytest.mark.asyncio
    async def test_unsupported_language_uses_python_default(
        self, pipeline, tmp_path, mock_chunker
    ):
        """Files with unsupported language default to 'python' for chunking."""
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Documentation\n\nSome content here.\n")

        await pipeline.ingest_repository(str(tmp_path))

        # chunk_file should be called with "python" as fallback language
        if mock_chunker.chunk_file.called:
            call_args = mock_chunker.chunk_file.call_args
            assert call_args[0][2] == "python"  # Third arg is language


class TestIngestRepositoryMissingDependencies:
    """Tests for missing dependency validation."""

    @pytest.mark.asyncio
    async def test_missing_chunker_raises_error(
        self, mock_semantic_retriever, mock_bm25_engine, mock_supabase, tmp_path
    ):
        """Raises RuntimeError when chunker is not configured."""
        pipeline = IngestionPipeline(
            chunker=None,
            semantic_retriever=mock_semantic_retriever,
            bm25_engine=mock_bm25_engine,
            supabase_client=mock_supabase,
        )
        with pytest.raises(RuntimeError, match="chunker"):
            await pipeline.ingest_repository(str(tmp_path))

    @pytest.mark.asyncio
    async def test_missing_semantic_retriever_raises_error(
        self, mock_chunker, mock_bm25_engine, mock_supabase, tmp_path
    ):
        """Raises RuntimeError when semantic_retriever is not configured."""
        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            semantic_retriever=None,
            bm25_engine=mock_bm25_engine,
            supabase_client=mock_supabase,
        )
        with pytest.raises(RuntimeError, match="semantic_retriever"):
            await pipeline.ingest_repository(str(tmp_path))

    @pytest.mark.asyncio
    async def test_missing_bm25_engine_raises_error(
        self, mock_chunker, mock_semantic_retriever, mock_supabase, tmp_path
    ):
        """Raises RuntimeError when bm25_engine is not configured."""
        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            semantic_retriever=mock_semantic_retriever,
            bm25_engine=None,
            supabase_client=mock_supabase,
        )
        with pytest.raises(RuntimeError, match="bm25_engine"):
            await pipeline.ingest_repository(str(tmp_path))

    @pytest.mark.asyncio
    async def test_missing_supabase_raises_error(
        self, mock_chunker, mock_semantic_retriever, mock_bm25_engine, tmp_path
    ):
        """Raises RuntimeError when supabase_client is not configured."""
        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            semantic_retriever=mock_semantic_retriever,
            bm25_engine=mock_bm25_engine,
            supabase_client=None,
        )
        with pytest.raises(RuntimeError, match="supabase_client"):
            await pipeline.ingest_repository(str(tmp_path))


class TestStoreChunks:
    """Tests for the _store_chunks helper method."""

    @pytest.mark.asyncio
    async def test_batch_insertion(
        self, pipeline, mock_supabase, mock_chunker, mock_semantic_retriever, temp_repo
    ):
        """Large numbers of chunks are inserted in batches."""
        # Create 150 chunks to test batching (batch_size=100)
        chunks = [
            _make_chunk(content=f"def func_{i}(): pass", chunk_id=f"chunk-{i}")
            for i in range(150)
        ]
        mock_chunker.chunk_file.return_value = chunks
        mock_semantic_retriever.embed_chunks.return_value = [[0.1] * 384] * 150

        await pipeline.ingest_repository(str(temp_repo))

        # Should have multiple insert calls due to batching
        table_mock = mock_supabase.table.return_value
        insert_calls = table_mock.insert.call_args_list
        # First file produces 150 chunks, but there are 2 files = 300 chunks
        # 300 / 100 = 3 batches
        assert len(insert_calls) >= 2

    @pytest.mark.asyncio
    async def test_chunk_row_contains_embedding(
        self, pipeline, mock_supabase, mock_chunker, mock_semantic_retriever, temp_repo
    ):
        """Each stored row includes the embedding vector."""
        embedding = [0.42] * 384
        mock_chunker.chunk_file.return_value = [
            _make_chunk(content="def test(): pass", chunk_id="c-1")
        ]
        mock_semantic_retriever.embed_chunks.return_value = [embedding]

        await pipeline.ingest_repository(str(temp_repo))

        table_mock = mock_supabase.table.return_value
        insert_args = table_mock.insert.call_args[0][0]
        row = insert_args[0]
        assert row["embedding"] == embedding

    @pytest.mark.asyncio
    async def test_chunk_row_contains_metadata(
        self, pipeline, mock_supabase, mock_chunker, mock_semantic_retriever, temp_repo
    ):
        """Each stored row includes chunk metadata fields."""
        chunk = _make_chunk(
            content="def hello(): pass",
            chunk_id="c-meta",
            file_path="src/utils.py",
        )
        chunk.metadata.function_name = "hello"
        chunk.metadata.class_name = "Utils"
        mock_chunker.chunk_file.return_value = [chunk]
        mock_semantic_retriever.embed_chunks.return_value = [[0.1] * 384]

        await pipeline.ingest_repository(str(temp_repo))

        table_mock = mock_supabase.table.return_value
        insert_args = table_mock.insert.call_args[0][0]
        row = insert_args[0]
        assert row["function_name"] == "hello"
        assert row["class_name"] == "Utils"
        assert row["start_line"] == 1
        assert row["end_line"] == 1
        assert row["chunk_type"] == "function"
