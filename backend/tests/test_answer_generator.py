"""Unit tests for the AnswerGenerator service."""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.answer import GroundedAnswer, SourceReference
from app.models.chunk import ChunkMetadata, ChunkType, CodeChunk
from app.models.retrieval import RetrievalResult
from app.services.answer_generator import AnswerGenerator
from app.utils.tokenizer import estimate_tokens


def _make_chunk(
    content: str = "def hello(): pass",
    chunk_id: str = "chunk-1",
    file_path: str = "src/main.py",
    language: str = "python",
    function_name: str | None = "hello",
    start_line: int = 1,
    end_line: int = 5,
) -> CodeChunk:
    """Helper to create a CodeChunk with sensible defaults."""
    return CodeChunk(
        id=chunk_id,
        repo_id="repo-1",
        file_path=file_path,
        language=language,
        chunk_type=ChunkType.FUNCTION,
        content=content,
        start_line=start_line,
        end_line=end_line,
        metadata=ChunkMetadata(function_name=function_name),
    )


def _make_retrieval_result(
    content: str = "def hello(): pass",
    fused_score: float = 0.8,
    chunk_id: str = "chunk-1",
    file_path: str = "src/main.py",
    function_name: str | None = "hello",
    start_line: int = 1,
    end_line: int = 5,
) -> RetrievalResult:
    """Helper to create a RetrievalResult with sensible defaults."""
    chunk = _make_chunk(
        content=content,
        chunk_id=chunk_id,
        file_path=file_path,
        function_name=function_name,
        start_line=start_line,
        end_line=end_line,
    )
    return RetrievalResult(
        chunk=chunk,
        fused_score=fused_score,
        bm25_rank=1,
        semantic_rank=2,
        context_snippet=content,
    )


class TestSelectChunks:
    """Tests for AnswerGenerator.select_chunks."""

    def test_empty_input_returns_empty(self):
        """An empty chunk list should produce no selected chunks."""
        gen = AnswerGenerator(max_context_tokens=4096)
        result = gen.select_chunks([])
        assert result == []

    def test_single_chunk_within_budget(self):
        """A single chunk fitting within budget is selected."""
        gen = AnswerGenerator(max_context_tokens=4096)
        chunk = _make_retrieval_result(content="def foo(): return 42")
        result = gen.select_chunks([chunk])
        assert len(result) == 1
        assert result[0].chunk.content == "def foo(): return 42"

    def test_chunks_sorted_by_fused_score_descending(self):
        """Chunks should be selected in descending fused_score order."""
        gen = AnswerGenerator(max_context_tokens=4096)
        low = _make_retrieval_result(
            content="low score chunk", fused_score=0.3, chunk_id="low"
        )
        high = _make_retrieval_result(
            content="high score chunk", fused_score=0.9, chunk_id="high"
        )
        mid = _make_retrieval_result(
            content="mid score chunk", fused_score=0.6, chunk_id="mid"
        )
        result = gen.select_chunks([low, high, mid])
        assert result[0].fused_score == 0.9
        assert result[1].fused_score == 0.6
        assert result[2].fused_score == 0.3

    def test_stops_before_exceeding_budget(self):
        """Should stop adding chunks before exceeding the token budget."""
        # Use a small budget
        gen = AnswerGenerator(max_context_tokens=20)
        chunk1 = _make_retrieval_result(
            content="short", fused_score=0.9, chunk_id="c1"
        )
        # Create a chunk with enough content to exceed budget
        chunk2 = _make_retrieval_result(
            content="this is a much longer piece of content that should exceed the remaining token budget easily",
            fused_score=0.5,
            chunk_id="c2",
        )
        result = gen.select_chunks([chunk1, chunk2])
        # First chunk should fit, second should be excluded
        total_tokens = sum(estimate_tokens(r.chunk.content) for r in result)
        assert total_tokens <= 20

    def test_first_chunk_truncated_when_exceeds_budget(self):
        """If the first chunk exceeds the budget, it should be truncated."""
        gen = AnswerGenerator(max_context_tokens=10)
        # Create content that will exceed 10 tokens
        long_content = "def calculate_average_of_numbers(numbers_list): return sum(numbers_list) / len(numbers_list) if numbers_list else 0"
        chunk = _make_retrieval_result(
            content=long_content, fused_score=0.9, chunk_id="big"
        )
        result = gen.select_chunks([chunk])
        assert len(result) == 1
        # The truncated content should fit within budget
        assert estimate_tokens(result[0].chunk.content) <= 10
        # The content should be a prefix of the original
        assert long_content.startswith(result[0].chunk.content)

    def test_multiple_chunks_accumulate_tokens(self):
        """Multiple small chunks should accumulate until budget reached."""
        gen = AnswerGenerator(max_context_tokens=50)
        chunks = [
            _make_retrieval_result(
                content=f"def func_{i}(): pass",
                fused_score=1.0 - (i * 0.1),
                chunk_id=f"c{i}",
            )
            for i in range(10)
        ]
        result = gen.select_chunks(chunks)
        total_tokens = sum(estimate_tokens(r.chunk.content) for r in result)
        assert total_tokens <= 50
        assert len(result) >= 1
        assert len(result) < 10  # Not all should fit


class TestBuildContextPrompt:
    """Tests for AnswerGenerator.build_context_prompt."""

    def test_includes_system_instruction(self):
        """The prompt should include the system instruction text."""
        gen = AnswerGenerator()
        chunks = [_make_retrieval_result()]
        prompt = gen.build_context_prompt("How does hello work?", chunks)
        assert "You are a code assistant" in prompt
        assert "ONLY the provided source code and documentation" in prompt
        assert "[Source N] format" in prompt

    def test_includes_question(self):
        """The prompt should include the developer's question."""
        gen = AnswerGenerator()
        question = "How does the authentication module work?"
        chunks = [_make_retrieval_result()]
        prompt = gen.build_context_prompt(question, chunks)
        assert f"Question: {question}" in prompt

    def test_labels_chunks_with_source_n_format(self):
        """Each chunk should be labeled as [Source N]."""
        gen = AnswerGenerator()
        chunks = [
            _make_retrieval_result(
                content="def foo(): pass", chunk_id="c1", fused_score=0.9
            ),
            _make_retrieval_result(
                content="def bar(): pass",
                chunk_id="c2",
                fused_score=0.7,
                file_path="src/utils.py",
                function_name="bar",
            ),
        ]
        prompt = gen.build_context_prompt("What do foo and bar do?", chunks)
        assert "[Source 1]" in prompt
        assert "[Source 2]" in prompt

    def test_includes_file_path_and_metadata(self):
        """The prompt should include file path, function name, and line numbers."""
        gen = AnswerGenerator()
        chunks = [
            _make_retrieval_result(
                file_path="src/auth.py",
                function_name="authenticate",
                start_line=10,
                end_line=25,
            )
        ]
        prompt = gen.build_context_prompt("How does auth work?", chunks)
        assert "File: src/auth.py" in prompt
        assert "Function: authenticate" in prompt
        assert "Lines: 10-25" in prompt

    def test_includes_chunk_content_in_code_block(self):
        """Chunk content should appear in a fenced code block with language."""
        gen = AnswerGenerator()
        content = "def greet(name):\n    return f'Hello, {name}'"
        chunks = [_make_retrieval_result(content=content)]
        prompt = gen.build_context_prompt("How does greet work?", chunks)
        assert "```python" in prompt
        assert content in prompt

    def test_empty_chunks_produces_minimal_prompt(self):
        """An empty chunk list should still produce a valid prompt structure."""
        gen = AnswerGenerator()
        prompt = gen.build_context_prompt("What is this?", [])
        assert "You are a code assistant" in prompt
        assert "Question: What is this?" in prompt
        assert "Sources:\n" in prompt

    def test_function_name_none_shows_na(self):
        """When function_name is None, it should display N/A."""
        gen = AnswerGenerator()
        chunks = [_make_retrieval_result(function_name=None)]
        prompt = gen.build_context_prompt("question?", chunks)
        assert "Function: N/A" in prompt

    def test_citation_instruction_present(self):
        """The prompt must instruct the LLM to cite using [Source N] format."""
        gen = AnswerGenerator()
        chunks = [_make_retrieval_result()]
        prompt = gen.build_context_prompt("test?", chunks)
        assert "You MUST cite sources using [Source N] format" in prompt

    def test_only_use_provided_sources_instruction(self):
        """The prompt must instruct to only use provided sources."""
        gen = AnswerGenerator()
        chunks = [_make_retrieval_result()]
        prompt = gen.build_context_prompt("test?", chunks)
        assert "cannot be found in the sources, say so" in prompt


class TestTruncateToFit:
    """Tests for the internal _truncate_to_fit method."""

    def test_content_within_budget_unchanged(self):
        """Content already within budget should not be modified."""
        gen = AnswerGenerator()
        content = "short text"
        result = gen._truncate_to_fit(content, max_tokens=100)
        assert result == content

    def test_long_content_is_truncated(self):
        """Long content should be truncated to fit within budget."""
        gen = AnswerGenerator()
        long_content = "word " * 500  # Many tokens
        result = gen._truncate_to_fit(long_content, max_tokens=10)
        assert estimate_tokens(result) <= 10
        assert len(result) < len(long_content)

    def test_truncated_is_prefix_of_original(self):
        """Truncated content should be a prefix of the original."""
        gen = AnswerGenerator()
        content = "def foo(x, y, z): return x + y + z  # adds three numbers together"
        result = gen._truncate_to_fit(content, max_tokens=5)
        assert content.startswith(result)


class TestIntegration:
    """Integration tests combining select_chunks and build_context_prompt."""

    def test_select_then_build_prompt(self):
        """Full flow: select chunks within budget, then build prompt."""
        gen = AnswerGenerator(max_context_tokens=100)
        chunks = [
            _make_retrieval_result(
                content="def add(a, b): return a + b",
                fused_score=0.9,
                chunk_id="c1",
                function_name="add",
            ),
            _make_retrieval_result(
                content="def subtract(a, b): return a - b",
                fused_score=0.7,
                chunk_id="c2",
                function_name="subtract",
            ),
        ]
        selected = gen.select_chunks(chunks)
        prompt = gen.build_context_prompt("How do math functions work?", selected)

        assert "[Source 1]" in prompt
        assert "add" in prompt
        assert "Question: How do math functions work?" in prompt


class TestExtractSources:
    """Tests for AnswerGenerator.extract_sources."""

    def test_no_citations_returns_empty_with_zero_confidence(self):
        """Answer with no [Source N] references returns empty list and 0.0."""
        gen = AnswerGenerator()
        chunks = [_make_retrieval_result()]
        sources, confidence = gen.extract_sources(
            "This is an answer without any citations.", chunks
        )
        assert sources == []
        assert confidence == 0.0

    def test_single_valid_citation(self):
        """A single valid [Source 1] reference is extracted correctly."""
        gen = AnswerGenerator()
        chunks = [
            _make_retrieval_result(
                content="def hello(): pass",
                fused_score=0.9,
                file_path="src/main.py",
                function_name="hello",
                start_line=1,
                end_line=5,
            )
        ]
        answer = "The function is defined here [Source 1]."
        sources, confidence = gen.extract_sources(answer, chunks)

        assert len(sources) == 1
        assert sources[0].file_path == "src/main.py"
        assert sources[0].function_name == "hello"
        assert sources[0].start_line == 1
        assert sources[0].end_line == 5
        assert sources[0].relevance == 0.9
        assert confidence == 1.0

    def test_multiple_valid_citations(self):
        """Multiple valid citations are all extracted."""
        gen = AnswerGenerator()
        chunks = [
            _make_retrieval_result(
                content="def foo(): pass",
                fused_score=0.9,
                chunk_id="c1",
                file_path="src/foo.py",
                function_name="foo",
                start_line=1,
                end_line=3,
            ),
            _make_retrieval_result(
                content="def bar(): pass",
                fused_score=0.7,
                chunk_id="c2",
                file_path="src/bar.py",
                function_name="bar",
                start_line=10,
                end_line=15,
            ),
        ]
        answer = "foo is at [Source 1] and bar is at [Source 2]."
        sources, confidence = gen.extract_sources(answer, chunks)

        assert len(sources) == 2
        assert sources[0].file_path == "src/foo.py"
        assert sources[1].file_path == "src/bar.py"
        assert confidence == 1.0

    def test_invalid_citation_index_too_high(self):
        """A [Source N] where N exceeds chunk count is skipped."""
        gen = AnswerGenerator()
        chunks = [_make_retrieval_result()]
        answer = "See [Source 1] and [Source 5]."
        sources, confidence = gen.extract_sources(answer, chunks)

        assert len(sources) == 1
        assert sources[0].file_path == "src/main.py"
        # 1 valid out of 2 total
        assert confidence == 0.5

    def test_invalid_citation_index_zero(self):
        """A [Source 0] reference is invalid since indexing is 1-based."""
        gen = AnswerGenerator()
        chunks = [_make_retrieval_result()]
        answer = "See [Source 0] for details."
        sources, confidence = gen.extract_sources(answer, chunks)

        assert sources == []
        # 0 valid out of 1 total citation
        assert confidence == 0.0

    def test_duplicate_citations_counted_correctly(self):
        """Duplicate [Source N] references are deduplicated in sources list."""
        gen = AnswerGenerator()
        chunks = [
            _make_retrieval_result(
                content="def hello(): pass",
                fused_score=0.9,
                file_path="src/main.py",
                function_name="hello",
            )
        ]
        answer = "First [Source 1], and again [Source 1]."
        sources, confidence = gen.extract_sources(answer, chunks)

        # Only one source reference despite two mentions
        assert len(sources) == 1
        # 1 unique valid index / 2 total citations
        assert confidence == 0.5

    def test_mixed_valid_and_invalid_citations(self):
        """Mix of valid and invalid citations produces correct confidence."""
        gen = AnswerGenerator()
        chunks = [
            _make_retrieval_result(
                content="def a(): pass", fused_score=0.9, chunk_id="c1"
            ),
            _make_retrieval_result(
                content="def b(): pass", fused_score=0.7, chunk_id="c2"
            ),
        ]
        # Source 1 valid, Source 3 invalid, Source 2 valid
        answer = "See [Source 1], [Source 3], and [Source 2]."
        sources, confidence = gen.extract_sources(answer, chunks)

        assert len(sources) == 2
        # 2 unique valid / 3 total citations
        assert abs(confidence - 2.0 / 3.0) < 1e-9

    def test_snippet_truncated_to_200_chars(self):
        """Source snippet is truncated to first 200 characters of content."""
        gen = AnswerGenerator()
        long_content = "x" * 500
        chunks = [
            _make_retrieval_result(content=long_content, fused_score=0.8)
        ]
        answer = "See [Source 1]."
        sources, confidence = gen.extract_sources(answer, chunks)

        assert len(sources) == 1
        assert len(sources[0].snippet) == 200
        assert sources[0].snippet == "x" * 200

    def test_function_name_none_preserved(self):
        """When chunk has no function_name, source reference has None."""
        gen = AnswerGenerator()
        chunks = [
            _make_retrieval_result(function_name=None, fused_score=0.8)
        ]
        answer = "See [Source 1]."
        sources, confidence = gen.extract_sources(answer, chunks)

        assert len(sources) == 1
        assert sources[0].function_name is None

    def test_empty_answer_text(self):
        """Empty answer text has no citations."""
        gen = AnswerGenerator()
        chunks = [_make_retrieval_result()]
        sources, confidence = gen.extract_sources("", chunks)

        assert sources == []
        assert confidence == 0.0

    def test_empty_chunks_list_all_citations_invalid(self):
        """With empty chunks list, all citations are invalid."""
        gen = AnswerGenerator()
        answer = "See [Source 1] and [Source 2]."
        sources, confidence = gen.extract_sources(answer, [])

        assert sources == []
        assert confidence == 0.0

    def test_relevance_uses_fused_score(self):
        """SourceReference.relevance matches the chunk's fused_score."""
        gen = AnswerGenerator()
        chunks = [
            _make_retrieval_result(fused_score=0.42, chunk_id="c1"),
            _make_retrieval_result(fused_score=0.87, chunk_id="c2"),
        ]
        answer = "See [Source 2]."
        sources, confidence = gen.extract_sources(answer, chunks)

        assert len(sources) == 1
        assert sources[0].relevance == 0.87

    def test_source_references_are_valid_models(self):
        """Extracted sources are proper SourceReference instances."""
        gen = AnswerGenerator()
        chunks = [_make_retrieval_result(fused_score=0.8)]
        answer = "See [Source 1]."
        sources, confidence = gen.extract_sources(answer, chunks)

        assert len(sources) == 1
        assert isinstance(sources[0], SourceReference)


class TestGenerateAnswer:
    """Tests for AnswerGenerator.generate_answer end-to-end method."""

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_no_sources_answer(self):
        """Empty retrieved_chunks returns a 'no relevant sources' answer."""
        gen = AnswerGenerator(llm_client=None)
        result = await gen.generate_answer("How does auth work?", [])
        assert isinstance(result, GroundedAnswer)
        assert "No relevant sources were found" in result.answer_text
        assert result.sources == []
        assert result.confidence == 0.0
        assert result.retrieval_metadata["chunks_used"] == 0

    @pytest.mark.asyncio
    async def test_successful_llm_generation(self):
        """Successful LLM call returns grounded answer with sources."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = (
            "The function adds two numbers [Source 1]."
        )
        gen = AnswerGenerator(llm_client=mock_llm, max_context_tokens=4096)
        chunks = [
            _make_retrieval_result(
                content="def add(a, b): return a + b",
                fused_score=0.9,
                chunk_id="c1",
                function_name="add",
            )
        ]
        result = await gen.generate_answer("What does add do?", chunks)
        assert isinstance(result, GroundedAnswer)
        assert "adds two numbers" in result.answer_text
        assert len(result.sources) == 1
        assert result.sources[0].file_path == "src/main.py"
        assert result.confidence == 1.0  # 1 unique valid / 1 total citation
        assert result.retrieval_metadata["chunks_used"] == 1

    @pytest.mark.asyncio
    async def test_llm_failure_returns_fallback(self):
        """When LLM fails after retries, returns fallback answer."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("LLM timeout")
        gen = AnswerGenerator(llm_client=mock_llm, max_context_tokens=4096)
        chunks = [
            _make_retrieval_result(
                content="def foo(): pass",
                fused_score=0.8,
                chunk_id="c1",
                file_path="src/foo.py",
                function_name="foo",
                start_line=10,
                end_line=15,
            )
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gen.generate_answer("What does foo do?", chunks)
        assert isinstance(result, GroundedAnswer)
        assert "couldn't generate a natural language answer" in result.answer_text
        assert "src/foo.py" in result.answer_text
        assert result.confidence == 0.0
        assert result.retrieval_metadata.get("fallback") is True
        assert len(result.sources) == 1
        assert result.sources[0].file_path == "src/foo.py"

    @pytest.mark.asyncio
    async def test_no_llm_client_returns_fallback(self):
        """When no LLM client is configured, returns fallback answer."""
        gen = AnswerGenerator(llm_client=None, max_context_tokens=4096)
        chunks = [
            _make_retrieval_result(
                content="def bar(): pass",
                fused_score=0.7,
                chunk_id="c1",
                function_name="bar",
            )
        ]
        result = await gen.generate_answer("What does bar do?", chunks)
        assert isinstance(result, GroundedAnswer)
        assert "couldn't generate a natural language answer" in result.answer_text
        assert result.confidence == 0.0
        assert result.retrieval_metadata.get("fallback") is True

    @pytest.mark.asyncio
    async def test_metadata_includes_token_count(self):
        """Successful generation includes total_tokens in metadata."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "The answer is [Source 1]."
        gen = AnswerGenerator(llm_client=mock_llm, max_context_tokens=4096)
        content = "def hello(): return 'hi'"
        chunks = [
            _make_retrieval_result(content=content, fused_score=0.9, chunk_id="c1")
        ]
        result = await gen.generate_answer("What does hello do?", chunks)
        expected_tokens = estimate_tokens(content)
        assert result.retrieval_metadata["total_tokens"] == expected_tokens

    @pytest.mark.asyncio
    async def test_multiple_chunks_all_cited(self):
        """Multiple chunks are used and cited correctly."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = (
            "Function add [Source 1] is used by calculate [Source 2]."
        )
        gen = AnswerGenerator(llm_client=mock_llm, max_context_tokens=4096)
        chunks = [
            _make_retrieval_result(
                content="def add(a, b): return a + b",
                fused_score=0.9,
                chunk_id="c1",
                function_name="add",
            ),
            _make_retrieval_result(
                content="def calculate(x): return add(x, 1)",
                fused_score=0.7,
                chunk_id="c2",
                function_name="calculate",
                file_path="src/calc.py",
            ),
        ]
        result = await gen.generate_answer("How does calculate work?", chunks)
        assert len(result.sources) == 2
        assert result.confidence == 1.0
        assert result.retrieval_metadata["chunks_used"] == 2


class TestCallLlmWithRetry:
    """Tests for AnswerGenerator._call_llm_with_retry."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """LLM call succeeds on first attempt without retry."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "The answer is 42."
        gen = AnswerGenerator(llm_client=mock_llm)
        result = await gen._call_llm_with_retry("test prompt")
        assert result == "The answer is 42."
        assert mock_llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_success_on_second_attempt(self):
        """LLM call fails first, then succeeds on second attempt."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            Exception("timeout"),
            "Success on retry",
        ]
        gen = AnswerGenerator(llm_client=mock_llm)
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await gen._call_llm_with_retry("test prompt")
        assert result == "Success on retry"
        assert mock_llm.generate.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_success_on_third_attempt(self):
        """LLM call fails twice, then succeeds on third attempt."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            Exception("error 1"),
            Exception("error 2"),
            "Third time's the charm",
        ]
        gen = AnswerGenerator(llm_client=mock_llm)
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await gen._call_llm_with_retry("test prompt")
        assert result == "Third time's the charm"
        assert mock_llm.generate.call_count == 3
        # Should have slept with 1s then 2s delays
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_returns_none(self):
        """When all retries fail, returns None."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("persistent error")
        gen = AnswerGenerator(llm_client=mock_llm)
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await gen._call_llm_with_retry("test prompt")
        assert result is None
        assert mock_llm.generate.call_count == 3
        # Sleeps between attempts: 1s and 2s (not after last attempt)
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Retry delays follow exponential backoff: 1s, 2s, 4s pattern."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("error")
        gen = AnswerGenerator(llm_client=mock_llm)
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await gen._call_llm_with_retry("test prompt", max_retries=3)
        assert result is None
        # Delays should be 1.0 and 2.0 (no sleep after final attempt)
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_no_llm_client_returns_none_immediately(self):
        """When llm_client is None, returns None without retrying."""
        gen = AnswerGenerator(llm_client=None)
        result = await gen._call_llm_with_retry("test prompt")
        assert result is None


class TestBuildFallbackAnswer:
    """Tests for AnswerGenerator._build_fallback_answer."""

    def test_single_chunk_fallback(self):
        """Fallback answer includes source info for a single chunk."""
        gen = AnswerGenerator()
        chunks = [
            _make_retrieval_result(
                content="def hello(): print('hi')",
                fused_score=0.8,
                chunk_id="c1",
                file_path="src/greet.py",
                function_name="hello",
                start_line=5,
                end_line=6,
            )
        ]
        result = gen._build_fallback_answer("What does hello do?", chunks)
        assert isinstance(result, GroundedAnswer)
        assert "couldn't generate a natural language answer" in result.answer_text
        assert "src/greet.py" in result.answer_text
        assert "(lines 5-6)" in result.answer_text
        assert result.confidence == 0.0
        assert result.retrieval_metadata["fallback"] is True
        assert result.retrieval_metadata["chunks_used"] == 1
        assert len(result.sources) == 1
        assert result.sources[0].file_path == "src/greet.py"
        assert result.sources[0].function_name == "hello"
        assert result.sources[0].start_line == 5
        assert result.sources[0].end_line == 6
        assert result.sources[0].relevance == 0.8

    def test_multiple_chunks_fallback(self):
        """Fallback answer includes all source chunks."""
        gen = AnswerGenerator()
        chunks = [
            _make_retrieval_result(
                content="def foo(): pass",
                fused_score=0.9,
                chunk_id="c1",
                file_path="src/a.py",
                function_name="foo",
                start_line=1,
                end_line=2,
            ),
            _make_retrieval_result(
                content="def bar(): pass",
                fused_score=0.7,
                chunk_id="c2",
                file_path="src/b.py",
                function_name="bar",
                start_line=10,
                end_line=11,
            ),
        ]
        result = gen._build_fallback_answer("question?", chunks)
        assert "[Source 1]" in result.answer_text
        assert "[Source 2]" in result.answer_text
        assert "src/a.py" in result.answer_text
        assert "src/b.py" in result.answer_text
        assert len(result.sources) == 2

    def test_snippet_truncated_to_200_chars(self):
        """Source snippet is truncated to first 200 characters."""
        gen = AnswerGenerator()
        long_content = "x" * 500
        chunks = [
            _make_retrieval_result(
                content=long_content,
                fused_score=0.8,
                chunk_id="c1",
            )
        ]
        result = gen._build_fallback_answer("question?", chunks)
        assert len(result.sources[0].snippet) == 200
