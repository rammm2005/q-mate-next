"""Answer generator service for constructing LLM prompts from retrieved context.

Handles chunk selection within token budgets, structured prompt construction
with [Source N] format, LLM instructions for source-grounded answers,
and end-to-end answer generation with retry and fallback logic.
"""

import asyncio
import re

from app.models.answer import GroundedAnswer, SourceReference
from app.models.retrieval import RetrievalResult
from app.utils.tokenizer import estimate_tokens


class AnswerGenerator:
    """Generates source-grounded answers from retrieved code chunks.

    Selects chunks within a token budget, builds structured prompts with
    [Source N] labels, and instructs the LLM to cite sources exclusively
    using [Source N] format.
    """

    def __init__(self, llm_client=None, max_context_tokens: int = 4096):
        """Initialize the AnswerGenerator.

        Args:
            llm_client: Optional LLM client for generating answers.
            max_context_tokens: Maximum number of tokens allowed for context
                chunks in the prompt. Defaults to 4096.
        """
        self.llm_client = llm_client
        self.max_context_tokens = max_context_tokens

    def select_chunks(
        self, retrieved_chunks: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Select chunks within token budget, sorted by fused_score descending.

        Processes chunks in descending fused_score order, accumulating chunks
        until adding the next chunk would exceed max_context_tokens. If the
        first chunk alone exceeds the budget, it is truncated to fit.

        Args:
            retrieved_chunks: List of retrieval results to select from.

        Returns:
            List of selected RetrievalResult objects that fit within budget.
        """
        if not retrieved_chunks:
            return []

        # Sort by fused_score descending
        sorted_chunks = sorted(
            retrieved_chunks, key=lambda r: r.fused_score, reverse=True
        )

        selected: list[RetrievalResult] = []
        total_tokens = 0

        for i, result in enumerate(sorted_chunks):
            content = result.chunk.content
            chunk_tokens = estimate_tokens(content)

            if i == 0 and chunk_tokens > self.max_context_tokens:
                # First chunk exceeds budget: truncate it to fit
                truncated_content = self._truncate_to_fit(
                    content, self.max_context_tokens
                )
                truncated_result = RetrievalResult(
                    chunk=result.chunk.model_copy(
                        update={"content": truncated_content}
                    ),
                    fused_score=result.fused_score,
                    bm25_rank=result.bm25_rank,
                    semantic_rank=result.semantic_rank,
                    context_snippet=result.context_snippet,
                )
                selected.append(truncated_result)
                break

            if total_tokens + chunk_tokens > self.max_context_tokens:
                # Adding this chunk would exceed budget, stop
                break

            selected.append(result)
            total_tokens += chunk_tokens

        return selected

    def build_context_prompt(
        self, question: str, selected_chunks: list[RetrievalResult]
    ) -> str:
        """Build structured LLM prompt with [Source N] labels.

        Constructs a prompt that includes system instructions for source
        citation, the developer's question, and each selected chunk labeled
        as [Source 1], [Source 2], etc.

        Args:
            question: The developer's original question.
            selected_chunks: Chunks selected within the token budget.

        Returns:
            A formatted prompt string ready for LLM consumption.
        """
        system_instruction = (
            "You are a code assistant. Answer the developer's question using "
            "ONLY the provided source code and documentation. You MUST cite "
            "sources using [Source N] format. If the answer cannot be found "
            "in the sources, say so."
        )

        context_sections = []
        for idx, result in enumerate(selected_chunks, start=1):
            chunk = result.chunk
            function_name = (
                chunk.metadata.function_name
                if chunk.metadata.function_name
                else "N/A"
            )
            section = (
                f"[Source {idx}] File: {chunk.file_path}\n"
                f"Function: {function_name}\n"
                f"Lines: {chunk.start_line}-{chunk.end_line}\n"
                f"```{chunk.language}\n{chunk.content}\n```"
            )
            context_sections.append(section)

        sources_text = "\n\n".join(context_sections)

        prompt = (
            f"{system_instruction}\n\n"
            f"Question: {question}\n\n"
            f"Sources:\n{sources_text}"
        )

        return prompt

    def extract_sources(
        self, answer_text: str, selected_chunks: list[RetrievalResult]
    ) -> tuple[list[SourceReference], float]:
        """Extract and validate [Source N] references from LLM output.

        Parses all [Source N] patterns from the answer text, validates that
        each N maps to a valid 1-indexed position in selected_chunks, and
        computes a confidence score based on the ratio of validated citations
        to total citations found.

        Args:
            answer_text: The raw LLM-generated answer text containing
                [Source N] references.
            selected_chunks: The list of chunks that were provided as context
                to the LLM (1-indexed in the prompt).

        Returns:
            Tuple of (valid_sources, confidence_score) where valid_sources
            is a list of SourceReference objects for validated citations,
            and confidence_score is validated_citations / total_citations
            (0.0 if no citations found).
        """
        # Find all [Source N] patterns
        pattern = r"\[Source (\d+)\]"
        matches = re.findall(pattern, answer_text)

        total_citations = len(matches)
        if total_citations == 0:
            return [], 0.0

        valid_sources: list[SourceReference] = []
        seen_indices: set[int] = set()

        for match in matches:
            n = int(match)
            if n < 1 or n > len(selected_chunks):
                continue  # Invalid reference, skip
            if n in seen_indices:
                continue  # Already processed
            seen_indices.add(n)

            chunk = selected_chunks[n - 1].chunk  # 1-indexed
            source = SourceReference(
                file_path=chunk.file_path,
                function_name=chunk.metadata.function_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                snippet=chunk.content[:200],  # First 200 chars
                relevance=selected_chunks[n - 1].fused_score,
            )
            valid_sources.append(source)

        confidence = len(seen_indices) / total_citations if total_citations > 0 else 0.0
        return valid_sources, confidence

    async def generate_answer(
        self, question: str, retrieved_chunks: list[RetrievalResult]
    ) -> GroundedAnswer:
        """Generate a source-grounded answer from retrieved context.

        The LLM is used to make answers more readable - it takes retrieved
        code chunks and generates a natural language explanation. On LLM
        failure, falls back to showing raw sources.

        Args:
            question: The developer's original question.
            retrieved_chunks: List of retrieval results from the hybrid
                retriever, sorted by fused_score descending.

        Returns:
            A GroundedAnswer containing the answer text, source references,
            confidence score, and retrieval metadata.
        """
        if not retrieved_chunks:
            return GroundedAnswer(
                answer_text="No relevant sources were found for your question.",
                sources=[],
                confidence=0.0,
                retrieval_metadata={"chunks_used": 0},
            )

        # Select chunks within token budget
        selected_chunks = self.select_chunks(retrieved_chunks)

        # Build prompt
        prompt = self.build_context_prompt(question, selected_chunks)

        # Try LLM generation with retry
        raw_answer = await self._call_llm_with_retry(prompt)

        if raw_answer is None:
            # LLM failed - return fallback with just the sources
            return self._build_fallback_answer(question, selected_chunks)

        # Extract and validate source references
        sources, confidence = self.extract_sources(raw_answer, selected_chunks)

        return GroundedAnswer(
            answer_text=raw_answer,
            sources=sources,
            confidence=confidence,
            retrieval_metadata={
                "chunks_used": len(selected_chunks),
                "total_tokens": sum(
                    estimate_tokens(c.chunk.content) for c in selected_chunks
                ),
            },
        )

    async def _call_llm_with_retry(
        self, prompt: str, max_retries: int = 3
    ) -> str | None:
        """Call LLM with exponential backoff retry.

        Retries up to max_retries times with delays of 1s, 2s, 4s
        (doubling each attempt). Returns None on total failure.

        Args:
            prompt: The prompt to send to the LLM.
            max_retries: Maximum number of attempts (default 3).

        Returns:
            The LLM response string, or None if all attempts failed.
        """
        if self.llm_client is None:
            return None

        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.generate(prompt)
                return response
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
        return None

    def _build_fallback_answer(
        self, question: str, selected_chunks: list[RetrievalResult]
    ) -> GroundedAnswer:
        """Build a fallback answer showing raw sources when LLM is unavailable.

        Constructs a human-readable answer listing the found sources with
        file paths and line ranges, plus creates proper SourceReference
        objects for each chunk.

        Args:
            question: The developer's original question.
            selected_chunks: The chunks that were selected within token budget.

        Returns:
            A GroundedAnswer with confidence 0.0 and a fallback flag in metadata.
        """
        source_texts = []
        sources = []
        for i, result in enumerate(selected_chunks, 1):
            chunk = result.chunk
            source_texts.append(
                f"[Source {i}] {chunk.file_path} "
                f"(lines {chunk.start_line}-{chunk.end_line})"
            )
            sources.append(
                SourceReference(
                    file_path=chunk.file_path,
                    function_name=chunk.metadata.function_name,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    snippet=chunk.content[:200],
                    relevance=result.fused_score,
                )
            )

        answer_text = (
            "I found relevant sources but couldn't generate a natural language "
            "answer. Here are the raw sources:\n\n"
            + "\n".join(source_texts)
        )

        return GroundedAnswer(
            answer_text=answer_text,
            sources=sources,
            confidence=0.0,
            retrieval_metadata={
                "chunks_used": len(selected_chunks),
                "fallback": True,
            },
        )

    def _truncate_to_fit(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within the given token budget.

        Uses a binary search approach to find the longest prefix of the
        content that fits within max_tokens.

        Args:
            content: The text content to truncate.
            max_tokens: Maximum number of tokens allowed.

        Returns:
            Truncated content that fits within the token budget.
        """
        if estimate_tokens(content) <= max_tokens:
            return content

        # Binary search for the longest fitting prefix
        # Use characters as the search space
        low = 0
        high = len(content)

        while low < high:
            mid = (low + high + 1) // 2
            if estimate_tokens(content[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1

        return content[:low]
