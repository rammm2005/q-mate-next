"""Tests for the token estimation utility."""

import pytest

from app.utils.tokenizer import estimate_tokens


class TestEstimateTokens:
    """Unit tests for estimate_tokens function."""

    def test_empty_string_returns_zero(self):
        """Empty string should return 0 tokens."""
        assert estimate_tokens("") == 0

    def test_single_word(self):
        """A single common word should return a positive integer."""
        result = estimate_tokens("hello")
        assert isinstance(result, int)
        assert result > 0

    def test_returns_non_negative_integer(self):
        """Result must always be a non-negative integer."""
        texts = [
            "",
            " ",
            "a",
            "hello world",
            "def foo(x): return x + 1",
            "class MyClass:\n    pass",
            "x" * 10000,
        ]
        for text in texts:
            result = estimate_tokens(text)
            assert isinstance(result, int)
            assert result >= 0

    def test_longer_text_more_tokens(self):
        """Longer text should generally produce more tokens."""
        short = estimate_tokens("hello")
        long = estimate_tokens("hello world this is a longer piece of text with more words")
        assert long > short

    def test_code_snippet(self):
        """Code snippets should produce reasonable token counts."""
        code = "def calculate_sum(numbers: list[int]) -> int:\n    return sum(numbers)"
        result = estimate_tokens(code)
        assert isinstance(result, int)
        assert result > 0

    def test_whitespace_only_returns_positive(self):
        """Whitespace-only strings may produce tokens (spaces are tokenizable)."""
        result = estimate_tokens("   ")
        assert isinstance(result, int)
        assert result >= 0
