"""Token estimation utility for CodeQ-Mate.

Uses tiktoken with cl100k_base encoding (GPT-4) to estimate token counts
for text content. This is used during chunk selection to stay within
context window budgets.
"""

import tiktoken

# Cache the encoding instance at module level for efficiency,
# since this function is called frequently during chunk selection.
_encoding = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses the cl100k_base encoding (GPT-4) for accurate token counting.

    Args:
        text: The input text to estimate tokens for.

    Returns:
        A non-negative integer representing the estimated token count.
        Returns 0 for empty strings.
    """
    if not text:
        return 0
    return len(_encoding.encode(text))
