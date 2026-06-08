"""Code-aware tokenization for BM25 lexical search.

This module provides tokenization that understands programming naming conventions
such as camelCase, snake_case, and dot notation. It splits identifiers into
constituent words while preserving the full original identifier as an additional
token for exact matching.

Reference: Arwan et al., SIET 2023 - "Tokenization in source code is slightly
different from the text in general because in the code we often find techniques
for writing Class, Method, and Variable names in the form of camel case, snake
case (e.g. AddPatientAction, Patient_Name). Each word on the form should be
separated."
"""

import re


def code_aware_tokenize(text: str, language: str | None = None) -> list[str]:
    """Tokenize source code text with awareness of programming conventions.

    Handles:
    - camelCase splitting: getUserName → [getusername, get, user, name]
    - snake_case splitting: get_user_name → [get_user_name, get, user, name]
    - dot notation: module.class.method → [module.class.method, module, class, method]
    - Preserves full original identifiers as additional tokens

    Args:
        text: The source code text to tokenize.
        language: Optional programming language hint (reserved for future use).

    Returns:
        A list of unique lowercase tokens with length > 1, preserving insertion order.
        Returns an empty list for empty or whitespace-only input.
    """
    if not text or not text.strip():
        return []

    tokens: list[str] = []

    # Step 1: Split on whitespace and punctuation (preserve identifiers)
    raw_tokens = re.split(r'[\s\(\)\{\}\[\];,=<>!&|+\-*/]+', text)

    for raw in raw_tokens:
        if not raw:
            continue

        # Preserve the full token
        tokens.append(raw.lower())

        # Step 2: Split camelCase (e.g., getUserName → get_User_Name → [get, User, Name])
        camel_parts = re.sub(r'([a-z])([A-Z])', r'\1_\2', raw).split('_')

        if len(camel_parts) > 1:
            tokens.extend(part.lower() for part in camel_parts if part)

        # Step 3: Split on dots (module.class.method)
        if '.' in raw:
            dot_parts = raw.split('.')
            tokens.extend(part.lower() for part in dot_parts if part)

        # Step 4: Split snake_case (if not already split by camelCase)
        if '_' in raw and len(camel_parts) <= 1:
            snake_parts = raw.split('_')
            tokens.extend(part.lower() for part in snake_parts if part)

    # Step 5: Remove duplicates while preserving order, skip single-char tokens
    seen: set[str] = set()
    unique_tokens: list[str] = []
    for t in tokens:
        if t not in seen and len(t) > 1:
            seen.add(t)
            unique_tokens.append(t)

    return unique_tokens
