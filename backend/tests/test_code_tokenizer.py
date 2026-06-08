"""Comprehensive unit tests for code-aware tokenization.

Tests cover:
- camelCase splitting (Requirement 4.1)
- snake_case splitting (Requirement 4.2)
- dot notation splitting (Requirement 4.3)
- Full identifier preservation (Requirement 4.4)
- Lowercase and deduplication (Requirement 4.5)
- Single-character token exclusion (Requirement 4.6)
- Tokenization determinism (Requirement 4.7)
"""

import pytest

from app.services.tokenizer import code_aware_tokenize


class TestCamelCaseSplitting:
    """Tests for camelCase identifier splitting (Requirement 4.1)."""

    def test_simple_camel_case(self):
        """getUserName should split into [get, user, name]."""
        result = code_aware_tokenize("getUserName")
        assert "get" in result
        assert "user" in result
        assert "name" in result

    def test_camel_case_preserves_original(self):
        """The full camelCase identifier should also be present."""
        result = code_aware_tokenize("getUserName")
        assert "getusername" in result

    def test_two_word_camel_case(self):
        """isValid should split into [is, valid]."""
        result = code_aware_tokenize("isValid")
        assert "isvalid" in result
        assert "is" in result
        assert "valid" in result

    def test_multiple_camel_case_words(self):
        """parseJsonResponseBody should split into its constituent words."""
        result = code_aware_tokenize("parseJsonResponseBody")
        assert "parse" in result
        assert "json" in result
        assert "response" in result
        assert "body" in result

    def test_camel_case_in_expression(self):
        """camelCase within a larger expression."""
        result = code_aware_tokenize("myVar = getUserName()")
        assert "getusername" in result
        assert "get" in result
        assert "user" in result
        assert "name" in result
        assert "myvar" in result


class TestSnakeCaseSplitting:
    """Tests for snake_case identifier splitting (Requirement 4.2)."""

    def test_simple_snake_case(self):
        """get_user_name should split into [get, user, name]."""
        result = code_aware_tokenize("get_user_name")
        assert "get" in result
        assert "user" in result
        assert "name" in result

    def test_snake_case_preserves_original(self):
        """The full snake_case identifier should also be present."""
        result = code_aware_tokenize("get_user_name")
        assert "get_user_name" in result

    def test_two_word_snake_case(self):
        """is_valid should split into [is, valid]."""
        result = code_aware_tokenize("is_valid")
        assert "is_valid" in result
        assert "is" in result
        assert "valid" in result

    def test_multiple_snake_case_words(self):
        """parse_json_response_body should split correctly."""
        result = code_aware_tokenize("parse_json_response_body")
        assert "parse" in result
        assert "json" in result
        assert "response" in result
        assert "body" in result


class TestDotNotationSplitting:
    """Tests for dot notation splitting (Requirement 4.3)."""

    def test_simple_dot_notation(self):
        """module.class.method should split into [module, class, method]."""
        result = code_aware_tokenize("module.class.method")
        assert "module" in result
        assert "class" in result
        assert "method" in result

    def test_dot_notation_preserves_original(self):
        """The full dot notation string should be preserved."""
        result = code_aware_tokenize("module.class.method")
        assert "module.class.method" in result

    def test_two_part_dot_notation(self):
        """os.path should split into [os, path]."""
        result = code_aware_tokenize("os.path")
        assert "os.path" in result
        assert "os" in result
        assert "path" in result

    def test_long_dot_notation(self):
        """com.example.project.utils.Helper should split correctly."""
        result = code_aware_tokenize("com.example.project.utils.Helper")
        assert "com" in result
        assert "example" in result
        assert "project" in result
        assert "utils" in result
        assert "helper" in result


class TestFullIdentifierPreservation:
    """Tests for preserving full original identifiers (Requirement 4.4)."""

    def test_camel_case_original_preserved(self):
        """Full camelCase identifier should be in results."""
        result = code_aware_tokenize("getUserName")
        assert "getusername" in result

    def test_snake_case_original_preserved(self):
        """Full snake_case identifier should be in results."""
        result = code_aware_tokenize("get_user_name")
        assert "get_user_name" in result

    def test_dot_notation_original_preserved(self):
        """Full dot notation identifier should be in results."""
        result = code_aware_tokenize("os.path.join")
        assert "os.path.join" in result

    def test_plain_word_preserved(self):
        """A plain word with no special casing should be preserved."""
        result = code_aware_tokenize("hello")
        assert "hello" in result

    def test_multiple_identifiers_all_preserved(self):
        """Multiple identifiers in text should all be preserved."""
        result = code_aware_tokenize("getUserName get_user_name")
        assert "getusername" in result
        assert "get_user_name" in result


class TestLowercaseAndDeduplication:
    """Tests for lowercase conversion and deduplication (Requirement 4.5)."""

    def test_all_tokens_lowercase(self):
        """All tokens should be lowercase."""
        result = code_aware_tokenize("GetUserName HTTP_REQUEST Module.Class")
        for token in result:
            assert token == token.lower(), f"Token '{token}' is not lowercase"

    def test_no_duplicates(self):
        """There should be no duplicate tokens."""
        result = code_aware_tokenize("getUserName getUserName")
        assert len(result) == len(set(result))

    def test_order_preserved(self):
        """Insertion order should be preserved after deduplication."""
        result = code_aware_tokenize("getUserName setUserName")
        # 'get' appears first (from getUserName), then 'set' (from setUserName)
        # 'user' and 'name' appear only once since they're deduplicated
        get_idx = result.index("get")
        set_idx = result.index("set")
        assert get_idx < set_idx

    def test_mixed_case_deduplicates(self):
        """Tokens that differ only in case should be deduplicated."""
        result = code_aware_tokenize("User user")
        # Both 'User' and 'user' lowercase to 'user', so only one should exist
        assert result.count("user") == 1


class TestSingleCharExclusion:
    """Tests for single-character token exclusion (Requirement 4.6)."""

    def test_single_char_variables_excluded(self):
        """Single character tokens like 'x', 'y' should be excluded."""
        result = code_aware_tokenize("x = y + z")
        assert "x" not in result
        assert "y" not in result
        assert "z" not in result

    def test_single_char_from_splitting_excluded(self):
        """Single chars resulting from identifier splitting should be excluded."""
        # If we had a camelCase like 'aValue', 'a' should be excluded
        result = code_aware_tokenize("aValue")
        assert "a" not in result
        assert "value" in result

    def test_two_char_tokens_included(self):
        """Two character tokens should be included."""
        result = code_aware_tokenize("is_ok")
        assert "is" in result
        assert "ok" in result

    def test_empty_string_from_punctuation_excluded(self):
        """Empty strings from splitting should not appear."""
        result = code_aware_tokenize("  hello  world  ")
        assert "" not in result


class TestTokenizationDeterminism:
    """Tests for tokenization determinism (Requirement 4.7)."""

    def test_same_input_same_output(self):
        """Same input should always produce same output."""
        text = "getUserName get_user_name module.class.method"
        result1 = code_aware_tokenize(text)
        result2 = code_aware_tokenize(text)
        assert result1 == result2

    def test_determinism_over_many_calls(self):
        """Running tokenization many times should always give same result."""
        text = "parseJsonResponseBody some_snake_case os.path.join"
        results = [code_aware_tokenize(text) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_language_param_does_not_affect_output(self):
        """Language parameter should not change tokenization (reserved for future)."""
        text = "getUserName"
        result_none = code_aware_tokenize(text, language=None)
        result_python = code_aware_tokenize(text, language="python")
        result_ts = code_aware_tokenize(text, language="typescript")
        assert result_none == result_python == result_ts


class TestEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert code_aware_tokenize("") == []

    def test_whitespace_only(self):
        """Whitespace-only string should return empty list."""
        assert code_aware_tokenize("   ") == []
        assert code_aware_tokenize("\t\n") == []

    def test_only_punctuation(self):
        """Only punctuation should return empty list."""
        assert code_aware_tokenize("(){}[];,") == []

    def test_only_single_chars(self):
        """Input with only single-char tokens should return empty list."""
        result = code_aware_tokenize("a b c d")
        assert result == []

    def test_numbers_preserved(self):
        """Numeric tokens should be preserved if length > 1."""
        result = code_aware_tokenize("var123 count42")
        assert "var123" in result
        assert "count42" in result

    def test_all_uppercase_identifier(self):
        """ALL_CAPS constants should be handled correctly."""
        result = code_aware_tokenize("MAX_SIZE")
        assert "max_size" in result
        assert "max" in result
        assert "size" in result

    def test_mixed_conventions(self):
        """Text mixing camelCase, snake_case, and dot notation."""
        result = code_aware_tokenize("getUserName get_user_name os.path.join")
        assert "getusername" in result
        assert "get_user_name" in result
        assert "os.path.join" in result
        assert "get" in result
        assert "user" in result
        assert "name" in result
        assert "os" in result
        assert "path" in result
        assert "join" in result

    def test_code_snippet(self):
        """A realistic code snippet should tokenize correctly."""
        code = "def calculate_total(itemList, taxRate):"
        result = code_aware_tokenize(code)
        # 'def' is 3 chars (> 1), so it should be included
        assert "def" in result
        assert "calculate_total" in result
        assert "calculate" in result
        assert "total" in result
        assert "itemlist" in result
        assert "item" in result
        assert "list" in result
        assert "taxrate" in result
        assert "tax" in result
        assert "rate" in result

    def test_consecutive_uppercase(self):
        """Consecutive uppercase like 'XMLParser' - the regex only splits at lower→upper."""
        # re.sub(r'([a-z])([A-Z])') only splits at lower→upper transitions.
        # 'XMLParser' has no such transition (it's all uppercase then lowercase),
        # so no split occurs. This is correct per the algorithm design.
        result = code_aware_tokenize("XMLParser")
        assert "xmlparser" in result

    def test_camel_case_with_acronym_followed_by_word(self):
        """When there's a lower→upper transition like 'httpClient', it splits."""
        result = code_aware_tokenize("httpClient")
        assert "httpclient" in result
        assert "http" in result
        assert "client" in result

    def test_return_type_is_list(self):
        """Return type should always be a list."""
        assert isinstance(code_aware_tokenize("hello"), list)
        assert isinstance(code_aware_tokenize(""), list)

    def test_no_empty_tokens(self):
        """No empty string tokens should ever appear in results."""
        test_inputs = [
            "getUserName",
            "get_user_name",
            "module.class.method",
            "  spaced  out  ",
            "a = b + c",
            "func()",
        ]
        for text in test_inputs:
            result = code_aware_tokenize(text)
            assert "" not in result, f"Empty token found for input: {text}"


class TestRealWorldExamples:
    """Tests with realistic code patterns."""

    def test_python_function_definition(self):
        """Python function signature tokenization."""
        code = "def process_user_request(userId, requestType):"
        result = code_aware_tokenize(code)
        assert "process" in result
        assert "user" in result
        assert "request" in result
        assert "userid" in result
        assert "requesttype" in result

    def test_typescript_class_method(self):
        """TypeScript class method tokenization."""
        code = "getUserById(userId: string): Promise<User>"
        result = code_aware_tokenize(code)
        assert "getuserbyid" in result
        assert "get" in result
        assert "user" in result

    def test_import_statement(self):
        """Import statement tokenization."""
        code = "from app.services.tokenizer import code_aware_tokenize"
        result = code_aware_tokenize(code)
        assert "app.services.tokenizer" in result
        assert "app" in result
        assert "services" in result
        assert "tokenizer" in result
        assert "code_aware_tokenize" in result
        assert "code" in result
        assert "aware" in result
        assert "tokenize" in result

    def test_go_package_path(self):
        """Go package path tokenization."""
        code = "github.com.user.project.internal.handler"
        result = code_aware_tokenize(code)
        assert "github" in result
        assert "com" in result
        assert "user" in result
        assert "project" in result
        assert "internal" in result
        assert "handler" in result
