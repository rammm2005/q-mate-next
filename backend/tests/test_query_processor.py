"""Unit tests for QueryProcessor intent classification and query expansion."""

import pytest

from app.models.query import QueryIntent
from app.services.query_processor import QueryProcessor, MAX_QUESTION_LENGTH


@pytest.fixture
def processor():
    """Create a QueryProcessor instance for tests."""
    return QueryProcessor()


class TestClassifyIntentValidation:
    """Tests for input validation in classify_intent."""

    def test_empty_string_raises_value_error(self, processor):
        with pytest.raises(ValueError, match="between 1 and 1000"):
            processor.classify_intent("")

    def test_whitespace_only_raises_value_error(self, processor):
        with pytest.raises(ValueError, match="between 1 and 1000"):
            processor.classify_intent("   \n\t  ")

    def test_exceeds_1000_chars_raises_value_error(self, processor):
        long_question = "a" * 1001
        with pytest.raises(ValueError, match="between 1 and 1000"):
            processor.classify_intent(long_question)

    def test_exactly_1000_chars_does_not_raise(self, processor):
        question = "a" * 1000
        # Should not raise, defaults to CODE_LOOKUP
        result = processor.classify_intent(question)
        assert result == QueryIntent.CODE_LOOKUP

    def test_single_char_question_valid(self, processor):
        result = processor.classify_intent("x")
        assert result == QueryIntent.CODE_LOOKUP


class TestClassifyIntentDebugging:
    """Tests for DEBUGGING intent classification."""

    def test_bug_keyword(self, processor):
        assert processor.classify_intent("There's a bug in the login") == QueryIntent.DEBUGGING

    def test_error_keyword(self, processor):
        assert processor.classify_intent("What causes this error?") == QueryIntent.DEBUGGING

    def test_fix_keyword(self, processor):
        assert processor.classify_intent("How do I fix the timeout?") == QueryIntent.DEBUGGING

    def test_crash_keyword(self, processor):
        assert processor.classify_intent("The app keeps crashing") == QueryIntent.DEBUGGING

    def test_debug_keyword(self, processor):
        assert processor.classify_intent("How to debug the parser") == QueryIntent.DEBUGGING

    def test_issue_keyword(self, processor):
        assert processor.classify_intent("There's an issue with auth") == QueryIntent.DEBUGGING


class TestClassifyIntentAPIUsage:
    """Tests for API_USAGE intent classification."""

    def test_api_keyword(self, processor):
        assert processor.classify_intent("How to call the user API") == QueryIntent.API_USAGE

    def test_endpoint_keyword(self, processor):
        assert processor.classify_intent("What endpoints are available?") == QueryIntent.API_USAGE

    def test_request_keyword(self, processor):
        assert processor.classify_intent("How to make a request to users") == QueryIntent.API_USAGE

    def test_route_keyword(self, processor):
        assert processor.classify_intent("Which route handles login?") == QueryIntent.API_USAGE

    def test_http_keyword(self, processor):
        assert processor.classify_intent("What HTTP methods does it support?") == QueryIntent.API_USAGE

    def test_rest_keyword(self, processor):
        assert processor.classify_intent("Is this a REST service?") == QueryIntent.API_USAGE


class TestClassifyIntentArchitecture:
    """Tests for ARCHITECTURE intent classification."""

    def test_architecture_keyword(self, processor):
        assert processor.classify_intent("What is the architecture?") == QueryIntent.ARCHITECTURE

    def test_structure_keyword(self, processor):
        assert processor.classify_intent("What's the project structure?") == QueryIntent.ARCHITECTURE

    def test_module_keyword(self, processor):
        assert processor.classify_intent("Which module handles auth?") == QueryIntent.ARCHITECTURE

    def test_dependency_keyword(self, processor):
        assert processor.classify_intent("What are the dependency chains?") == QueryIntent.ARCHITECTURE

    def test_design_keyword(self, processor):
        assert processor.classify_intent("What's the design of this system?") == QueryIntent.ARCHITECTURE


class TestClassifyIntentDocumentation:
    """Tests for DOCUMENTATION intent classification."""

    def test_document_keyword(self, processor):
        assert processor.classify_intent("Where is the document for auth?") == QueryIntent.DOCUMENTATION

    def test_explain_keyword(self, processor):
        assert processor.classify_intent("Can you explain this function?") == QueryIntent.DOCUMENTATION

    def test_how_does_keyword(self, processor):
        assert processor.classify_intent("How does the cache work?") == QueryIntent.DOCUMENTATION

    def test_what_is_keyword(self, processor):
        assert processor.classify_intent("What is the retry logic?") == QueryIntent.DOCUMENTATION

    def test_describe_keyword(self, processor):
        assert processor.classify_intent("Describe the auth flow") == QueryIntent.DOCUMENTATION


class TestClassifyIntentCodeLookup:
    """Tests for CODE_LOOKUP intent classification."""

    def test_find_keyword(self, processor):
        assert processor.classify_intent("Find the getUserName function") == QueryIntent.CODE_LOOKUP

    def test_where_is_keyword(self, processor):
        assert processor.classify_intent("Where is the config loaded?") == QueryIntent.CODE_LOOKUP

    def test_show_me_keyword(self, processor):
        assert processor.classify_intent("Show me the database connection code") == QueryIntent.CODE_LOOKUP

    def test_function_keyword(self, processor):
        assert processor.classify_intent("What does the function calculate?") == QueryIntent.CODE_LOOKUP

    def test_class_keyword(self, processor):
        assert processor.classify_intent("Where is the UserService class?") == QueryIntent.CODE_LOOKUP


class TestClassifyIntentDefault:
    """Tests for default CODE_LOOKUP when no pattern matches."""

    def test_no_matching_keywords_defaults_to_code_lookup(self, processor):
        assert processor.classify_intent("getUserName") == QueryIntent.CODE_LOOKUP

    def test_random_identifier_defaults_to_code_lookup(self, processor):
        assert processor.classify_intent("parse_config_yaml") == QueryIntent.CODE_LOOKUP

    def test_generic_question_defaults_to_code_lookup(self, processor):
        assert processor.classify_intent("hello world") == QueryIntent.CODE_LOOKUP


class TestClassifyIntentDeterminism:
    """Tests that classification is deterministic."""

    def test_same_input_same_output(self, processor):
        question = "How to debug the API endpoint?"
        first_result = processor.classify_intent(question)
        for _ in range(10):
            assert processor.classify_intent(question) == first_result

    def test_case_insensitive(self, processor):
        assert processor.classify_intent("Find the BUG") == QueryIntent.DEBUGGING
        assert processor.classify_intent("find the bug") == QueryIntent.DEBUGGING
        assert processor.classify_intent("FIND THE BUG") == QueryIntent.DEBUGGING


class TestClassifyIntentPriority:
    """Tests for priority ordering when multiple keywords match."""

    def test_debugging_takes_priority_over_api(self, processor):
        # Contains both "bug" and "api"
        result = processor.classify_intent("There's a bug in the API")
        assert result == QueryIntent.DEBUGGING

    def test_debugging_takes_priority_over_documentation(self, processor):
        # Contains both "error" and "explain"
        result = processor.classify_intent("Explain this error message")
        assert result == QueryIntent.DEBUGGING

    def test_api_takes_priority_over_architecture(self, processor):
        # Contains both "endpoint" and "module"
        result = processor.classify_intent("Which module defines the endpoint?")
        assert result == QueryIntent.API_USAGE

    def test_api_takes_priority_over_documentation(self, processor):
        # Contains both "api" and "explain"
        result = processor.classify_intent("Explain the API usage")
        assert result == QueryIntent.API_USAGE


class TestExpandQueryBaseTokens:
    """Tests that expand_query always includes base tokens from code-aware tokenization."""

    def test_base_tokens_included_for_code_lookup(self, processor):
        result = processor.expand_query("getUserName", QueryIntent.CODE_LOOKUP)
        # code_aware_tokenize("getUserName") produces tokens like [getusername, get, user, name]
        assert "get" in result
        assert "user" in result
        assert "name" in result

    def test_base_tokens_included_for_api_usage(self, processor):
        result = processor.expand_query("users endpoint", QueryIntent.API_USAGE)
        assert "users" in result
        assert "endpoint" in result

    def test_base_tokens_included_for_documentation(self, processor):
        result = processor.expand_query("auth flow", QueryIntent.DOCUMENTATION)
        assert "auth" in result
        assert "flow" in result

    def test_base_tokens_included_for_architecture(self, processor):
        result = processor.expand_query("service layer", QueryIntent.ARCHITECTURE)
        assert "service" in result
        assert "layer" in result

    def test_base_tokens_included_for_debugging(self, processor):
        result = processor.expand_query("parse error", QueryIntent.DEBUGGING)
        assert "parse" in result
        assert "error" in result

    def test_returns_list(self, processor):
        result = processor.expand_query("hello world", QueryIntent.CODE_LOOKUP)
        assert isinstance(result, list)
        assert len(result) > 0


class TestExpandQueryCodeLookup:
    """Tests for CODE_LOOKUP intent expansion patterns."""

    def test_adds_def_pattern(self, processor):
        result = processor.expand_query("getUserName", QueryIntent.CODE_LOOKUP)
        # Should have "def <token>" for each base token
        assert any(item.startswith("def ") for item in result)

    def test_adds_function_pattern(self, processor):
        result = processor.expand_query("getUserName", QueryIntent.CODE_LOOKUP)
        assert any(item.startswith("function ") for item in result)

    def test_adds_func_pattern(self, processor):
        result = processor.expand_query("getUserName", QueryIntent.CODE_LOOKUP)
        assert any(item.startswith("func ") for item in result)

    def test_adds_class_pattern(self, processor):
        result = processor.expand_query("getUserName", QueryIntent.CODE_LOOKUP)
        assert any(item.startswith("class ") for item in result)

    def test_adds_call_pattern(self, processor):
        result = processor.expand_query("getUserName", QueryIntent.CODE_LOOKUP)
        assert any(item.endswith("()") for item in result)

    def test_expands_each_base_token(self, processor):
        result = processor.expand_query("parse config", QueryIntent.CODE_LOOKUP)
        # Both "parse" and "config" should get function signature patterns
        assert "def parse" in result
        assert "def config" in result
        assert "function parse" in result
        assert "function config" in result
        assert "func parse" in result
        assert "func config" in result
        assert "class parse" in result
        assert "class config" in result
        assert "parse()" in result
        assert "config()" in result


class TestExpandQueryAPIUsage:
    """Tests for API_USAGE intent expansion patterns."""

    def test_adds_app_get_pattern(self, processor):
        result = processor.expand_query("users", QueryIntent.API_USAGE)
        assert any('@app.get("/' in item for item in result)

    def test_adds_app_post_pattern(self, processor):
        result = processor.expand_query("users", QueryIntent.API_USAGE)
        assert any('@app.post("/' in item for item in result)

    def test_adds_router_pattern(self, processor):
        result = processor.expand_query("users", QueryIntent.API_USAGE)
        assert any(item.startswith("router.") for item in result)

    def test_adds_endpoint_pattern(self, processor):
        result = processor.expand_query("users", QueryIntent.API_USAGE)
        assert any(item.startswith("endpoint ") for item in result)

    def test_expands_each_base_token(self, processor):
        result = processor.expand_query("user login", QueryIntent.API_USAGE)
        assert '@app.get("/user")' in result
        assert '@app.get("/login")' in result
        assert '@app.post("/user")' in result
        assert '@app.post("/login")' in result
        assert "router.user" in result
        assert "router.login" in result
        assert "endpoint user" in result
        assert "endpoint login" in result


class TestExpandQueryDocumentation:
    """Tests for DOCUMENTATION intent expansion patterns."""

    def test_adds_readme(self, processor):
        result = processor.expand_query("auth flow", QueryIntent.DOCUMENTATION)
        assert "readme" in result

    def test_adds_docs(self, processor):
        result = processor.expand_query("auth flow", QueryIntent.DOCUMENTATION)
        assert "docs" in result

    def test_adds_guide(self, processor):
        result = processor.expand_query("auth flow", QueryIntent.DOCUMENTATION)
        assert "guide" in result

    def test_adds_tutorial(self, processor):
        result = processor.expand_query("auth flow", QueryIntent.DOCUMENTATION)
        assert "tutorial" in result

    def test_adds_example(self, processor):
        result = processor.expand_query("auth flow", QueryIntent.DOCUMENTATION)
        assert "example" in result

    def test_does_not_add_function_patterns(self, processor):
        result = processor.expand_query("auth flow", QueryIntent.DOCUMENTATION)
        assert not any(item.startswith("def ") for item in result)
        assert not any(item.startswith("function ") for item in result)


class TestExpandQueryArchitecture:
    """Tests for ARCHITECTURE intent expansion patterns."""

    def test_adds_import(self, processor):
        result = processor.expand_query("service layer", QueryIntent.ARCHITECTURE)
        assert "import" in result

    def test_adds_module(self, processor):
        result = processor.expand_query("service layer", QueryIntent.ARCHITECTURE)
        assert "module" in result

    def test_adds_package(self, processor):
        result = processor.expand_query("service layer", QueryIntent.ARCHITECTURE)
        assert "package" in result

    def test_adds_dependency(self, processor):
        result = processor.expand_query("service layer", QueryIntent.ARCHITECTURE)
        assert "dependency" in result

    def test_adds_config(self, processor):
        result = processor.expand_query("service layer", QueryIntent.ARCHITECTURE)
        assert "config" in result

    def test_does_not_add_http_patterns(self, processor):
        result = processor.expand_query("service layer", QueryIntent.ARCHITECTURE)
        assert not any('@app.get("/' in item for item in result)
        assert not any('@app.post("/' in item for item in result)


class TestExpandQueryDebugging:
    """Tests for DEBUGGING intent - should only include base tokens (no extra patterns)."""

    def test_only_base_tokens(self, processor):
        from app.services.tokenizer import code_aware_tokenize
        question = "null pointer"
        base_tokens = code_aware_tokenize(question)
        result = processor.expand_query(question, QueryIntent.DEBUGGING)
        assert result == base_tokens

    def test_no_function_patterns(self, processor):
        result = processor.expand_query("null pointer", QueryIntent.DEBUGGING)
        assert not any(item.startswith("def ") for item in result)
        assert not any(item.startswith("function ") for item in result)

    def test_no_api_patterns(self, processor):
        result = processor.expand_query("null pointer", QueryIntent.DEBUGGING)
        assert not any('@app.get("/' in item for item in result)
        assert not any("router." in item for item in result)
