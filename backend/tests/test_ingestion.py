"""Unit tests for the repository ingestion pipeline."""

import os
import tempfile
import shutil

import pytest

from app.services.ingestion import (
    IngestionConfig,
    IngestionPipeline,
    ParsedFile,
    SUPPORTED_EXTENSIONS,
    SECRET_FILENAMES,
    DEFAULT_EXCLUSIONS,
)


@pytest.fixture
def pipeline():
    """Create an IngestionPipeline instance."""
    return IngestionPipeline()


@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary repository structure for testing."""
    # Create supported language files
    py_file = tmp_path / "src" / "main.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("def hello():\n    return 'world'\n")

    ts_file = tmp_path / "src" / "app.ts"
    ts_file.write_text("export function greet(): string { return 'hi'; }\n")

    js_file = tmp_path / "lib" / "utils.js"
    js_file.parent.mkdir(parents=True)
    js_file.write_text("function add(a, b) { return a + b; }\n")

    go_file = tmp_path / "cmd" / "main.go"
    go_file.parent.mkdir(parents=True)
    go_file.write_text('package main\n\nfunc main() {\n    fmt.Println("hello")\n}\n')

    php_file = tmp_path / "web" / "index.php"
    php_file.parent.mkdir(parents=True)
    php_file.write_text("<?php\nfunction render() { echo 'hello'; }\n")

    # Create unsupported language file
    txt_file = tmp_path / "docs" / "readme.md"
    txt_file.parent.mkdir(parents=True)
    txt_file.write_text("# Project Documentation\n\nThis is a sample project.\n")

    # Create a binary file
    bin_file = tmp_path / "assets" / "image.png"
    bin_file.parent.mkdir(parents=True)
    bin_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    # Create a node_modules directory (should be excluded)
    nm_file = tmp_path / "node_modules" / "pkg" / "index.js"
    nm_file.parent.mkdir(parents=True)
    nm_file.write_text("module.exports = {};\n")

    # Create __pycache__ (should be excluded)
    cache_file = tmp_path / "__pycache__" / "main.cpython-311.pyc"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"\x00\x00\x00\x00")

    # Create .git directory (should be excluded)
    git_file = tmp_path / ".git" / "config"
    git_file.parent.mkdir(parents=True)
    git_file.write_text("[core]\n    bare = false\n")

    return tmp_path


@pytest.fixture
def temp_repo_with_secrets(tmp_path):
    """Create a repository with files containing secrets."""
    # File with API key
    api_key_file = tmp_path / "config.py"
    api_key_file.write_text('API_KEY = "sk_live_abcdefghijklmnop1234567890"\n')

    # File with private key
    key_file = tmp_path / "server.key"
    key_file.write_text(
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAK...\n-----END RSA PRIVATE KEY-----\n"
    )

    # File with AWS credentials
    aws_file = tmp_path / "aws_config.py"
    aws_file.write_text("aws_access_key_id = AKIAIOSFODNN7EXAMPLE\n")

    # .env file (secret filename)
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=postgres://user:pass@localhost/db\n")

    # Safe file (no secrets)
    safe_file = tmp_path / "app.py"
    safe_file.write_text("def main():\n    print('hello world')\n")

    return tmp_path


class TestDetectLanguage:
    """Tests for language detection from file extension."""

    def test_detect_python(self, pipeline):
        assert pipeline._detect_language("src/main.py") == "python"

    def test_detect_typescript(self, pipeline):
        assert pipeline._detect_language("src/app.ts") == "typescript"

    def test_detect_tsx(self, pipeline):
        assert pipeline._detect_language("components/Button.tsx") == "typescript"

    def test_detect_javascript(self, pipeline):
        assert pipeline._detect_language("lib/utils.js") == "javascript"

    def test_detect_jsx(self, pipeline):
        assert pipeline._detect_language("components/App.jsx") == "javascript"

    def test_detect_php(self, pipeline):
        assert pipeline._detect_language("web/index.php") == "php"

    def test_detect_go(self, pipeline):
        assert pipeline._detect_language("cmd/main.go") == "go"

    def test_unsupported_extension(self, pipeline):
        assert pipeline._detect_language("readme.md") is None

    def test_no_extension(self, pipeline):
        assert pipeline._detect_language("Makefile") is None

    def test_case_insensitive_extension(self, pipeline):
        assert pipeline._detect_language("Module.PY") == "python"


class TestIsBinary:
    """Tests for binary file detection."""

    def test_text_file_is_not_binary(self, pipeline, tmp_path):
        f = tmp_path / "text.py"
        f.write_text("def hello(): pass\n")
        assert pipeline._is_binary(str(f)) is False

    def test_binary_file_with_null_bytes(self, pipeline, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"header\x00\x01\x02\x03data")
        assert pipeline._is_binary(str(f)) is True

    def test_empty_file_is_not_binary(self, pipeline, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        assert pipeline._is_binary(str(f)) is False

    def test_nonexistent_file_treated_as_binary(self, pipeline):
        assert pipeline._is_binary("/nonexistent/path/file.txt") is True


class TestMatchesExclusion:
    """Tests for exclusion pattern matching."""

    def test_matches_node_modules(self, pipeline):
        assert pipeline._matches_exclusion(
            "node_modules/pkg/index.js", DEFAULT_EXCLUSIONS
        ) is True

    def test_matches_pycache(self, pipeline):
        assert pipeline._matches_exclusion(
            "__pycache__/main.cpython-311.pyc", DEFAULT_EXCLUSIONS
        ) is True

    def test_matches_git_directory(self, pipeline):
        assert pipeline._matches_exclusion(
            ".git/config", DEFAULT_EXCLUSIONS
        ) is True

    def test_matches_pyc_extension(self, pipeline):
        assert pipeline._matches_exclusion(
            "src/module.pyc", DEFAULT_EXCLUSIONS
        ) is True

    def test_matches_min_js(self, pipeline):
        assert pipeline._matches_exclusion(
            "dist/bundle.min.js", DEFAULT_EXCLUSIONS
        ) is True

    def test_matches_dist_directory(self, pipeline):
        assert pipeline._matches_exclusion(
            "dist/app.js", DEFAULT_EXCLUSIONS
        ) is True

    def test_does_not_match_source_file(self, pipeline):
        assert pipeline._matches_exclusion(
            "src/main.py", DEFAULT_EXCLUSIONS
        ) is False

    def test_does_not_match_normal_js(self, pipeline):
        assert pipeline._matches_exclusion(
            "src/app.js", DEFAULT_EXCLUSIONS
        ) is False

    def test_custom_pattern(self, pipeline):
        patterns = ["vendor/**", "*.generated.ts"]
        assert pipeline._matches_exclusion("vendor/lib/foo.js", patterns) is True
        assert pipeline._matches_exclusion("src/types.generated.ts", patterns) is True
        assert pipeline._matches_exclusion("src/types.ts", patterns) is False


class TestContainsSecrets:
    """Tests for secret detection in file content."""

    def test_detects_api_key(self, pipeline, tmp_path):
        f = tmp_path / "config.py"
        f.write_text('api_key = "sk_live_abcdefghijklmnop1234567890"\n')
        assert pipeline._contains_secrets(str(f)) is True

    def test_detects_private_key(self, pipeline, tmp_path):
        f = tmp_path / "key.pem"
        f.write_text("-----BEGIN PRIVATE KEY-----\ndata\n-----END PRIVATE KEY-----\n")
        assert pipeline._contains_secrets(str(f)) is True

    def test_detects_rsa_private_key(self, pipeline, tmp_path):
        f = tmp_path / "rsa.key"
        f.write_text("-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----\n")
        assert pipeline._contains_secrets(str(f)) is True

    def test_detects_aws_credentials(self, pipeline, tmp_path):
        f = tmp_path / "aws.cfg"
        f.write_text("aws_access_key_id = AKIAIOSFODNN7EXAMPLE\n")
        assert pipeline._contains_secrets(str(f)) is True

    def test_detects_password_assignment(self, pipeline, tmp_path):
        f = tmp_path / "db.py"
        f.write_text('password = "SuperSecret1234567890abcdef"\n')
        assert pipeline._contains_secrets(str(f)) is True

    def test_safe_file_no_secrets(self, pipeline, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("def main():\n    print('hello world')\n")
        assert pipeline._contains_secrets(str(f)) is False

    def test_nonexistent_file_no_secrets(self, pipeline):
        assert pipeline._contains_secrets("/nonexistent/file.py") is False

    def test_short_values_not_flagged(self, pipeline, tmp_path):
        """Values shorter than 16 chars should not trigger API key pattern."""
        f = tmp_path / "config.py"
        f.write_text('api_key = "short"\n')
        assert pipeline._contains_secrets(str(f)) is False


class TestWalkRepository:
    """Tests for the full repository walking functionality."""

    def test_finds_supported_language_files(self, pipeline, temp_repo):
        results = pipeline.walk_repository(str(temp_repo))
        supported_files = [f for f in results if f.is_supported_language]

        # Should find py, ts, js, go, php files
        languages_found = {f.language for f in supported_files}
        assert "python" in languages_found
        assert "typescript" in languages_found
        assert "javascript" in languages_found
        assert "go" in languages_found
        assert "php" in languages_found

    def test_finds_unsupported_files_as_plaintext(self, pipeline, temp_repo):
        results = pipeline.walk_repository(str(temp_repo))
        unsupported_files = [f for f in results if not f.is_supported_language]

        # Should find the markdown file
        assert len(unsupported_files) >= 1
        md_files = [f for f in unsupported_files if f.file_path.endswith(".md")]
        assert len(md_files) == 1
        assert md_files[0].language is None

    def test_skips_binary_files(self, pipeline, temp_repo):
        results = pipeline.walk_repository(str(temp_repo))
        file_paths = [f.file_path for f in results]

        # Binary PNG file should not be included
        assert not any("image.png" in p for p in file_paths)

    def test_skips_excluded_directories(self, pipeline, temp_repo):
        results = pipeline.walk_repository(str(temp_repo))
        file_paths = [f.file_path for f in results]

        # node_modules, __pycache__, .git should all be excluded
        assert not any("node_modules" in p for p in file_paths)
        assert not any("__pycache__" in p for p in file_paths)
        assert not any(".git" in p for p in file_paths)

    def test_skips_secret_files(self, pipeline, temp_repo_with_secrets):
        results = pipeline.walk_repository(str(temp_repo_with_secrets))
        file_paths = [f.file_path for f in results]

        # Only safe file should remain
        assert len(results) == 1
        assert results[0].file_path == "app.py"

    def test_skips_env_files_by_name(self, pipeline, temp_repo_with_secrets):
        results = pipeline.walk_repository(str(temp_repo_with_secrets))
        file_paths = [f.file_path for f in results]

        assert not any(".env" in p for p in file_paths)

    def test_relative_paths_use_forward_slashes(self, pipeline, temp_repo):
        results = pipeline.walk_repository(str(temp_repo))

        for f in results:
            assert "\\" not in f.file_path

    def test_file_content_is_populated(self, pipeline, temp_repo):
        results = pipeline.walk_repository(str(temp_repo))

        for f in results:
            assert f.content.strip() != ""

    def test_nonexistent_path_raises_error(self, pipeline):
        with pytest.raises(FileNotFoundError):
            pipeline.walk_repository("/nonexistent/repo/path")

    def test_file_path_raises_error(self, pipeline, tmp_path):
        f = tmp_path / "file.py"
        f.write_text("content")
        with pytest.raises(NotADirectoryError):
            pipeline.walk_repository(str(f))

    def test_custom_exclusion_patterns(self, pipeline, tmp_path):
        # Create files
        src = tmp_path / "src" / "app.py"
        src.parent.mkdir(parents=True)
        src.write_text("print('hello')\n")

        vendor = tmp_path / "vendor" / "lib.py"
        vendor.parent.mkdir(parents=True)
        vendor.write_text("print('vendor')\n")

        config = IngestionConfig(exclude_patterns=["vendor/**"])
        results = pipeline.walk_repository(str(tmp_path), config)
        file_paths = [f.file_path for f in results]

        assert any("src/app.py" in p for p in file_paths)
        assert not any("vendor" in p for p in file_paths)

    def test_empty_repository(self, pipeline, tmp_path):
        results = pipeline.walk_repository(str(tmp_path))
        assert results == []

    def test_skips_empty_files(self, pipeline, tmp_path):
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        whitespace_file = tmp_path / "whitespace.py"
        whitespace_file.write_text("   \n\n  ")

        results = pipeline.walk_repository(str(tmp_path))
        assert results == []


class TestIngestionConfig:
    """Tests for IngestionConfig defaults."""

    def test_default_languages(self):
        config = IngestionConfig()
        # Should include all supported languages
        assert "python" in config.languages
        assert "typescript" in config.languages
        assert "javascript" in config.languages
        assert "php" in config.languages
        assert "go" in config.languages

    def test_default_exclusions(self):
        config = IngestionConfig()
        assert "node_modules/**" in config.exclude_patterns
        assert "__pycache__/**" in config.exclude_patterns
        assert ".git/**" in config.exclude_patterns

    def test_default_chunk_settings(self):
        config = IngestionConfig()
        assert config.max_chunk_tokens == 512
        assert config.overlap_tokens == 50

    def test_custom_config(self):
        config = IngestionConfig(
            languages=["python", "go"],
            exclude_patterns=["vendor/**"],
            max_chunk_tokens=1024,
            overlap_tokens=100,
        )
        assert config.languages == ["python", "go"]
        assert config.exclude_patterns == ["vendor/**"]
        assert config.max_chunk_tokens == 1024
        assert config.overlap_tokens == 100
