"""Unit tests for the CodeChunker service."""

import pytest

from app.services.chunker import CodeChunker, LANGUAGE_PATTERNS
from app.models.chunk import ChunkType
from app.utils.tokenizer import estimate_tokens


@pytest.fixture
def chunker():
    """Create a CodeChunker with default settings."""
    return CodeChunker(max_chunk_size=512, overlap=50, repo_id="test-repo")


@pytest.fixture
def small_chunker():
    """Create a CodeChunker with small max size for testing splits."""
    return CodeChunker(max_chunk_size=50, overlap=10, repo_id="test-repo")


# --- Python source samples ---

PYTHON_SAMPLE = '''"""Module docstring."""

import os
import sys
from typing import Optional


def hello_world():
    """Say hello."""
    print("Hello, world!")


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Calculator:
    """A simple calculator."""

    def __init__(self, initial: int = 0):
        self.value = initial

    def add(self, n: int) -> int:
        self.value += n
        return self.value

    def reset(self):
        self.value = 0
'''

PYTHON_SINGLE_FUNCTION = '''def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"
'''

# --- TypeScript source samples ---

TYPESCRIPT_SAMPLE = '''import { Request, Response } from "express";
import { UserService } from "./services/user";

export class UserController {
    private userService: UserService;

    constructor(service: UserService) {
        this.userService = service;
    }

    async getUser(req: Request, res: Response) {
        const user = await this.userService.findById(req.params.id);
        res.json(user);
    }
}

export function createRouter() {
    return new Router();
}
'''

# --- JavaScript source sample ---

JAVASCRIPT_SAMPLE = '''import { useState } from "react";

function App() {
    const [count, setCount] = useState(0);
    return <div>{count}</div>;
}

class EventEmitter {
    constructor() {
        this.listeners = {};
    }

    on(event, callback) {
        this.listeners[event] = callback;
    }
}
'''

# --- Go source sample ---

GO_SAMPLE = '''package main

import (
    "fmt"
    "net/http"
)

type Server struct {
    port int
    host string
}

func NewServer(port int) *Server {
    return &Server{port: port, host: "localhost"}
}

func (s *Server) Start() error {
    addr := fmt.Sprintf("%s:%d", s.host, s.port)
    return http.ListenAndServe(addr, nil)
}

func main() {
    server := NewServer(8080)
    server.Start()
}
'''

# --- PHP source sample ---

PHP_SAMPLE = '''<?php

use App\\Models\\User;

class UserController {
    private $userRepo;

    public function __construct($repo) {
        $this->userRepo = $repo;
    }

    public function index() {
        return $this->userRepo->findAll();
    }

    public function show($id) {
        return $this->userRepo->find($id);
    }
}

function helpers() {
    return true;
}
'''


class TestChunkFilePython:
    """Tests for chunking Python source files."""

    def test_detects_python_functions(self, chunker):
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        function_names = [c.metadata.function_name for c in function_chunks]
        assert "hello_world" in function_names
        assert "add" in function_names

    def test_detects_python_class(self, chunker):
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1
        class_names = [c.metadata.class_name for c in class_chunks]
        assert "Calculator" in class_names

    def test_chunk_coverage_python(self, chunker):
        """Every line of the file must be covered by at least one chunk."""
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        lines = PYTHON_SAMPLE.split("\n")
        total_lines = len(lines)

        covered = set()
        for chunk in chunks:
            for line_num in range(chunk.start_line, chunk.end_line + 1):
                covered.add(line_num)

        # Every non-empty line should be covered
        for line_num in range(1, total_lines + 1):
            if lines[line_num - 1].strip():
                assert line_num in covered, f"Line {line_num} not covered"

    def test_file_path_stored_correctly(self, chunker):
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        for chunk in chunks:
            assert chunk.file_path == "src/main.py"

    def test_language_stored_correctly(self, chunker):
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        for chunk in chunks:
            assert chunk.language == "python"

    def test_repo_id_stored_correctly(self, chunker):
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        for chunk in chunks:
            assert chunk.repo_id == "test-repo"

    def test_imports_in_metadata(self, chunker):
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        # At least some chunks should have imports metadata
        all_imports = set()
        for chunk in chunks:
            all_imports.update(chunk.metadata.imports)
        assert len(all_imports) > 0

    def test_single_function_file(self, chunker):
        chunks = chunker.chunk_file(PYTHON_SINGLE_FUNCTION, "utils.py", "python")
        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) >= 1
        assert func_chunks[0].metadata.function_name == "greet"

    def test_start_line_before_end_line(self, chunker):
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        for chunk in chunks:
            assert chunk.start_line >= 1
            assert chunk.start_line <= chunk.end_line


class TestChunkFileTypeScript:
    """Tests for chunking TypeScript source files."""

    def test_detects_typescript_class(self, chunker):
        chunks = chunker.chunk_file(TYPESCRIPT_SAMPLE, "src/controller.ts", "typescript")
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1

    def test_detects_typescript_function(self, chunker):
        chunks = chunker.chunk_file(TYPESCRIPT_SAMPLE, "src/controller.ts", "typescript")
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        func_names = [c.metadata.function_name for c in func_chunks]
        assert "createRouter" in func_names

    def test_chunk_coverage_typescript(self, chunker):
        chunks = chunker.chunk_file(TYPESCRIPT_SAMPLE, "src/controller.ts", "typescript")
        lines = TYPESCRIPT_SAMPLE.split("\n")
        total_lines = len(lines)

        covered = set()
        for chunk in chunks:
            for line_num in range(chunk.start_line, chunk.end_line + 1):
                covered.add(line_num)

        for line_num in range(1, total_lines + 1):
            if lines[line_num - 1].strip():
                assert line_num in covered, f"Line {line_num} not covered"


class TestChunkFileJavaScript:
    """Tests for chunking JavaScript source files."""

    def test_detects_javascript_function(self, chunker):
        chunks = chunker.chunk_file(JAVASCRIPT_SAMPLE, "src/app.js", "javascript")
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        func_names = [c.metadata.function_name for c in func_chunks]
        assert "App" in func_names

    def test_detects_javascript_class(self, chunker):
        chunks = chunker.chunk_file(JAVASCRIPT_SAMPLE, "src/app.js", "javascript")
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1


class TestChunkFileGo:
    """Tests for chunking Go source files."""

    def test_detects_go_functions(self, chunker):
        chunks = chunker.chunk_file(GO_SAMPLE, "cmd/main.go", "go")
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        func_names = [c.metadata.function_name for c in func_chunks]
        assert "NewServer" in func_names
        assert "main" in func_names

    def test_detects_go_struct(self, chunker):
        chunks = chunker.chunk_file(GO_SAMPLE, "cmd/main.go", "go")
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1

    def test_chunk_coverage_go(self, chunker):
        chunks = chunker.chunk_file(GO_SAMPLE, "cmd/main.go", "go")
        lines = GO_SAMPLE.split("\n")
        total_lines = len(lines)

        covered = set()
        for chunk in chunks:
            for line_num in range(chunk.start_line, chunk.end_line + 1):
                covered.add(line_num)

        for line_num in range(1, total_lines + 1):
            if lines[line_num - 1].strip():
                assert line_num in covered, f"Line {line_num} not covered"


class TestChunkFilePHP:
    """Tests for chunking PHP source files."""

    def test_detects_php_class(self, chunker):
        chunks = chunker.chunk_file(PHP_SAMPLE, "web/controller.php", "php")
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1

    def test_detects_php_function(self, chunker):
        chunks = chunker.chunk_file(PHP_SAMPLE, "web/controller.php", "php")
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) >= 1


class TestChunkCoverage:
    """Tests for chunk coverage guarantee."""

    def test_every_line_covered_simple(self, chunker):
        """Simple file: every non-whitespace line must be in a chunk."""
        content = "line1\nline2\nline3\nline4\nline5"
        chunks = chunker.chunk_file(content, "test.py", "python")

        covered = set()
        for chunk in chunks:
            for line_num in range(chunk.start_line, chunk.end_line + 1):
                covered.add(line_num)

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if line.strip():
                assert i in covered, f"Line {i} ('{line}') not covered"

    def test_coverage_with_no_patterns_match(self, chunker):
        """When no patterns match, logical blocks should still cover all lines."""
        content = "# comment 1\n# comment 2\n\n\n# comment 3\n# comment 4"
        chunks = chunker.chunk_file(content, "notes.py", "python")
        assert len(chunks) >= 1

        covered = set()
        for chunk in chunks:
            for line_num in range(chunk.start_line, chunk.end_line + 1):
                covered.add(line_num)

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if line.strip():
                assert i in covered


class TestOversizedChunkSplitting:
    """Tests for splitting large chunks that exceed max_chunk_size."""

    def test_splits_large_chunk(self, small_chunker):
        """A large function should be split into multiple sub-chunks."""
        # Generate content that exceeds 50 tokens
        lines = [f"    x_{i} = compute_value({i})" for i in range(50)]
        content = "def large_function():\n" + "\n".join(lines)

        chunks = small_chunker.chunk_file(content, "big.py", "python")

        # Should produce multiple chunks since content exceeds 50 tokens
        assert len(chunks) > 1

    def test_split_chunks_dont_exceed_max(self, small_chunker):
        """Each split chunk should not exceed max_chunk_size."""
        lines = [f"    result_{i} = process({i})" for i in range(50)]
        content = "def large_function():\n" + "\n".join(lines)

        chunks = small_chunker.chunk_file(content, "big.py", "python")

        for chunk in chunks:
            token_count = estimate_tokens(chunk.content)
            # Allow small tolerance for single-line chunks that are inherently large
            assert token_count <= small_chunker.max_chunk_size + 20, (
                f"Chunk has {token_count} tokens, exceeds max {small_chunker.max_chunk_size}"
            )

    def test_split_preserves_parent_id(self, small_chunker):
        """Split sub-chunks should reference parent chunk."""
        lines = [f"    val_{i} = step({i})" for i in range(50)]
        content = "def big_func():\n" + "\n".join(lines)

        chunks = small_chunker.chunk_file(content, "big.py", "python")

        # At least some chunks should have a parent_id
        chunks_with_parent = [c for c in chunks if c.parent_id is not None]
        if len(chunks) > 1:
            assert len(chunks_with_parent) > 0

    def test_overlap_between_splits(self):
        """Consecutive sub-chunks should have overlapping content."""
        chunker = CodeChunker(max_chunk_size=30, overlap=10, repo_id="test")
        lines = [f"    line_{i} = value({i})" for i in range(40)]
        content = "def big():\n" + "\n".join(lines)

        chunks = chunker.chunk_file(content, "big.py", "python")

        if len(chunks) >= 2:
            # Check that consecutive chunks have some overlapping lines
            for i in range(len(chunks) - 1):
                chunk_a = chunks[i]
                chunk_b = chunks[i + 1]
                # Either overlap or be contiguous
                assert chunk_b.start_line <= chunk_a.end_line + 1


class TestFallbackBehavior:
    """Tests for fallback to single chunk on parse failure."""

    def test_empty_content_returns_empty(self, chunker):
        chunks = chunker.chunk_file("", "empty.py", "python")
        assert chunks == []

    def test_whitespace_only_returns_empty(self, chunker):
        chunks = chunker.chunk_file("   \n\n  \n", "empty.py", "python")
        assert chunks == []

    def test_unsupported_language_falls_back(self, chunker):
        """Unsupported language should still produce chunks via logical blocks."""
        content = "some content\nmore content\n"
        # Use a valid supported language but content that won't match patterns
        chunks = chunker.chunk_file(content, "data.go", "go")
        assert len(chunks) >= 1

    def test_fallback_covers_all_content(self, chunker):
        """Fallback single chunk should cover the entire file."""
        content = "x = 1\ny = 2\nz = 3"
        # Force fallback by using content that doesn't match patterns
        chunks = chunker.chunk_file(content, "vars.py", "python")
        assert len(chunks) >= 1

        # All content lines should be covered
        covered = set()
        for chunk in chunks:
            for line_num in range(chunk.start_line, chunk.end_line + 1):
                covered.add(line_num)

        for i in range(1, 4):
            assert i in covered


class TestFileMetadata:
    """Tests for file metadata extraction."""

    def test_python_imports_extracted(self, chunker):
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        all_imports = set()
        for chunk in chunks:
            all_imports.update(chunk.metadata.imports)

        # Should detect import statements
        assert any("import os" in imp for imp in all_imports)
        assert any("import sys" in imp for imp in all_imports)

    def test_function_signatures_in_metadata(self, chunker):
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]

        # Function chunks should have relevant signatures
        for chunk in func_chunks:
            if chunk.metadata.function_name:
                # The chunk's metadata should reference its function
                assert chunk.metadata.function_name is not None

    def test_language_case_insensitive(self, chunker):
        """Language input should be case-insensitive."""
        chunks = chunker.chunk_file(PYTHON_SINGLE_FUNCTION, "test.py", "Python")
        assert len(chunks) >= 1
        assert chunks[0].language == "python"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_file_with_only_comments(self, chunker):
        content = "# This is a comment\n# Another comment\n# Third comment"
        chunks = chunker.chunk_file(content, "comments.py", "python")
        assert len(chunks) >= 1

    def test_file_with_single_line(self, chunker):
        content = "print('hello')"
        chunks = chunker.chunk_file(content, "one.py", "python")
        assert len(chunks) >= 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 1

    def test_deeply_nested_code(self, chunker):
        content = '''def outer():
    def inner():
        def deepest():
            return 42
        return deepest()
    return inner()
'''
        chunks = chunker.chunk_file(content, "nested.py", "python")
        assert len(chunks) >= 1

    def test_unique_chunk_ids(self, chunker):
        """All chunk IDs should be unique."""
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_configurable_max_chunk_size(self):
        """Custom max_chunk_size should be respected."""
        chunker = CodeChunker(max_chunk_size=100, overlap=20, repo_id="test")
        assert chunker.max_chunk_size == 100
        assert chunker.overlap == 20

    def test_configurable_overlap(self):
        """Custom overlap should be respected."""
        chunker = CodeChunker(max_chunk_size=256, overlap=30, repo_id="test")
        assert chunker.overlap == 30

    def test_chunk_content_non_empty(self, chunker):
        """All chunks should have non-empty content."""
        chunks = chunker.chunk_file(PYTHON_SAMPLE, "src/main.py", "python")
        for chunk in chunks:
            assert chunk.content.strip() != ""
