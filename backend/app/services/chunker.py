"""AST-based code chunking using regex pattern matching.

Provides language-aware code chunking that identifies function, class,
method, and documentation boundaries. Uses regex patterns as a simpler
alternative to tree-sitter for identifying code structure boundaries.

Falls back to raw content as a single chunk on parse failure.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.models.chunk import ChunkMetadata, ChunkType, CodeChunk
from app.utils.tokenizer import estimate_tokens


# Language-specific regex patterns for identifying code boundaries
LANGUAGE_PATTERNS: dict[str, dict[str, re.Pattern]] = {
    "python": {
        "function": re.compile(
            r"^([ \t]*)(async\s+)?def\s+(\w+)\s*\(", re.MULTILINE
        ),
        "class": re.compile(r"^([ \t]*)class\s+(\w+)\s*[\(:]", re.MULTILINE),
        "documentation": re.compile(
            r'^([ \t]*)"""[\s\S]*?"""', re.MULTILINE
        ),
        "import": re.compile(
            r"^(?:from\s+\S+\s+)?import\s+.+$", re.MULTILINE
        ),
    },
    "typescript": {
        "function": re.compile(
            r"^([ \t]*)(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[\(<]",
            re.MULTILINE,
        ),
        "class": re.compile(
            r"^([ \t]*)(?:export\s+)?(?:abstract\s+)?class\s+(\w+)",
            re.MULTILINE,
        ),
        "method": re.compile(
            r"^([ \t]+)(?:public|private|protected|static|async|get|set|\s)*(\w+)\s*\(",
            re.MULTILINE,
        ),
        "import": re.compile(
            r"^import\s+.+$", re.MULTILINE
        ),
    },
    "javascript": {
        "function": re.compile(
            r"^([ \t]*)(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(",
            re.MULTILINE,
        ),
        "class": re.compile(
            r"^([ \t]*)(?:export\s+)?class\s+(\w+)", re.MULTILINE
        ),
        "method": re.compile(
            r"^([ \t]+)(\w+)\s*\(", re.MULTILINE
        ),
        "import": re.compile(
            r"^(?:import|const\s+\w+\s*=\s*require)\s*[({].+$", re.MULTILINE
        ),
    },
    "php": {
        "function": re.compile(
            r"^([ \t]*)(?:public|private|protected|static|\s)*function\s+(\w+)\s*\(",
            re.MULTILINE,
        ),
        "class": re.compile(
            r"^([ \t]*)(?:abstract\s+)?class\s+(\w+)", re.MULTILINE
        ),
        "import": re.compile(
            r"^(?:use|require|include)\s+.+;$", re.MULTILINE
        ),
    },
    "go": {
        "function": re.compile(
            r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", re.MULTILINE
        ),
        "class": re.compile(
            r"^type\s+(\w+)\s+struct\s*\{", re.MULTILINE
        ),
        "import": re.compile(
            r'^import\s+(?:\([\s\S]*?\)|"[^"]+")$', re.MULTILINE
        ),
    },
}


@dataclass
class CodeBoundary:
    """Represents a detected code boundary in source file."""

    chunk_type: ChunkType
    name: str
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    indent_level: int
    content: str


@dataclass
class FileMetadata:
    """Metadata extracted from a source file."""

    file_path: str
    language: str
    imports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    function_signatures: list[str] = field(default_factory=list)


class CodeChunker:
    """Chunks source code files into semantically meaningful segments.

    Uses regex-based pattern matching to identify function, class, method,
    and documentation boundaries. Falls back to logical block splitting
    (blank line separation) when patterns don't match. Ensures every line
    of the file belongs to at least one chunk (chunk coverage).

    Args:
        max_chunk_size: Maximum tokens per chunk (default 512).
        overlap: Token overlap between split sub-chunks (default 50).
        repo_id: Repository identifier for generated chunks.
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap: int = 50,
        repo_id: str = "default",
    ) -> None:
        """Initialize the CodeChunker.

        Args:
            max_chunk_size: Maximum number of tokens per chunk.
            overlap: Number of overlapping tokens between split sub-chunks.
            repo_id: Repository identifier for generated chunks.
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.repo_id = repo_id

    def chunk_file(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> list[CodeChunk]:
        """Chunk a source file into semantically meaningful segments.

        Attempts regex-based boundary detection first. If that fails or
        produces no results, falls back to raw content as a single chunk.

        Args:
            content: Full text content of the source file.
            file_path: Relative path of the file within the repository.
            language: Programming language of the file.

        Returns:
            List of CodeChunk objects covering every line of the file.
        """
        if not content or not content.strip():
            return []

        language_lower = language.lower()

        try:
            # Extract file-level metadata
            file_metadata = self._extract_metadata(content, file_path, language_lower)

            # Try regex-based boundary detection
            boundaries = self._detect_boundaries(content, language_lower)

            if boundaries:
                chunks = self._create_chunks_from_boundaries(
                    content, file_path, language_lower, boundaries, file_metadata
                )
            else:
                # Fall back to logical block splitting
                chunks = self._split_by_logical_blocks(
                    content, file_path, language_lower, file_metadata
                )

            # Ensure chunk coverage: every line must be in at least one chunk
            chunks = self._ensure_coverage(
                chunks, content, file_path, language_lower, file_metadata
            )

            # Split oversized chunks
            chunks = self._split_oversized_chunks(
                chunks, file_path, language_lower, file_metadata
            )

            return chunks

        except Exception:
            # Fall back to raw content as single chunk on any failure
            return self._fallback_single_chunk(content, file_path, language_lower)

    def _extract_metadata(
        self, content: str, file_path: str, language: str
    ) -> FileMetadata:
        """Extract file-level metadata including imports and function signatures.

        Args:
            content: Full file content.
            file_path: Relative file path.
            language: Programming language.

        Returns:
            FileMetadata with extracted information.
        """
        metadata = FileMetadata(file_path=file_path, language=language)

        patterns = LANGUAGE_PATTERNS.get(language, {})

        # Extract imports
        import_pattern = patterns.get("import")
        if import_pattern:
            for match in import_pattern.finditer(content):
                metadata.imports.append(match.group(0).strip())

        # Extract function signatures
        func_pattern = patterns.get("function")
        if func_pattern:
            for match in func_pattern.finditer(content):
                # Get the full line as signature
                line_start = content.rfind("\n", 0, match.start()) + 1
                line_end = content.find("\n", match.start())
                if line_end == -1:
                    line_end = len(content)
                signature = content[line_start:line_end].strip()
                metadata.function_signatures.append(signature)

        return metadata

    def _detect_boundaries(
        self, content: str, language: str
    ) -> list[CodeBoundary]:
        """Detect code boundaries using language-specific regex patterns.

        Args:
            content: Full file content.
            language: Programming language.

        Returns:
            List of detected code boundaries sorted by start line.
        """
        patterns = LANGUAGE_PATTERNS.get(language, {})
        if not patterns:
            return []

        lines = content.split("\n")
        boundaries: list[CodeBoundary] = []

        # Detect functions
        func_pattern = patterns.get("function")
        if func_pattern:
            for match in func_pattern.finditer(content):
                start_line = content[:match.start()].count("\n") + 1

                # Extract indent and name based on language
                if language == "go":
                    indent = 0
                    name = match.group(1)
                elif language == "python":
                    indent = len(match.group(1))
                    name = match.group(3)
                else:
                    indent = len(match.group(1))
                    name = match.group(2)

                # Find end of function body
                end_line = self._find_block_end(
                    lines, start_line - 1, indent, language
                )

                block_content = "\n".join(lines[start_line - 1:end_line])
                boundaries.append(
                    CodeBoundary(
                        chunk_type=ChunkType.FUNCTION,
                        name=name,
                        start_line=start_line,
                        end_line=end_line,
                        indent_level=indent,
                        content=block_content,
                    )
                )

        # Detect classes
        class_pattern = patterns.get("class")
        if class_pattern:
            for match in class_pattern.finditer(content):
                start_line = content[:match.start()].count("\n") + 1

                # Extract indent and name based on language
                if language == "go":
                    indent = 0
                    name = match.group(1)
                else:
                    indent = len(match.group(1))
                    name = match.group(2)

                # Find end of class body
                end_line = self._find_block_end(
                    lines, start_line - 1, indent, language
                )

                block_content = "\n".join(lines[start_line - 1:end_line])
                boundaries.append(
                    CodeBoundary(
                        chunk_type=ChunkType.CLASS,
                        name=name,
                        start_line=start_line,
                        end_line=end_line,
                        indent_level=indent,
                        content=block_content,
                    )
                )

        # Sort by start line
        boundaries.sort(key=lambda b: b.start_line)
        return boundaries

    def _find_block_end(
        self, lines: list[str], start_idx: int, base_indent: int, language: str
    ) -> int:
        """Find the end line of a code block starting at start_idx.

        Uses indentation for Python and brace matching for other languages.

        Args:
            lines: List of all file lines.
            start_idx: 0-indexed start line of the block.
            base_indent: Indentation level of the block declaration.
            language: Programming language.

        Returns:
            1-indexed end line number.
        """
        if language == "python":
            return self._find_python_block_end(lines, start_idx, base_indent)
        else:
            return self._find_brace_block_end(lines, start_idx)

    def _find_python_block_end(
        self, lines: list[str], start_idx: int, base_indent: int
    ) -> int:
        """Find end of a Python block using indentation.

        Args:
            lines: All file lines.
            start_idx: 0-indexed start line.
            base_indent: Indentation of the def/class line.

        Returns:
            1-indexed end line number.
        """
        last_content_line = start_idx

        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            # Skip empty lines
            if not line.strip():
                continue
            # Get current indentation
            current_indent = len(line) - len(line.lstrip())
            # If indentation drops to or below base, block ended
            if current_indent <= base_indent:
                break
            last_content_line = i

        return last_content_line + 1  # Convert to 1-indexed

    def _find_brace_block_end(
        self, lines: list[str], start_idx: int
    ) -> int:
        """Find end of a brace-delimited block.

        Args:
            lines: All file lines.
            start_idx: 0-indexed start line.

        Returns:
            1-indexed end line number.
        """
        brace_count = 0
        found_opening = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == "{":
                    brace_count += 1
                    found_opening = True
                elif char == "}":
                    brace_count -= 1

            if found_opening and brace_count <= 0:
                return i + 1  # Convert to 1-indexed

        # If no matching brace found, extend to end of file
        return len(lines)

    def _create_chunks_from_boundaries(
        self,
        content: str,
        file_path: str,
        language: str,
        boundaries: list[CodeBoundary],
        file_metadata: FileMetadata,
    ) -> list[CodeChunk]:
        """Create CodeChunk objects from detected boundaries.

        Args:
            content: Full file content.
            file_path: Relative file path.
            language: Programming language.
            boundaries: Detected code boundaries.
            file_metadata: Extracted file metadata.

        Returns:
            List of CodeChunk objects.
        """
        chunks: list[CodeChunk] = []

        for boundary in boundaries:
            metadata = ChunkMetadata(
                function_name=(
                    boundary.name
                    if boundary.chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD)
                    else None
                ),
                class_name=(
                    boundary.name if boundary.chunk_type == ChunkType.CLASS else None
                ),
                imports=file_metadata.imports,
                dependencies=file_metadata.dependencies,
                signatures=(
                    [s for s in file_metadata.function_signatures if boundary.name in s]
                ),
            )

            chunk = CodeChunk(
                id=str(uuid.uuid4()),
                repo_id=self.repo_id,
                file_path=file_path,
                language=language,
                chunk_type=boundary.chunk_type,
                content=boundary.content,
                start_line=boundary.start_line,
                end_line=boundary.end_line,
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks

    def _split_by_logical_blocks(
        self,
        content: str,
        file_path: str,
        language: str,
        file_metadata: FileMetadata,
    ) -> list[CodeChunk]:
        """Split file into chunks by blank line separation.

        Used as fallback when regex patterns don't detect boundaries.

        Args:
            content: Full file content.
            file_path: Relative file path.
            language: Programming language.
            file_metadata: Extracted file metadata.

        Returns:
            List of CodeChunk objects.
        """
        lines = content.split("\n")
        chunks: list[CodeChunk] = []
        current_block_start = 0
        current_block_lines: list[str] = []

        for i, line in enumerate(lines):
            if line.strip() == "" and current_block_lines:
                # Check if the next non-empty line starts a new block
                # (two or more consecutive blank lines)
                next_non_empty = i + 1
                while next_non_empty < len(lines) and lines[next_non_empty].strip() == "":
                    next_non_empty += 1

                if next_non_empty - i >= 2 or i == len(lines) - 1:
                    # End current block
                    block_content = "\n".join(current_block_lines)
                    if block_content.strip():
                        chunk = CodeChunk(
                            id=str(uuid.uuid4()),
                            repo_id=self.repo_id,
                            file_path=file_path,
                            language=language,
                            chunk_type=ChunkType.MODULE,
                            content=block_content,
                            start_line=current_block_start + 1,
                            end_line=current_block_start + len(current_block_lines),
                            metadata=ChunkMetadata(
                                imports=file_metadata.imports,
                                dependencies=file_metadata.dependencies,
                            ),
                        )
                        chunks.append(chunk)
                    current_block_lines = []
                    current_block_start = i + 1
                else:
                    current_block_lines.append(line)
            else:
                if not current_block_lines:
                    current_block_start = i
                current_block_lines.append(line)

        # Handle remaining block
        if current_block_lines:
            block_content = "\n".join(current_block_lines)
            if block_content.strip():
                chunk = CodeChunk(
                    id=str(uuid.uuid4()),
                    repo_id=self.repo_id,
                    file_path=file_path,
                    language=language,
                    chunk_type=ChunkType.MODULE,
                    content=block_content,
                    start_line=current_block_start + 1,
                    end_line=current_block_start + len(current_block_lines),
                    metadata=ChunkMetadata(
                        imports=file_metadata.imports,
                        dependencies=file_metadata.dependencies,
                    ),
                )
                chunks.append(chunk)

        return chunks

    def _ensure_coverage(
        self,
        chunks: list[CodeChunk],
        content: str,
        file_path: str,
        language: str,
        file_metadata: FileMetadata,
    ) -> list[CodeChunk]:
        """Ensure every line of the file is covered by at least one chunk.

        Fills gaps between chunks with MODULE-type chunks.

        Args:
            chunks: Existing chunks.
            content: Full file content.
            file_path: Relative file path.
            language: Programming language.
            file_metadata: Extracted file metadata.

        Returns:
            Complete list of chunks covering all lines.
        """
        if not chunks:
            return self._fallback_single_chunk(content, file_path, language)

        lines = content.split("\n")
        total_lines = len(lines)

        # Track which lines are covered
        covered = set()
        for chunk in chunks:
            for line_num in range(chunk.start_line, chunk.end_line + 1):
                covered.add(line_num)

        # Find uncovered ranges
        gap_chunks: list[CodeChunk] = []
        gap_start: Optional[int] = None

        for line_num in range(1, total_lines + 1):
            if line_num not in covered:
                if gap_start is None:
                    gap_start = line_num
            else:
                if gap_start is not None:
                    # Close the gap
                    gap_end = line_num - 1
                    gap_content = "\n".join(lines[gap_start - 1:gap_end])
                    if gap_content.strip():
                        gap_chunk = CodeChunk(
                            id=str(uuid.uuid4()),
                            repo_id=self.repo_id,
                            file_path=file_path,
                            language=language,
                            chunk_type=ChunkType.MODULE,
                            content=gap_content,
                            start_line=gap_start,
                            end_line=gap_end,
                            metadata=ChunkMetadata(
                                imports=file_metadata.imports,
                                dependencies=file_metadata.dependencies,
                            ),
                        )
                        gap_chunks.append(gap_chunk)
                    gap_start = None

        # Handle trailing gap
        if gap_start is not None:
            gap_end = total_lines
            gap_content = "\n".join(lines[gap_start - 1:gap_end])
            if gap_content.strip():
                gap_chunk = CodeChunk(
                    id=str(uuid.uuid4()),
                    repo_id=self.repo_id,
                    file_path=file_path,
                    language=language,
                    chunk_type=ChunkType.MODULE,
                    content=gap_content,
                    start_line=gap_start,
                    end_line=gap_end,
                    metadata=ChunkMetadata(
                        imports=file_metadata.imports,
                        dependencies=file_metadata.dependencies,
                    ),
                )
                gap_chunks.append(gap_chunk)

        # Combine and sort by start line
        all_chunks = chunks + gap_chunks
        all_chunks.sort(key=lambda c: c.start_line)
        return all_chunks

    def _split_oversized_chunks(
        self,
        chunks: list[CodeChunk],
        file_path: str,
        language: str,
        file_metadata: FileMetadata,
    ) -> list[CodeChunk]:
        """Split chunks that exceed max_chunk_size into sub-chunks with overlap.

        Args:
            chunks: List of chunks to process.
            file_path: Relative file path.
            language: Programming language.
            file_metadata: Extracted file metadata.

        Returns:
            List of chunks where none exceeds max_chunk_size.
        """
        result: list[CodeChunk] = []

        for chunk in chunks:
            token_count = estimate_tokens(chunk.content)
            if token_count <= self.max_chunk_size:
                result.append(chunk)
            else:
                # Split this chunk into sub-chunks with overlap
                sub_chunks = self._split_chunk_with_overlap(
                    chunk, file_path, language, file_metadata
                )
                result.extend(sub_chunks)

        return result

    def _split_chunk_with_overlap(
        self,
        chunk: CodeChunk,
        file_path: str,
        language: str,
        file_metadata: FileMetadata,
    ) -> list[CodeChunk]:
        """Split a single oversized chunk into smaller sub-chunks with overlap.

        Splits by lines, trying to keep each sub-chunk under max_chunk_size
        while maintaining overlap between consecutive sub-chunks.

        Args:
            chunk: The oversized chunk to split.
            file_path: Relative file path.
            language: Programming language.
            file_metadata: Extracted file metadata.

        Returns:
            List of sub-chunks with overlap.
        """
        lines = chunk.content.split("\n")
        sub_chunks: list[CodeChunk] = []
        current_start_idx = 0

        while current_start_idx < len(lines):
            # Find how many lines fit within max_chunk_size
            end_idx = current_start_idx
            current_content = ""

            for i in range(current_start_idx, len(lines)):
                test_content = "\n".join(lines[current_start_idx:i + 1])
                if estimate_tokens(test_content) > self.max_chunk_size:
                    break
                end_idx = i
                current_content = test_content

            # If we couldn't fit even one line, take it anyway to make progress
            if end_idx == current_start_idx and not current_content:
                end_idx = current_start_idx
                current_content = lines[current_start_idx]

            if current_content.strip():
                sub_chunk = CodeChunk(
                    id=str(uuid.uuid4()),
                    repo_id=self.repo_id,
                    file_path=file_path,
                    language=language,
                    chunk_type=chunk.chunk_type,
                    content=current_content,
                    start_line=chunk.start_line + current_start_idx,
                    end_line=chunk.start_line + end_idx,
                    parent_id=chunk.id,
                    metadata=ChunkMetadata(
                        function_name=chunk.metadata.function_name,
                        class_name=chunk.metadata.class_name,
                        imports=file_metadata.imports,
                        dependencies=file_metadata.dependencies,
                    ),
                )
                sub_chunks.append(sub_chunk)

            # Calculate overlap in lines
            if self.overlap > 0 and end_idx + 1 < len(lines):
                # Find how many lines make up the overlap tokens
                overlap_lines = self._calculate_overlap_lines(
                    lines, end_idx, self.overlap
                )
                next_start = end_idx + 1 - overlap_lines
                if next_start <= current_start_idx:
                    next_start = end_idx + 1  # Prevent infinite loop
                current_start_idx = next_start
            else:
                current_start_idx = end_idx + 1

        return sub_chunks

    def _calculate_overlap_lines(
        self, lines: list[str], end_idx: int, overlap_tokens: int
    ) -> int:
        """Calculate how many lines from the end correspond to overlap_tokens.

        Args:
            lines: All lines of the chunk.
            end_idx: End index of current sub-chunk.
            overlap_tokens: Target overlap in tokens.

        Returns:
            Number of lines to overlap.
        """
        overlap_lines = 0
        token_count = 0

        for i in range(end_idx, -1, -1):
            line_tokens = estimate_tokens(lines[i])
            if token_count + line_tokens > overlap_tokens:
                break
            token_count += line_tokens
            overlap_lines += 1

        return max(1, overlap_lines)  # At least 1 line of overlap

    def _fallback_single_chunk(
        self, content: str, file_path: str, language: str
    ) -> list[CodeChunk]:
        """Create chunk(s) from the file content.

        If the content fits within max_chunk_size, creates a single chunk.
        Otherwise, splits by lines into multiple chunks with overlap.

        Used as fallback when parsing fails or for unsupported languages.

        Args:
            content: Full file content.
            file_path: Relative file path.
            language: Programming language.

        Returns:
            List of CodeChunk objects covering the entire file.
        """
        lines = content.split("\n")
        total_tokens = estimate_tokens(content)

        # If it fits in one chunk, return as-is
        if total_tokens <= self.max_chunk_size:
            return [
                CodeChunk(
                    id=str(uuid.uuid4()),
                    repo_id=self.repo_id,
                    file_path=file_path,
                    language=language,
                    chunk_type=ChunkType.MODULE,
                    content=content,
                    start_line=1,
                    end_line=len(lines),
                    metadata=ChunkMetadata(),
                )
            ]

        # Split into multiple chunks by lines
        chunks: list[CodeChunk] = []
        current_start = 0

        while current_start < len(lines):
            # Find how many lines fit within max_chunk_size
            end_idx = current_start
            for i in range(current_start, len(lines)):
                test_content = "\n".join(lines[current_start:i + 1])
                if estimate_tokens(test_content) > self.max_chunk_size:
                    break
                end_idx = i

            # If we couldn't fit even one line, take it anyway
            if end_idx == current_start:
                end_idx = current_start

            chunk_content = "\n".join(lines[current_start:end_idx + 1])

            if chunk_content.strip():
                chunks.append(
                    CodeChunk(
                        id=str(uuid.uuid4()),
                        repo_id=self.repo_id,
                        file_path=file_path,
                        language=language,
                        chunk_type=ChunkType.MODULE,
                        content=chunk_content,
                        start_line=current_start + 1,
                        end_line=end_idx + 1,
                        metadata=ChunkMetadata(),
                    )
                )

            # Move forward with overlap
            if self.overlap > 0 and end_idx + 1 < len(lines):
                overlap_lines = max(1, self.overlap // 10)  # rough line estimate
                next_start = end_idx + 1 - overlap_lines
                if next_start <= current_start:
                    next_start = end_idx + 1
                current_start = next_start
            else:
                current_start = end_idx + 1

        return chunks if chunks else [
            CodeChunk(
                id=str(uuid.uuid4()),
                repo_id=self.repo_id,
                file_path=file_path,
                language=language,
                chunk_type=ChunkType.MODULE,
                content=content[:1000],  # Absolute fallback: first 1000 chars
                start_line=1,
                end_line=min(len(lines), 50),
                metadata=ChunkMetadata(),
            )
        ]
