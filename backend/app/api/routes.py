"""API routes for CodeQ-Mate.

Provides endpoints for:
- POST /api/query - Ask questions about indexed code
- POST /api/ingest - Index a GitHub repository
- GET /api/status - Check if a repo is indexed
- GET /health - Health check
"""

import html
from typing import Optional
import os
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.services.answer_generator import AnswerGenerator
from app.services.query_processor import QueryProcessor
from app.services.search_engine import get_search_engine
from app.services.repo_manager import clone_repository, validate_github_url

router = APIRouter()


# --- Request/Response Models ---


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""
    question: str = Field(min_length=1, max_length=1000)


class SourceItem(BaseModel):
    """A single source reference in the response."""
    file_path: str
    function_name: Optional[str] = None
    start_line: int
    end_line: int
    snippet: str
    relevance: float


class QueryResponse(BaseModel):
    """Response body for the query endpoint."""
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Request body for the ingestion endpoint."""
    github_url: str = Field(min_length=1)


class IngestResponse(BaseModel):
    """Response body for the ingestion endpoint."""
    status: str
    repo_name: str = ""
    total_files: int = 0
    total_chunks: int = 0
    languages: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class StatusResponse(BaseModel):
    """Response body for the status endpoint."""
    is_indexed: bool
    repo_name: str
    total_chunks: int


# --- Authentication ---


async def verify_api_key(x_api_key: str = Header(default="default-key")) -> str:
    """Verify API key. Accepts any non-empty key for demo mode."""
    if not x_api_key or not x_api_key.strip():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# --- Endpoints ---


@router.post("/api/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    request: IngestRequest,
    api_key: str = Depends(verify_api_key),
) -> IngestResponse:
    """Clone a GitHub repository and index it for searching.

    Accepts a GitHub URL, clones the repo locally, parses all source files,
    chunks them, and builds a BM25 search index in memory.

    Example:
        POST /api/ingest
        {"github_url": "https://github.com/pallets/flask"}
    """
    github_url = request.github_url.strip()

    # Validate URL
    if not validate_github_url(github_url):
        return IngestResponse(
            status="error",
            error="Invalid GitHub URL. Use format: https://github.com/owner/repo"
        )

    # Clone the repository
    clone_result = clone_repository(github_url)

    if not clone_result.success:
        return IngestResponse(
            status="error",
            repo_name=clone_result.repo_name,
            error=clone_result.error,
        )

    # Ingest into search engine
    try:
        engine = get_search_engine()
        stats = engine.ingest_local_repo(
            repo_path=clone_result.local_path,
            repo_name=clone_result.repo_name,
        )

        return IngestResponse(
            status="success",
            repo_name=stats.repo_name,
            total_files=stats.total_files,
            total_chunks=stats.total_chunks,
            languages=stats.languages_found,
        )
    except Exception as e:
        return IngestResponse(
            status="error",
            repo_name=clone_result.repo_name,
            error=f"Ingestion failed: {str(e)}",
        )


@router.post("/api/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key),
) -> QueryResponse:
    """Ask a question about the indexed repository.

    Uses BM25 retrieval to find relevant code chunks, then optionally
    uses Gemini LLM to generate a readable answer.

    The repository must be indexed first via POST /api/ingest.
    """
    engine = get_search_engine()

    if not engine.is_indexed:
        return QueryResponse(
            answer="No repository has been indexed yet. Please ingest a GitHub repository first using the 'Index Repository' button.",
            sources=[],
            confidence=0.0,
            metadata={"error": "no_repo_indexed"},
        )

    # Sanitize input
    question = html.escape(request.question.strip(), quote=True)

    # Search the indexed repo
    results = engine.search(question, top_k=5)

    if not results:
        return QueryResponse(
            answer=f"No relevant code found for your question in '{engine.repo_name}'. Try rephrasing or using specific function/class names.",
            sources=[],
            confidence=0.0,
            metadata={"repo_name": engine.repo_name},
        )

    # Build answer from retrieved chunks
    answer_generator = AnswerGenerator(llm_client=None, max_context_tokens=4096)

    # Try with Gemini if available
    try:
        from app.services.gemini_client import GeminiClient
        gemini = GeminiClient()
        answer_generator = AnswerGenerator(llm_client=gemini, max_context_tokens=4096)
    except (ValueError, ImportError):
        pass  # No Gemini key - use fallback mode

    # Generate answer
    grounded_answer = await answer_generator.generate_answer(question, results)

    # Build source items
    sources = [
        SourceItem(
            file_path=src.file_path,
            function_name=src.function_name,
            start_line=src.start_line,
            end_line=src.end_line,
            snippet=src.snippet,
            relevance=src.relevance,
        )
        for src in grounded_answer.sources
    ]

    return QueryResponse(
        answer=grounded_answer.answer_text,
        sources=sources,
        confidence=grounded_answer.confidence,
        metadata={
            "repo_name": engine.repo_name,
            **grounded_answer.retrieval_metadata,
        },
    )


@router.get("/api/status", response_model=StatusResponse)
async def status_endpoint() -> StatusResponse:
    """Check if a repository has been indexed and return stats."""
    engine = get_search_engine()
    status = engine.get_status()
    return StatusResponse(
        is_indexed=status["is_indexed"],
        repo_name=status["repo_name"],
        total_chunks=status["total_chunks"],
    )


@router.get("/api/filetree")
async def filetree_endpoint() -> dict:
    """Return the folder structure of the indexed repository.

    Returns a nested tree structure of all indexed files grouped by directory.
    """
    engine = get_search_engine()

    if not engine.is_indexed or not engine.chunks:
        return {"tree": [], "repo_name": ""}

    # Collect unique file paths
    file_paths = sorted(set(chunk.file_path for chunk in engine.chunks))

    # Build nested tree structure
    tree = _build_file_tree(file_paths)

    return {"tree": tree, "repo_name": engine.repo_name}


def _build_file_tree(file_paths: list[str]) -> list[dict]:
    """Build a nested tree structure from flat file paths.

    Args:
        file_paths: Sorted list of relative file paths (e.g. "src/main.py")

    Returns:
        Nested list of dicts with {name, type, children?, path?}
    """
    root: dict = {}

    for path in file_paths:
        parts = path.split("/")
        current = root
        for i, part in enumerate(parts):
            if part not in current:
                current[part] = {}
            current = current[part]

    def dict_to_tree(node: dict, prefix: str = "") -> list[dict]:
        items: list[dict] = []
        for name, children in sorted(node.items()):
            full_path = f"{prefix}/{name}" if prefix else name
            if children:  # Has children = folder
                items.append({
                    "name": name,
                    "type": "folder",
                    "path": full_path,
                    "children": dict_to_tree(children, full_path),
                })
            else:  # No children = file
                items.append({
                    "name": name,
                    "type": "file",
                    "path": full_path,
                })
        return items

    return dict_to_tree(root)



@router.get("/api/file/content")
async def get_file_content(file_path: str) -> dict:
    """Get the content of a specific file from the indexed repository.
    
    Args:
        file_path: Relative path to the file (e.g., "src/main.py")
    
    Returns:
        Dict with file content, language, and metadata
    """
    engine = get_search_engine()
    
    if not engine.is_indexed:
        raise HTTPException(status_code=404, detail="No repository indexed")
    
    # Find the chunk with matching file_path
    file_chunks = [c for c in engine.chunks if c.file_path == file_path]
    
    if not file_chunks:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Try to read the full file from disk
    try:
        repo_path = Path(engine.repo_path) if hasattr(engine, 'repo_path') else None
        if repo_path:
            full_path = repo_path / file_path
            if full_path.exists():
                content = full_path.read_text(encoding='utf-8', errors='ignore')
                language = file_chunks[0].language
                
                # Count lines
                lines = content.split('\n')
                total_lines = len(lines)
                
                return {
                    "file_path": file_path,
                    "content": content,
                    "language": language,
                    "total_lines": total_lines,
                    "size_bytes": len(content.encode('utf-8')),
                }
    except Exception as e:
        # Fallback to chunks if file read fails
        pass
    
    # Fallback: reconstruct from chunks
    sorted_chunks = sorted(file_chunks, key=lambda c: c.start_line)
    content = "\n".join(chunk.content for chunk in sorted_chunks)
    
    return {
        "file_path": file_path,
        "content": content,
        "language": file_chunks[0].language,
        "total_lines": sorted_chunks[-1].end_line if sorted_chunks else 0,
        "from_chunks": True,
    }


@router.get("/api/file/open")
async def open_file_in_editor(file_path: str, line: int = 1) -> dict:
    """Generate a URL to open the file in VS Code or other editors.
    
    Args:
        file_path: Relative path to the file
        line: Line number to jump to (default 1)
    
    Returns:
        Dict with various editor protocol URLs
    """
    engine = get_search_engine()
    
    if not engine.is_indexed:
        raise HTTPException(status_code=404, detail="No repository indexed")
    
    # Get absolute path
    repo_path = Path(engine.repo_path) if hasattr(engine, 'repo_path') else None
    if not repo_path:
        raise HTTPException(status_code=500, detail="Repository path not found")
    
    full_path = repo_path / file_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    absolute_path = str(full_path.absolute())
    
    # Generate various editor URLs
    return {
        "file_path": file_path,
        "absolute_path": absolute_path,
        "line": line,
        "editor_urls": {
            "vscode": f"vscode://file/{absolute_path}:{line}",
            "vscode_insiders": f"vscode-insiders://file/{absolute_path}:{line}",
            "intellij": f"idea://open?file={absolute_path}&line={line}",
            "sublime": f"subl://open?url=file://{absolute_path}&line={line}",
            "atom": f"atom://core/open/file?filename={absolute_path}&line={line}",
        }
    }
