"""Repository manager - handles cloning GitHub repos and managing local copies."""

import os
import shutil
import subprocess
import tempfile
import re
from dataclasses import dataclass


@dataclass
class CloneResult:
    """Result of a repository clone operation."""
    success: bool
    local_path: str
    repo_name: str
    error: str | None = None


# Directory to store cloned repos
REPOS_DIR = os.path.join(tempfile.gettempdir(), "codeqmate_repos")


def validate_github_url(url: str) -> bool:
    """Check if a URL is a valid GitHub repository URL."""
    pattern = r'^https?://github\.com/[\w\-\.]+/[\w\-\.]+/?$'
    return bool(re.match(pattern, url.strip()))


def extract_repo_name(url: str) -> str:
    """Extract repository name from GitHub URL."""
    # Remove trailing slash and .git
    clean = url.strip().rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]
    return clean.split("/")[-1]


def clone_repository(github_url: str) -> CloneResult:
    """Clone a GitHub repository to a temporary local directory.

    Args:
        github_url: Full GitHub URL (e.g. https://github.com/user/repo)

    Returns:
        CloneResult with success status and local path.
    """
    if not validate_github_url(github_url):
        return CloneResult(
            success=False,
            local_path="",
            repo_name="",
            error="Invalid GitHub URL. Expected format: https://github.com/owner/repo"
        )

    repo_name = extract_repo_name(github_url)
    local_path = os.path.join(REPOS_DIR, repo_name)

    # Create repos directory if it doesn't exist
    os.makedirs(REPOS_DIR, exist_ok=True)

    # If already cloned, try to remove (may fail on Windows due to file locks)
    if os.path.exists(local_path):
        try:
            # Try git pull to update
            pull_result = subprocess.run(
                ["git", "pull"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=local_path,
            )
            if pull_result.returncode == 0:
                return CloneResult(
                    success=True,
                    local_path=local_path,
                    repo_name=repo_name,
                )
        except Exception:
            pass

        # If pull failed, try fresh clone
        try:
            shutil.rmtree(local_path, ignore_errors=True)
        except Exception:
            pass

        # If still exists after rmtree attempt, just reuse it
        if os.path.exists(local_path):
            return CloneResult(
                success=True,
                local_path=local_path,
                repo_name=repo_name,
            )

    try:
        # Clone with depth=1 for faster download (only latest commit)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url.strip(), local_path],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        if result.returncode != 0:
            return CloneResult(
                success=False,
                local_path="",
                repo_name=repo_name,
                error=f"Git clone failed: {result.stderr.strip()}"
            )

        return CloneResult(
            success=True,
            local_path=local_path,
            repo_name=repo_name,
        )

    except subprocess.TimeoutExpired:
        return CloneResult(
            success=False,
            local_path="",
            repo_name=repo_name,
            error="Clone timed out (>2 minutes). Repository may be too large."
        )
    except FileNotFoundError:
        return CloneResult(
            success=False,
            local_path="",
            repo_name=repo_name,
            error="Git is not installed. Please install git first."
        )
    except Exception as e:
        return CloneResult(
            success=False,
            local_path="",
            repo_name=repo_name,
            error=f"Clone failed: {str(e)}"
        )
