"""
Clear all Python cache files to resolve Pydantic validation issues.
Run this if you encounter validation errors after code updates.
"""
import os
import shutil
from pathlib import Path


def clear_pycache(root_dir: str = "."):
    """Recursively remove all __pycache__ directories."""
    root = Path(root_dir)
    count = 0
    
    for pycache_dir in root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            print(f"✓ Removed: {pycache_dir}")
            count += 1
        except Exception as e:
            print(f"✗ Failed to remove {pycache_dir}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Cleared {count} cache directories")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("Clearing Python cache files...\n")
    clear_pycache("backend")
    clear_pycache("frontend")
    print("\n✓ Cache cleared successfully!")
    print("You can now restart your backend server.")
