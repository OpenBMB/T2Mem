from __future__ import annotations
"""Minimal config used by the evaluation scripts.

The judges only need a repository root and a directory helper, so this stays
independent from the pipeline config under src/*/utils/config.py.
"""

import os
from pathlib import Path

REPO_ROOT = Path(os.getenv("T2MEM_ROOT", str(Path(__file__).resolve().parents[1])))


def ensure_parent(path: Path) -> Path:
    """Create the parent directory of ``path`` if missing, and return ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
