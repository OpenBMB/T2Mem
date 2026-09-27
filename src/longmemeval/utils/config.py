from __future__ import annotations
"""Path configuration."""
import os
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

REPO_ROOT = Path(__file__).resolve().parents[3]  # release root
DEFAULT_LLM_BASE_URL = os.getenv("T2MEM_LLM_BASE_URL", "")  # OpenAI-compatible endpoint
DEFAULT_LLM_MODEL_NAME = os.getenv("T2MEM_LLM_MODEL_NAME", "")  # must match a model served by the endpoint

# Embedding model used to build the retrieval files. Either point this at a local
# sentence-transformers model, or use the API-based retrieval script.
DEFAULT_EMBEDDING_MODEL_PATH = Path(
    os.getenv("T2MEM_EMBEDDING_MODEL", "bge-m3")
).expanduser()
DEFAULT_LONGMEMEVAL_DATA_PATH = Path(os.getenv("T2MEM_LME_DATA", str(REPO_ROOT / "data" / "longmemeval_processed.json"))).expanduser()

# Create the directory if missing.
def ensure_dir(path: PathLike) -> Path:
    path_obj = Path(path).expanduser()
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

# Create the parent directory if missing.
def ensure_parent(path: PathLike) -> Path:
    path_obj = Path(path).expanduser()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj
