"""Embedding backends shared by all benchmark retrieval representations."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm


def add_embedding_arguments(
    parser: argparse.ArgumentParser, default_model_path: Path
) -> None:
    """Expose the same API/local options on every retrieval entry point."""
    parser.add_argument("--backend", choices=["api", "local"], default="local")
    parser.add_argument("--model-path", default=str(default_model_path))
    parser.add_argument("--base-url", default=os.getenv("T2MEM_EMBEDDING_BASE_URL", ""))
    parser.add_argument(
        "--model-name", default=os.getenv("T2MEM_EMBEDDING_MODEL_NAME", "bge-m3")
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("T2MEM_EMBEDDING_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
        or "EMPTY",
    )
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)


class APIEmbeddingModel:
    """SentenceTransformer-compatible encode interface for an embedding API."""

    def __init__(self, args: argparse.Namespace):
        if not args.base_url:
            raise ValueError(
                "Set --base-url or T2MEM_EMBEDDING_BASE_URL for API embeddings."
            )
        self.client = OpenAI(
            base_url=args.base_url,
            api_key=args.api_key or "EMPTY",
            timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        self.model_name = args.model_name

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive.")
        vectors = []
        batches = range(0, len(texts), batch_size)
        for start in tqdm(batches, disable=not show_progress_bar, desc="embeddings"):
            batch = [
                text if text.strip() else " "
                for text in texts[start : start + batch_size]
            ]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(batch))):
                raise RuntimeError(
                    "Embedding API returned missing or duplicate input indices."
                )
            vectors.extend(item.embedding for item in items)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        try:
            array = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Embedding API returned inconsistent vectors.") from exc
        if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
            raise RuntimeError("Embedding API returned invalid vectors.")
        if normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            array = array / np.where(norms == 0, 1, norms)
        return array


def build_embedding_model(args: argparse.Namespace):
    """Create the requested backend; local model loading remains optional."""
    if args.backend == "api":
        return APIEmbeddingModel(args)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require: python -m pip install -r requirements-local.txt"
        ) from exc
    return SentenceTransformer(str(args.model_path))
