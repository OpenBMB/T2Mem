from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.config import DEFAULT_LOCOMO_DATA_PATH, REPO_ROOT, ensure_parent


DEFAULT_EMBEDDING_BASE_URL = os.getenv(
    "T2MEM_EMBEDDING_BASE_URL", ""
)
DEFAULT_EMBEDDING_MODEL_NAME = os.getenv("T2MEM_EMBEDDING_MODEL_NAME", "bge-m3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve LoCoMo wiki sessions with OpenAI-compatible embeddings."
    )
    parser.add_argument("--data-path", default=str(DEFAULT_LOCOMO_DATA_PATH))
    parser.add_argument(
        "--wiki-path", default=str(REPO_ROOT / "data" / "locomo" / "locomo_wiki.jsonl")
    )
    parser.add_argument(
        "--output-path",
        default=str(REPO_ROOT / "data" / "locomo" / "locomo_retrieval_wiki.jsonl"),
    )
    parser.add_argument("--base-url", default=DEFAULT_EMBEDDING_BASE_URL)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument(
        "--api-key",
        default=os.getenv("T2MEM_EMBEDDING_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or "EMPTY",
        help="API key; uses T2MEM_EMBEDDING_API_KEY, OPENAI_API_KEY, or EMPTY for local servers.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def _items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _wiki_text(row: dict) -> str:
    return "\n".join(
        [
            f"date_time: {row.get('source_session_date_time')}",
            f"summary: {row.get('summary')}",
            "event_core:",
            *[f"- {item}" for item in _items(row.get("event_core"))],
            "atomic_facts:",
            *[f"- {item}" for item in _items(row.get("atomic_facts"))],
            "entities:",
            *[f"- {item}" for item in _items(row.get("entities"))],
            "temporal_clues:",
            *[f"- {item}" for item in _items(row.get("temporal_clues"))],
            "resolved_temporal_facts:",
            *[f"- {item}" for item in _items(row.get("resolved_temporal_facts"))],
            "ongoing_threads:",
            *[f"- {item}" for item in _items(row.get("ongoing_threads"))],
        ]
    )


def _load_openai():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is required for API-based wiki retrieval.") from exc
    return OpenAI


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def _embedding_from_item(item: object) -> list[float]:
    if isinstance(item, dict):
        return item["embedding"]
    return item.embedding


def embed_texts(
    *,
    client: object,
    model_name: str,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [text if text.strip() else " " for text in texts[start : start + batch_size]]
        response = client.embeddings.create(model=model_name, input=batch)
        response_embeddings = [_embedding_from_item(item) for item in response.data]
        if len(response_embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding API returned {len(response_embeddings)} vectors for {len(batch)} inputs."
            )
        embeddings.extend(response_embeddings)

    array = np.asarray(embeddings, dtype=np.float32)
    return _normalize_embeddings(array)


def main() -> None:
    args = parse_args()
    if not args.base_url:
        raise RuntimeError("Set --base-url or T2MEM_EMBEDDING_BASE_URL for the embedding endpoint.")

    data_path = Path(args.data_path).expanduser()
    wiki_path = Path(args.wiki_path).expanduser()
    output_path = ensure_parent(Path(args.output_path).expanduser())

    with data_path.open("r", encoding="utf-8") as file_obj:
        samples = json.load(file_obj)
    with wiki_path.open("r", encoding="utf-8") as file_obj:
        wiki_rows = [json.loads(line) for line in file_obj if line.strip()]

    OpenAI = _load_openai()
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key or "EMPTY",
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )

    print(f"Using embedding API: {args.base_url}")
    print(f"Using embedding model: {args.model_name}")

    sample_to_wiki_rows: dict[str, list[dict]] = {}
    for row in wiki_rows:
        sample_to_wiki_rows.setdefault(row["sample_id"], []).append(row)

    with output_path.open("w", encoding="utf-8") as file_obj:
        for sample in tqdm(samples, desc="samples", unit="sample"):
            queries = sample.get("qa")
            candidate_rows = sample_to_wiki_rows.get(sample["sample_id"])
            if not queries or not candidate_rows:
                continue

            wiki_embeddings = embed_texts(
                client=client,
                model_name=args.model_name,
                texts=[_wiki_text(row) for row in candidate_rows],
                batch_size=args.batch_size,
            )
            query_embeddings = embed_texts(
                client=client,
                model_name=args.model_name,
                texts=[query["question"] for query in queries],
                batch_size=args.batch_size,
            )
            scores = query_embeddings @ wiki_embeddings.T

            for question_index, (query, query_scores) in enumerate(
                zip(queries, scores), start=1
            ):
                ranked = np.argsort(-query_scores)[: min(args.top_k, len(candidate_rows))]
                topk_sessions = []
                for index in ranked:
                    row = candidate_rows[int(index)]
                    topk_sessions.append(
                        {
                            "source_session_id": row["source_session_id"],
                            "wiki_id": row["wiki_id"],
                            "score": float(query_scores[int(index)]),
                        }
                    )

                file_obj.write(
                    json.dumps(
                        {
                            "sample_id": sample["sample_id"],
                            "query_id": f"{sample['sample_id']}_q{question_index}",
                            "question": query["question"],
                            "answer": query.get("answer"),
                            "evidence": query.get("evidence"),
                            "category": query.get("category"),
                            "topk_sessions": topk_sessions,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                file_obj.flush()

    print(output_path)


if __name__ == "__main__":
    main()
