from __future__ import annotations

"""Retrieve LongMemEval summary-only wiki rows within each question split."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.config import DEFAULT_EMBEDDING_MODEL_PATH, REPO_ROOT, ensure_parent


DEFAULT_WIKI_PATH = (
    REPO_ROOT / "data" / "longmemeval" / "longmemeval_summary_wiki.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "data" / "longmemeval" / "longmemeval_retrieval_summary_wiki.jsonl"
)


def load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence_transformers is required for summary wiki retrieval.") from exc
    return SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve LongMemEval summary-only wiki rows.")
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "longmemeval_processed.json"))
    parser.add_argument("--wiki-path", default=str(DEFAULT_WIKI_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_EMBEDDING_MODEL_PATH))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def wiki_text(row: dict) -> str:
    """The summary branch embeds only the session date and natural-language summary."""
    return "\n".join(
        [
            f"date_time: {row.get('source_session_date_time')}",
            f"summary: {row.get('summary')}",
        ]
    )


def query_text(record: dict) -> str:
    return f"question_date: {record.get('question_date')}\nquestion: {record['question']}"


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser()
    wiki_path = Path(args.wiki_path).expanduser()
    output_path = ensure_parent(Path(args.output_path).expanduser())

    with data_path.open("r", encoding="utf-8") as file_obj:
        records = json.load(file_obj)
    with wiki_path.open("r", encoding="utf-8") as file_obj:
        wiki_rows = []
        for line in file_obj:
            if not line.strip():
                continue
            row = json.loads(line)
            if "error" not in row:
                wiki_rows.append(row)

    print(f"Loading embedding model from {args.model_path} ...")
    model = load_sentence_transformer()(str(args.model_path))

    query_embeddings = model.encode(
        [query_text(record) for record in records],
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

    question_to_wiki_rows: dict[str, list[dict]] = {}
    for row in wiki_rows:
        question_to_wiki_rows.setdefault(str(row["question_id"]), []).append(row)

    with output_path.open("w", encoding="utf-8") as file_obj:
        for record, query_embedding in tqdm(
            zip(records, query_embeddings),
            total=len(records),
            desc="queries",
            unit="query",
        ):
            candidate_rows = question_to_wiki_rows.get(str(record["question_id"]), [])
            if not candidate_rows:
                continue

            wiki_embeddings = model.encode(
                [wiki_text(row) for row in candidate_rows],
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            wiki_embeddings = np.asarray(wiki_embeddings, dtype=np.float32)

            query_scores = wiki_embeddings @ query_embedding
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
                        "question_id": record["question_id"],
                        "question_type": record.get("question_type"),
                        "question_date": record.get("question_date"),
                        "question": record["question"],
                        "answer": record.get("answer"),
                        "answer_session_ids": record.get("answer_session_ids"),
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
