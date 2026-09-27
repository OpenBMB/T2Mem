from __future__ import annotations

"""Retrieval over summary-only wikis; the structured-wiki branch is unaffected."""

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
    REPO_ROOT / "data" / "locomo" / "locomo_summary_wiki.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "data" / "locomo" / "locomo_retrieval_summary_wiki.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve over summary-only LoCoMo wikis")
    parser.add_argument(
        "--data-path",
        default=str(REPO_ROOT / "data" / "locomo_processed.json"),
    )
    parser.add_argument("--wiki-path", default=str(DEFAULT_WIKI_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_EMBEDDING_MODEL_PATH))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def wiki_text(row: dict) -> str:
    """The summary branch embeds only the date and the summary."""
    return "\n".join(
        [
            f"date_time: {row.get('source_session_date_time')}",
            f"summary: {row.get('summary')}",
        ]
    )


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser()
    wiki_path = Path(args.wiki_path).expanduser()
    output_path = ensure_parent(Path(args.output_path).expanduser())

    with data_path.open("r", encoding="utf-8") as file_obj:
        samples = json.load(file_obj)
    with wiki_path.open("r", encoding="utf-8") as file_obj:
        wiki_rows = []
        for line in file_obj:
            if not line.strip():
                continue
            row = json.loads(line)
            if "error" not in row:
                wiki_rows.append(row)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required") from exc

    print(f"Loading embedding model: {args.model_path}")
    model = SentenceTransformer(str(args.model_path))

    sample_wikis: dict[str, list[dict]] = {}
    for row in wiki_rows:
        sample_wikis.setdefault(str(row["sample_id"]), []).append(row)

    with output_path.open("w", encoding="utf-8") as file_obj:
        for sample in tqdm(samples, desc="samples", unit="sample"):
            queries = sample.get("qa", [])
            candidates = sample_wikis.get(str(sample["sample_id"]), [])
            if not queries or not candidates:
                continue

            wiki_embeddings = model.encode(
                [wiki_text(row) for row in candidates],
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            query_embeddings = model.encode(
                [query["question"] for query in queries],
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            scores = np.asarray(query_embeddings, dtype=np.float32) @ np.asarray(
                wiki_embeddings, dtype=np.float32
            ).T

            for index, (query, query_scores) in enumerate(
                zip(queries, scores), start=1
            ):
                ranked = np.argsort(-query_scores)[
                    : min(args.top_k, len(candidates))
                ]
                topk_sessions = [
                    {
                        "source_session_id": candidates[int(rank)][
                            "source_session_id"
                        ],
                        "wiki_id": candidates[int(rank)]["wiki_id"],
                        "score": float(query_scores[int(rank)]),
                    }
                    for rank in ranked
                ]
                file_obj.write(
                    json.dumps(
                        {
                            "sample_id": sample["sample_id"],
                            "query_id": f"{sample['sample_id']}_q{index}",
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

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
