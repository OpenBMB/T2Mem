from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(SRC_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT.parent))

from common.embeddings import add_embedding_arguments, build_embedding_model
from utils.config import DEFAULT_EMBEDDING_MODEL_PATH, REPO_ROOT, ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve LoCoMo wiki sessions from wiki jsonl.")
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "locomo_processed.json"))
    parser.add_argument("--wiki-path", default=str(REPO_ROOT / "data" / "locomo" / "locomo_wiki.jsonl"))
    parser.add_argument("--output-path", default=str(REPO_ROOT / "data" / "locomo" / "locomo_retrieval_wiki.jsonl"))
    add_embedding_arguments(parser, DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()

# assemble the wiki text
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

def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser()
    wiki_path = Path(args.wiki_path).expanduser()
    output_path = ensure_parent(Path(args.output_path).expanduser())

    with data_path.open("r", encoding="utf-8") as file_obj:
        samples = json.load(file_obj)
    with wiki_path.open("r", encoding="utf-8") as file_obj:
        wiki_rows = [json.loads(line) for line in file_obj if line.strip()]

    print(f"Embedding backend: {args.backend}")
    model = build_embedding_model(args)

    sample_to_wiki_rows: dict[str, list[dict]] = {}
    for row in wiki_rows:
        sample_to_wiki_rows.setdefault(row["sample_id"], []).append(row)

    with output_path.open("w", encoding="utf-8") as file_obj:
        for sample in tqdm(samples, desc="samples", unit="sample"):
            queries = sample.get("qa")
            candidate_rows = sample_to_wiki_rows.get(sample["sample_id"])
            if not queries or not candidate_rows:
                continue

            wiki_texts = [_wiki_text(row) for row in candidate_rows]
            wiki_embeddings = model.encode(
                wiki_texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            wiki_embeddings = np.asarray(wiki_embeddings, dtype=np.float32)

            query_embeddings = model.encode(
                [query["question"] for query in queries],
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

            scores = query_embeddings @ wiki_embeddings.T

            for question_index, (query, query_scores) in enumerate(zip(queries, scores), start=1):
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
