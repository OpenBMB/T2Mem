from __future__ import annotations

import argparse
import json
import re
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
    parser = argparse.ArgumentParser(description="Retrieve LoCoMo raw sessions from processed data.")
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "locomo_processed.json"))
    parser.add_argument("--output-path", default=str(REPO_ROOT / "data" / "locomo" / "locomo_retrieval_raw.jsonl"))
    add_embedding_arguments(parser, DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()

# assemble the session text
def _session_text(session: dict) -> str:
    return f"date_time: {session.get('date_time')}\n{session['transcript_text']}"


def _normalize_evidence(values: list[str] | None) -> list[str]:
    evidence = []
    for value in values or []:
        text = str(value).strip()
        if not text:
            continue
        if text == "D":
            evidence.append("D1:19")
            continue
        if text == "D:11:26":
            evidence.append("D11:26")
            continue

        matches = re.findall(r"D\d+:\d+", text)
        if matches:
            evidence.extend(matches)
        else:
            evidence.append(text)
    return evidence


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser()
    output_path = ensure_parent(Path(args.output_path).expanduser())

    with data_path.open("r", encoding="utf-8") as file_obj:
        samples = json.load(file_obj)

    print(f"Embedding backend: {args.backend}")
    model = build_embedding_model(args)

    with output_path.open("w", encoding="utf-8") as file_obj:
        for sample in tqdm(samples, desc="samples", unit="sample"):
            sessions = sample.get("sessions")
            queries = sample.get("qa")
            if not sessions or not queries:
                continue

            session_texts = [_session_text(session) for session in sessions]
            session_embeddings = model.encode(
                session_texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            session_embeddings = np.asarray(session_embeddings, dtype=np.float32)

            query_texts = [query["question"] for query in queries]
            query_embeddings = model.encode(
                query_texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

            scores = query_embeddings @ session_embeddings.T

            for question_index, (query, query_scores) in enumerate(zip(queries, scores), start=1):
                ranked = np.argsort(-query_scores)[: min(args.top_k, len(sessions))]
                topk_sessions = []
                for index in ranked:
                    session = sessions[int(index)]
                    topk_sessions.append(
                        {
                            "source_session_id": session["source_session_id"],
                            "wiki_id": session["wiki_id"],
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
                            "evidence": _normalize_evidence(query.get("evidence")),
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
