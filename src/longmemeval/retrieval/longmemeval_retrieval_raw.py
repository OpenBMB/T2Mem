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

from utils.config import DEFAULT_EMBEDDING_MODEL_PATH, REPO_ROOT, ensure_parent


def load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence_transformers is required for raw retrieval.") from exc
    return SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve LongMemEval raw sessions from processed data.")
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "longmemeval_processed.json"))
    #parser.add_argument("--output-path",default=str(REPO_ROOT / "outputs" / "longmemeval" / "longmemeval_retrieval_raw.jsonl"))
    parser.add_argument("--output-path", default=str(REPO_ROOT / "data" / "longmemeval" / "longmemeval_retrieval_raw.jsonl"))
    parser.add_argument("--model-path", default=str(DEFAULT_EMBEDDING_MODEL_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()

# assemble the session text
def _session_text(session: dict) -> str:
    return f"date_time: {session.get('date_time')}\n{session['transcript_text']}"

# assemble the query
def _query_text(record: dict) -> str:
    return f"question_date: {record.get('question_date')}\nquestion: {record['question']}"


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser()
    output_path = ensure_parent(Path(args.output_path).expanduser())

    with data_path.open("r", encoding="utf-8") as file_obj:
        records = json.load(file_obj)

    print(f"Loading embedding model from {args.model_path} ...")
    model = load_sentence_transformer()(str(args.model_path))
    query_embeddings = model.encode(
        [_query_text(record) for record in records],
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

    with output_path.open("w", encoding="utf-8") as file_obj:
        for record, query_embedding in tqdm(
            zip(records, query_embeddings),
            total=len(records),
            desc="queries",
            unit="query",
        ):
            sessions = record.get("sessions")
            if not sessions:
                continue

            session_texts = [_session_text(session) for session in sessions]
            session_embeddings = model.encode(
                session_texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            session_embeddings = np.asarray(session_embeddings, dtype=np.float32)

            query_scores = session_embeddings @ query_embedding
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
