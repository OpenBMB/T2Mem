from __future__ import annotations

"""T2Mem entry point for LongMemEval summary-only wiki rows."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from t2mem import ada_reading
from utils.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL_NAME, REPO_ROOT, ensure_parent

DEFAULT_DATA_PATH = REPO_ROOT / "data" / "longmemeval_processed.json"
DEFAULT_RETRIEVAL_PATH = (
    REPO_ROOT / "data" / "longmemeval" / "longmemeval_retrieval_summary_wiki.jsonl"
)
DEFAULT_WIKI_PATH = (
    REPO_ROOT / "data" / "longmemeval" / "longmemeval_summary_wiki.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "outputs" / "longmemeval_summary_t2mem_answers.jsonl"
)
DEFAULT_LLM_API_KEY = os.getenv("T2MEM_LLM_API_KEY", "")


def summary_wiki_text(row: dict[str, Any]) -> str:
    """The summary branch gives the controller only wiki_id, date, and summary."""
    return "\n".join(
        [
            f"Wiki ID: {row.get('wiki_id')}",
            f"Date: {row.get('source_session_date_time')}",
            f"summary: {row.get('summary')}",
        ]
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T2Mem on LongMemEval summary-only wiki rows.")
    parser.add_argument("--dataset", choices=["longmemeval"], default="longmemeval")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--retrieval-path", default=str(DEFAULT_RETRIEVAL_PATH))
    parser.add_argument("--wiki-path", default=str(DEFAULT_WIKI_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--base-url", default=str(DEFAULT_LLM_BASE_URL))
    parser.add_argument("--model-name", default=str(DEFAULT_LLM_MODEL_NAME))
    parser.add_argument("--api-key", default=DEFAULT_LLM_API_KEY)
    parser.add_argument("--controller-model-name", default=None)
    parser.add_argument("--answer-model-name", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--action-max-tokens", type=int, default=4096)
    parser.add_argument("--answer-max-tokens", type=int, default=1024)
    parser.add_argument("--span-token-budget", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.controller_model_name = args.controller_model_name or args.model_name
    args.answer_model_name = args.answer_model_name or args.model_name

    data_path = Path(args.data_path).expanduser()
    retrieval_path = Path(args.retrieval_path).expanduser()
    wiki_path = Path(args.wiki_path).expanduser()
    output_path = ensure_parent(Path(args.output_path).expanduser())

    retrieval_rows = ada_reading.load_jsonl(retrieval_path)
    if args.limit is not None:
        retrieval_rows = retrieval_rows[: max(0, args.limit)]

    query_sessions_map = ada_reading.load_query_session_map(args.dataset, data_path)
    query_wiki_map_by_question = ada_reading.load_query_wiki_map(wiki_path)
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key or os.getenv("T2MEM_LLM_API_KEY", "") or "EMPTY",
        timeout=args.request_timeout,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Retrieval file: {retrieval_path}")
    print(f"Summary wiki file: {wiki_path}")
    print(f"Output file: {output_path}")
    print(f"Initial top-k wiki: {args.top_k}")
    print(f"Max steps: {args.max_steps}")
    print(f"Controller model: {args.controller_model_name}")
    print(f"Answer model: {args.answer_model_name}")
    print(f"Queries to run: {len(retrieval_rows)}")

    original_wiki_text = ada_reading.wiki_text
    ada_reading.wiki_text = summary_wiki_text
    try:
        with output_path.open("w", encoding="utf-8") as file_obj:
            with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
                futures = [
                    executor.submit(
                        ada_reading.run_t2mem_query,
                        row,
                        dataset=args.dataset,
                        args=args,
                        client=client,
                        query_sessions_map=query_sessions_map,
                        query_wiki_map_by_question=query_wiki_map_by_question,
                    )
                    for row in retrieval_rows
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="summary-t2mem",
                    unit="query",
                ):
                    result: dict[str, Any] = future.result()
                    file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")
                    file_obj.flush()
    finally:
        ada_reading.wiki_text = original_wiki_text

    print(output_path)


if __name__ == "__main__":
    main()
