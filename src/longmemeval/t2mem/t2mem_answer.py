from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from t2mem.ada_reading import load_jsonl, load_query_session_map, load_query_wiki_map, run_t2mem_query
from utils.config import (DEFAULT_LONGMEMEVAL_DATA_PATH, DEFAULT_LLM_BASE_URL,
                         DEFAULT_LLM_MODEL_NAME, REPO_ROOT, ensure_parent)
from utils.llm_utils import build_client

DEFAULT_DATA_PATH = DEFAULT_LONGMEMEVAL_DATA_PATH
DEFAULT_RETRIEVAL_PATH = REPO_ROOT / "data" / "longmemeval" / "longmemeval_retrieval_wiki.jsonl"
DEFAULT_WIKI_PATH = REPO_ROOT / "data" / "longmemeval" / "longmemeval_wiki.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "longmemeval_t2mem_answers.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run t2mem adaptive reading for LongMemEval.")
    parser.add_argument("--dataset", choices=["longmemeval"], default="longmemeval")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--retrieval-path", default=None)
    parser.add_argument("--wiki-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--base-url", default=str(DEFAULT_LLM_BASE_URL))
    parser.add_argument("--api-key", default=None,
                        help="API key override; otherwise use T2MEM_LLM_API_KEY or EMPTY")
    parser.add_argument("--model-name", default=str(DEFAULT_LLM_MODEL_NAME))
    parser.add_argument("--controller-model-name", default=None)
    parser.add_argument("--answer-model-name", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--action-max-tokens", type=int, default=4096)
    parser.add_argument("--answer-max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--read-mode",
        choices=["full", "wiki-only", "raw-only"],
        default="full",
        help="Reading configuration (mutually exclusive): "
        "full = inject wiki + SEARCH_WIKI + READ_RAW_FROM_WIKI (default) | "
        "wiki-only = ablation A, wiki only, action set {SEARCH_WIKI, ANSWER} | "
        "raw-only = ablation B, raw sessions only, action set {SEARCH_RAW, ANSWER}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.controller_model_name = args.controller_model_name or args.model_name
    args.answer_model_name = args.answer_model_name or args.model_name

    data_path = Path(args.data_path).expanduser() if args.data_path else DEFAULT_DATA_PATH
    retrieval_path = Path(args.retrieval_path).expanduser() if args.retrieval_path else DEFAULT_RETRIEVAL_PATH
    wiki_path = Path(args.wiki_path).expanduser() if args.wiki_path else DEFAULT_WIKI_PATH
    output_path = ensure_parent(Path(args.output_path).expanduser() if args.output_path else DEFAULT_OUTPUT_PATH)

    retrieval_rows = load_jsonl(retrieval_path)
    if args.limit is not None:
        retrieval_rows = retrieval_rows[: max(0, args.limit)]

    query_sessions_map = load_query_session_map(args.dataset, data_path)
    query_wiki_map_by_question = load_query_wiki_map(wiki_path)
    client = build_client(args.base_url, timeout=args.request_timeout, api_key=args.api_key)

    print(f"Dataset: {args.dataset}")
    print(f"Retrieval file: {retrieval_path}")
    print(f"Wiki file: {wiki_path}")
    print(f"Output file: {output_path}")
    print(f"Initial top-k wiki: {args.top_k}")
    print(f"Max steps: {args.max_steps}")
    print(f"Controller model: {args.controller_model_name}")
    print(f"Answer model: {args.answer_model_name}")
    print(f"Queries to run: {len(retrieval_rows)}")

    with output_path.open("w", encoding="utf-8") as file_obj:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = [
                executor.submit(
                    run_t2mem_query,
                    row,
                    dataset=args.dataset,
                    args=args,
                    client=client,
                    query_sessions_map=query_sessions_map,
                    query_wiki_map_by_question=query_wiki_map_by_question,
                )
                for row in retrieval_rows
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="t2mem", unit="query"):
                result: dict[str, Any] = future.result()
                file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")
                file_obj.flush()

    print(output_path)


if __name__ == "__main__":
    main()
