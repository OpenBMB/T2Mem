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

from t2mem.ada_reading import (
    format_token_usage,
    load_jsonl,
    load_query_session_map,
    load_wiki_map_by_id,
    merge_token_usage,
    new_token_usage,
    run_t2mem_query,
    should_skip_t2mem_row,
    summarize_token_usage_jsonl,
)
from utils.config import (DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL_NAME,
                         DEFAULT_LOCOMO_DATA_PATH, REPO_ROOT, ensure_parent)
from utils.llm_utils import build_client

DATA_PATHS = {
    "locomo": DEFAULT_LOCOMO_DATA_PATH,
}

RETRIEVAL_PATHS = {
    "locomo": REPO_ROOT / "data" / "locomo" / "locomo_retrieval_wiki.jsonl",
}

WIKI_PATHS = {
    "locomo": REPO_ROOT / "data" / "locomo" / "locomo_wiki.jsonl",
}

OUTPUT_PATHS = {
    "locomo": REPO_ROOT / "outputs" / "locomo_t2mem_answers.jsonl",
}


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Run t2mem adaptive reading and answer generation.")
    parser.add_argument("--dataset", choices=["locomo"], default="locomo")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--retrieval-path", default=None)
    parser.add_argument("--wiki-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--base-url", default=str(DEFAULT_LLM_BASE_URL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name", default=str(DEFAULT_LLM_MODEL_NAME))
    parser.add_argument("--controller-model-name", default=None)
    parser.add_argument("--answer-model-name", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=99999999.0)
    parser.add_argument("--action-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=None,
        help="Word budget for controller reasoning: None=unlimited (default), 0=no reasoning, >0=at most N words",
    )
    parser.add_argument(
        "--output-budget-tokens",
        type=int,
        default=None,
        help="Total output-token budget per query (summed over controller steps; does not limit "
        "the number of rounds): None=unlimited (default), >0=hard cap",
    )
    parser.add_argument("--answer-max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--read-mode",
        choices=["full", "wiki-only", "raw-only"],
        default=None,
        help="Reading configuration (mutually exclusive): "
        "full = inject wiki + SEARCH_WIKI + READ_RAW_FROM_WIKI (default) | "
        "wiki-only = ablation A, wiki only, action set {SEARCH_WIKI, ANSWER} | "
        "raw-only = ablation B, raw sessions only, action set {SEARCH_RAW, ANSWER}",
    )
    parser.add_argument(
        "--no-read-raw",
        dest="no_read_raw",
        action="store_true",
        help="[DEPRECATED] alias for --read-mode wiki-only, kept to reproduce earlier runs",
    )
    parser.add_argument(
        "--dump-prompts",
        action="store_true",
        help="Dump the exact controller and answer prompts to the output JSONL "
        "(analysis only; never affects decisions)",
    )
    args = parser.parse_args()

    # Normalize to a single canonical field: --read-mode is authoritative, --no-read-raw is an alias.
    if args.read_mode is None:
        args.read_mode = "wiki-only" if args.no_read_raw else "full"
    elif args.no_read_raw and args.read_mode != "wiki-only":
        parser.error(
            f"--no-read-raw conflicts with --read-mode {args.read_mode}; "
            "use only --read-mode full|wiki-only|raw-only"
        )

    # Derive the two internal fields consumed downstream.
    args.read_unit = "raw" if args.read_mode == "raw-only" else "wiki"
    args.allow_read_raw = args.read_mode == "full"
    return args


def main() -> None:
    args = parse_args()

    args.controller_model_name = args.controller_model_name or args.model_name
    args.answer_model_name = args.answer_model_name or args.model_name

    data_path = Path(args.data_path).expanduser() if args.data_path else DATA_PATHS[args.dataset]
    retrieval_path = Path(args.retrieval_path).expanduser() if args.retrieval_path else RETRIEVAL_PATHS[args.dataset]
    wiki_path = Path(args.wiki_path).expanduser() if args.wiki_path else WIKI_PATHS[args.dataset]
    output_path = ensure_parent(Path(args.output_path).expanduser() if args.output_path else OUTPUT_PATHS[args.dataset])
    token_usage_summary_path = output_path.with_name(f"{output_path.stem}_token_usage_summary.json")

    retrieval_rows = load_jsonl(retrieval_path)
    skipped_category5 = 0
    if args.dataset == "locomo":
        filtered_rows = []
        for row in retrieval_rows:
            if should_skip_t2mem_row(args.dataset, row):
                skipped_category5 += 1
                continue
            filtered_rows.append(row)
        retrieval_rows = filtered_rows
    if args.limit is not None:
        retrieval_rows = retrieval_rows[: max(0, args.limit)]

    query_sessions_map = load_query_session_map(args.dataset, data_path)
    wiki_map_by_id = load_wiki_map_by_id(wiki_path)
    client = build_client(args.base_url, timeout=args.request_timeout, api_key=args.api_key)

    print(f"Dataset: {args.dataset}")
    print(f"Retrieval file: {retrieval_path}")
    print(f"Wiki file: {wiki_path}")
    print(f"Output file: {output_path}")
    print(f"Initial top-k wiki: {args.top_k}")
    print(f"Max steps: {args.max_steps}")
    print(f"Controller model: {args.controller_model_name}")
    print(f"Answer model: {args.answer_model_name}")
    if args.dataset == "locomo":
        print(f"Skipped category 5 queries: {skipped_category5}")
    print(f"Queries to run: {len(retrieval_rows)}")


    run_token_usage = new_token_usage()
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
                    wiki_map_by_id=wiki_map_by_id,
                )
                for row in retrieval_rows
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="t2mem", unit="query"):
                result: dict[str, Any] = future.result()
                merge_token_usage(run_token_usage, result.get("token_usage"))
                file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")
                file_obj.flush()

    completed_token_summary = summarize_token_usage_jsonl(output_path)
    with token_usage_summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "output_path": str(output_path),
                "note": "Token usage comes from response.usage on model responses.",
                "current_run": {
                    "queries_submitted": len(retrieval_rows),
                    "token_usage": run_token_usage,
                },
                "completed_output": completed_token_summary,
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    print(output_path)
    print(f"Current run token usage: {format_token_usage(run_token_usage)}")
    print(
        "Completed output token usage: "
        f"{format_token_usage(completed_token_summary['token_usage'])} "
        f"({completed_token_summary['queries_counted']} queries)"
    )
    print(f"Token usage summary: {token_usage_summary_path}")


if __name__ == "__main__":
    main()
