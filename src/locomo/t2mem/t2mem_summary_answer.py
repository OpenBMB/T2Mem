from __future__ import annotations

"""Standalone T2Mem entry point for summary-only wikis; the structured-wiki branch is untouched."""

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

from t2mem import ada_reading
from utils.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL_NAME, REPO_ROOT, ensure_parent
from utils.llm_utils import build_client


DEFAULT_RETRIEVAL_PATH = (
    REPO_ROOT / "data" / "locomo" / "locomo_retrieval_summary_wiki.jsonl"
)
DEFAULT_WIKI_PATH = (
    REPO_ROOT / "data" / "locomo" / "locomo_summary_wiki.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "outputs" / "locomo_summary_t2mem_answers.jsonl"
)


def summary_wiki_text(row: dict[str, Any]) -> str:
    """The summary branch sends only the wiki_id, date and summary to the controller."""
    return "\n".join(
        [
            f"Wiki ID: {row.get('wiki_id')}",
            f"Date: {row.get('source_session_date_time')}",
            f"summary: {row.get('summary')}",
        ]
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T2Mem over summary-only wikis")
    parser.add_argument(
        "--data-path",
        default=str(REPO_ROOT / "data" / "locomo_processed.json"),
    )
    parser.add_argument("--retrieval-path", default=str(DEFAULT_RETRIEVAL_PATH))
    parser.add_argument("--wiki-path", default=str(DEFAULT_WIKI_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--base-url", default=str(DEFAULT_LLM_BASE_URL))
    parser.add_argument("--api-key", default=None,
                        help="Override the default API key (needed for deployments that require one)")
    parser.add_argument("--model-name", default=str(DEFAULT_LLM_MODEL_NAME))
    parser.add_argument("--controller-model-name")
    parser.add_argument("--answer-model-name")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--action-max-tokens", type=int, default=4096)
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--span-token-budget", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = "locomo"
    args.controller_model_name = args.controller_model_name or args.model_name
    args.answer_model_name = args.answer_model_name or args.model_name

    data_path = Path(args.data_path).expanduser()
    retrieval_path = Path(args.retrieval_path).expanduser()
    wiki_path = Path(args.wiki_path).expanduser()
    output_path = ensure_parent(Path(args.output_path).expanduser())

    rows = ada_reading.load_jsonl(retrieval_path)
    rows = [
        row
        for row in rows
        if not ada_reading.should_skip_t2mem_row("locomo", row)
    ]
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    query_sessions = ada_reading.load_query_session_map("locomo", data_path)
    wiki_map = ada_reading.load_wiki_map_by_id(wiki_path)
    client = build_client(args.base_url, timeout=args.request_timeout, api_key=args.api_key)

    print(f"Retrieval file: {retrieval_path}")
    print(f"Summary wiki：{wiki_path}")
    print(f"Output file: {output_path}")
    print(f"Queries: {len(rows)}")

    # Swap the rendering function only for this branch, and restore it afterwards.
    original_wiki_text = ada_reading.wiki_text
    ada_reading.wiki_text = summary_wiki_text
    try:
        with output_path.open("w", encoding="utf-8") as file_obj:
            with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
                futures = [
                    executor.submit(
                        ada_reading.run_t2mem_query,
                        row,
                        dataset="locomo",
                        args=args,
                        client=client,
                        query_sessions_map=query_sessions,
                        wiki_map_by_id=wiki_map,
                    )
                    for row in rows
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="summary-t2mem",
                    unit="query",
                ):
                    file_obj.write(
                        json.dumps(future.result(), ensure_ascii=False) + "\n"
                    )
                    file_obj.flush()
    finally:
        ada_reading.wiki_text = original_wiki_text

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
