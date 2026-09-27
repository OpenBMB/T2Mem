from __future__ import annotations

"""Generate one natural-language summary per LoCoMo session as a summary-only wiki."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL_NAME, REPO_ROOT
from utils.llm_utils import build_client
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "locomo_processed.json"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "data" / "locomo" / "locomo_summary_wiki.jsonl"
)


SUMMARY_PROMPT = """Summarize the following conversation session.

Authoritative session date/time:
{date_time}

Transcript:
{transcript}

1. Use the authoritative conversation date/time as the temporal anchor, resolving relative or ambiguous time expressions (e.g., yesterday, last year, next month) into explicit dates whenever possible.
2. Preserve concrete details that may be useful for later QA, including preferences, objects, locations, plans, family details, activities, and quoted names etc.

Summary:"""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary-only LoCoMo wikis")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--model-name", default=DEFAULT_LLM_MODEL_NAME)
    parser.add_argument("--api-key", default=os.getenv("T2MEM_LLM_API_KEY", ""))
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Process only the first N sessions")
    parser.add_argument("--resume", action="store_true", help="Skip sessions that already exist")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--print-first-prompt", action="store_true")
    return parser.parse_args()


def load_tasks(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_obj:
        records = json.load(file_obj)

    tasks = [
        {"sample_id": str(record["sample_id"]), "session": session}
        for record in records
        for session in record.get("sessions", [])
    ]
    return tasks if limit is None else tasks[: max(0, limit)]


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as file_obj:
        return {
            str(row["source_session_id"])
            for line in file_obj
            if line.strip()
            for row in [json.loads(line)]
            if row.get("summary") and not row.get("error")
        }


def build_prompt(session: dict[str, Any]) -> str:
    return SUMMARY_PROMPT.format(
        date_time=session.get("date_time", ""),
        transcript=session.get("transcript_text", ""),
    )


def build_runner(args: argparse.Namespace):
    client = build_client(args.base_url, timeout=args.request_timeout, api_key=args.api_key)

    def run(task: dict[str, Any]) -> dict[str, Any]:
        session = task["session"]
        error = ""

        for attempt in range(args.max_retries):
            try:
                request = {
                    "model": args.model_name,
                    "messages": [{"role": "user", "content": build_prompt(session)}],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                }
                if args.disable_thinking:
                    request["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }

                response = client.chat.completions.create(**request)
                summary = (response.choices[0].message.content or "").strip()
                if not summary:
                    raise ValueError("model returned an empty response")

                return {
                    "sample_id": task["sample_id"],
                    "source_session_id": session["source_session_id"],
                    "wiki_id": session["wiki_id"],
                    "source_session_date_time": session.get("date_time", ""),
                    "summary": summary,
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < args.max_retries:
                    time.sleep(2**attempt)

        return {
            "sample_id": task["sample_id"],
            "source_session_id": session["source_session_id"],
            "wiki_id": session["wiki_id"],
            "source_session_date_time": session.get("date_time", ""),
            "error": error,
        }

    return run


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser()
    output_path = Path(args.output_path).expanduser()
    tasks = load_tasks(data_path, args.limit)

    if args.print_first_prompt:
        if not tasks:
            raise ValueError("no sessions to process")
        print(build_prompt(tasks[0]["session"]))
        return

    completed_ids = load_completed_ids(output_path) if args.resume else set()
    tasks = [
        task
        for task in tasks
        if task["session"]["source_session_id"] not in completed_ids
    ]

    print(f"Data: {data_path}")
    print(f"Output: {output_path}")
    print(f"Sessions to summarize: {len(tasks)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    run = build_runner(args)
    failed = 0

    with output_path.open(mode, encoding="utf-8") as file_obj:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = [executor.submit(run, task) for task in tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="summary-wiki",
                unit="session",
            ):
                row = future.result()
                failed += int("error" in row)
                file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
                file_obj.flush()

    print(f"Done: {output_path}")
    print(f"Failed sessions: {failed}")
    print(f"Data: {data_path}")
    print(f"Output: {output_path}")
    print(f"Model: {args.model_name}")
    print(f"Endpoint: {args.base_url}")
    print(f"Sessions to summarize: {len(tasks)}")

if __name__ == "__main__":
    main()
