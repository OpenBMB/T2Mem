from __future__ import annotations

"""Build summary-only wiki JSONL from processed LongMemEval session data."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL_NAME, REPO_ROOT, ensure_parent
from utils.llm_utils import build_client


DEFAULT_DISABLE_THINKING_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
DEFAULT_LLM_API_KEY = os.getenv("T2MEM_LLM_API_KEY", "") or "EMPTY"


DATASETS = {
    "longmemeval": {
        "default_data_path": REPO_ROOT / "data" / "longmemeval_processed.json",
        "default_output_path": REPO_ROOT
        / "data" / "longmemeval"
        / "longmemeval_summary_wiki.jsonl",
    },
}

SUMMARY_PROMPT = """Summarize the following conversation session.

Authoritative session date/time:
{date_time}

Conversation:
{transcript}

Requirements:
1. Use only information from the conversation, do not invent facts.
2. Preserve details that may help future question answering.
3. Resolve relative time expressions such as "yesterday", "last year", and "next month" using the authoritative session date/time when possible.
4. Keep the summary concise.
Summary: """


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _task_meta(dataset: str, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question_id": record["question_id"],
        "question_type": record.get("question_type"),
    }


def _record_tasks(dataset: str, record: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = _task_meta(dataset, record)
    return [
        {
            **meta,
            "session": session,
        }
        for session in record.get("sessions", [])
    ]


def _task_key(task: Dict[str, Any]) -> Tuple[str, str]:
    return _text(task.get("question_id")), _text(task["session"].get("wiki_id"))


def _success_row(task: Dict[str, Any], summary: str) -> Dict[str, Any]:
    session = task["session"]
    return {
        **{key: value for key, value in task.items() if key != "session"},
        "source_session_id": session["source_session_id"],
        "wiki_id": session["wiki_id"],
        "source_session_date_time": session.get("date_time"),
        "summary": summary,
    }


def _error_row(task: Dict[str, Any], message: str) -> Dict[str, Any]:
    session = task["session"]
    return {
        **{key: value for key, value in task.items() if key != "session"},
        "source_session_id": session["source_session_id"],
        "wiki_id": session["wiki_id"],
        "source_session_date_time": session.get("date_time"),
        "error": message or "No error details returned.",
    }


def _build_prompt(session: Dict[str, Any]) -> str:
    return SUMMARY_PROMPT.format(
        date_time=_text(session.get("date_time")),
        transcript=_text(session.get("transcript_text")),
    )


def _load_completed_keys(path: Path) -> set[Tuple[str, str]]:
    if not path.exists():
        return set()

    completed_keys: set[Tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("summary") and not row.get("error"):
                completed_keys.add((_text(row.get("question_id")), _text(row.get("wiki_id"))))
    return completed_keys


def _build_openai_runner(args: argparse.Namespace) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    client = build_client(args.base_url, api_key=args.api_key, timeout=args.request_timeout)
    extra_body = DEFAULT_DISABLE_THINKING_EXTRA_BODY if args.disable_thinking else None

    def run(task: Dict[str, Any]) -> Dict[str, Any]:
        session = task["session"]
        last_error = ""

        for attempt in range(1, args.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=[{"role": "user", "content": _build_prompt(session)}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    extra_body=extra_body,
                )
                summary = _text(response.choices[0].message.content)
                if not summary:
                    raise ValueError("Model returned an empty summary.")
                return _success_row(task, summary)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    time.sleep(1.0)

        return _error_row(task, last_error)

    return run


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary-only wiki JSONL from processed session data.")
    parser.add_argument("--dataset", choices=("longmemeval",), default="longmemeval")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--model-name", default=DEFAULT_LLM_MODEL_NAME)
    parser.add_argument("--api-key", default=DEFAULT_LLM_API_KEY)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sessions.")
    parser.add_argument("--resume", action="store_true", help="Skip completed question_id/wiki_id rows.")
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    parser.add_argument("--enable-thinking", action="store_false", dest="disable_thinking")
    parser.add_argument("--print-first-prompt", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    spec = DATASETS[args.dataset]
    data_path = Path(args.data_path).expanduser() if args.data_path else spec["default_data_path"]
    output_path = ensure_parent(Path(args.output_path).expanduser() if args.output_path else spec["default_output_path"])
    records = _load_records(data_path)
    tasks = [
        task
        for record in records
        for task in _record_tasks(args.dataset, record)
    ]
    if args.limit is not None:
        tasks = tasks[: max(0, args.limit)]

    if args.print_first_prompt:
        if not tasks:
            raise ValueError("No sessions are available.")
        print(_build_prompt(tasks[0]["session"]))
        return

    completed_keys = _load_completed_keys(output_path) if args.resume else set()
    if completed_keys:
        tasks = [task for task in tasks if _task_key(task) not in completed_keys]

    total_sessions = sum(len(record.get("sessions", [])) for record in records)
    failed_sessions = 0
    saved_sessions = 0

    print(f"Loaded {len(records)} {args.dataset} records from {data_path}")
    print(f"Total sessions: {total_sessions}")
    print(f"Summary sessions to build: {len(tasks)}")
    print(f"Using base_url: {args.base_url}")
    print(f"Using model_name: {args.model_name}")

    run = _build_openai_runner(args)
    mode = "a" if args.resume else "w"

    with output_path.open(mode, encoding="utf-8") as file_obj:
        if args.max_workers == 1:
            for task in tqdm(tasks, desc="summary-wiki", unit="session"):
                result = run(task)
                if "error" in result:
                    failed_sessions += 1

                file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")
                saved_sessions += 1

                if saved_sessions % 100 == 0:
                    file_obj.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(run, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="summary-wiki", unit="session"):
                    result = future.result()
                    if "error" in result:
                        failed_sessions += 1

                    file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")
                    saved_sessions += 1

                    if saved_sessions % 100 == 0:
                        file_obj.flush()

        file_obj.flush()

    print(output_path)
    print(
        f"total records read: {len(records)}, "
        f"total sessions saved: {saved_sessions}, "
        f"failed sessions: {failed_sessions}"
    )


if __name__ == "__main__":
    main()
