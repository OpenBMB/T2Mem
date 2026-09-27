from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prompt import t2mem_prompt
from utils.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL_NAME, REPO_ROOT, ensure_parent
from utils.llm_utils import (
    DEFAULT_DISABLE_THINKING_EXTRA_BODY,
    build_client,
    call_json_llm,
)

# prompts
SESSION_WIKI_SYSTEM_PROMPT = t2mem_prompt.SESSION_WIKI_SYSTEM_PROMPT
SESSION_WIKI_USER_PROMPT_TEMPLATE = t2mem_prompt.SESSION_WIKI_USER_PROMPT_TEMPLATE

DATASETS = {
    "longmemeval": {
        "default_data_path": Path(os.getenv(
            "T2MEM_LME_DATA", str(REPO_ROOT / "data" / "longmemeval_processed.json"))).expanduser(),
        "default_output_path": REPO_ROOT / "data" / "longmemeval" / "longmemeval_wiki.jsonl",
    },
}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _items(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize(fields: Any) -> Dict[str, Any]:
    if not isinstance(fields, dict):
        fields = {}
    return {
        "summary": _text(fields.get("summary")),
        "event_core": _items(fields.get("event_core")),
        "atomic_facts": _items(fields.get("atomic_facts")),
        "entities": list(dict.fromkeys(_items(fields.get("entities")))),
        "temporal_clues": _items(fields.get("temporal_clues")),
        "resolved_temporal_facts": _items(fields.get("resolved_temporal_facts")),
        "ongoing_threads": _items(fields.get("ongoing_threads")),
    }


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
        for session in record.get("sessions")
    ]


def _success_row(task: Dict[str, Any], wiki: Dict[str, Any]) -> Dict[str, Any]:
    session = task["session"]
    return {
        **{key: value for key, value in task.items() if key != "session"},
        "source_session_id": session["source_session_id"],
        "wiki_id": session["wiki_id"],
        "source_session_date_time": session.get("date_time"),
        **wiki,
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


def _build_openai_runner(args: argparse.Namespace) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    client = build_client(args.base_url, timeout=args.request_timeout, api_key=args.api_key)

    def run(task: Dict[str, Any]) -> Dict[str, Any]:
        session = task["session"]
        last_error = ""

        for attempt in range(1, args.max_retries + 1):
            try:
                wiki = _normalize(
                    call_json_llm(
                        client=client,
                        model_name=args.model_name,
                        system_prompt=SESSION_WIKI_SYSTEM_PROMPT,
                        user_prompt=SESSION_WIKI_USER_PROMPT_TEMPLATE.format(
                            date_time=_text(session.get("date_time")),
                            transcript=session["transcript_text"],
                        ),
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        extra_body=DEFAULT_DISABLE_THINKING_EXTRA_BODY,
                    )
                )
                return _success_row(task, wiki)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    time.sleep(1.0)

        return _error_row(task, last_error)

    return run


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wiki JSONL from processed session data.")
    parser.add_argument("--dataset", choices=("longmemeval",), default="longmemeval")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--api-key", default=None,
                        help="Override the default API key (needed for deployments that require one)")
    parser.add_argument("--model-name", default=DEFAULT_LLM_MODEL_NAME)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    spec = DATASETS[args.dataset]
    data_path = Path(args.data_path).expanduser() if args.data_path else spec["default_data_path"]
    output_path = ensure_parent(Path(args.output_path).expanduser() if args.output_path else spec["default_output_path"])
    records = _load_records(data_path)
    total_sessions = sum(len(record.get("sessions", [])) for record in records)

    print(f"Loaded {len(records)} {args.dataset} records from {data_path}")
    print(f"Total sessions: {total_sessions}")
    print(f"Using base_url: {args.base_url}")
    print(f"Using model_name: {args.model_name}")

    run = _build_openai_runner(args)
    failed_sessions = 0
    saved_sessions = 0

    with output_path.open("w", encoding="utf-8") as file_obj:
        with tqdm(total=len(records), desc="records", unit="record") as record_progress:
            with tqdm(total=total_sessions, desc="sessions", unit="session") as session_progress:
                if args.max_workers == 1:
                    for record in records:
                        for task in _record_tasks(args.dataset, record):
                            result = run(task)
                            if "error" in result:
                                failed_sessions += 1

                            file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")
                            saved_sessions += 1

                            if saved_sessions % 100 == 0:
                                file_obj.flush()

                            session_progress.update(1)

                        record_progress.update(1)
                else:
                    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                        for record in records:
                            futures = [executor.submit(run, task) for task in _record_tasks(args.dataset, record)]
                            for future in as_completed(futures):
                                result = future.result()
                                if "error" in result:
                                    failed_sessions += 1

                                file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")
                                saved_sessions += 1

                                if saved_sessions % 100 == 0:
                                    file_obj.flush()

                                session_progress.update(1)

                            record_progress.update(1)

    print(output_path)
    print(
        f"total records read: {len(records)}, "
        f"total sessions saved: {saved_sessions}, "
        f"failed sessions: {failed_sessions}"
    )


if __name__ == "__main__":
    main()
