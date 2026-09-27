from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence

from json_repair import repair_json
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.config import ensure_parent
from utils.llm_utils import DEFAULT_DISABLE_THINKING_EXTRA_BODY, build_client, call_llm
from wiki import build_wiki


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Regenerate wiki rows that contain error fields and merge them back.")
    parser.add_argument("--dataset", choices=("locomo",), default="locomo")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--wiki-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--base-url", default=build_wiki.DEFAULT_LLM_BASE_URL)
    parser.add_argument("--model-name", default=build_wiki.DEFAULT_LLM_MODEL_NAME)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.05)
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""
    with path.open("r", encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def row_key(dataset: str, row: dict[str, Any]) -> tuple[str, str]:
    """Extract the unique key from an output row to locate its source session."""
    return str(row["sample_id"]), str(row["source_session_id"])


def task_key(dataset: str, task: dict[str, Any]) -> tuple[str, str]:
    """Extract the same unique key from a build_wiki task."""
    session = task["session"]
    return str(task["sample_id"]), str(session["source_session_id"])


def build_task_map(dataset: str, data_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Expand the processed dataset into key -> task so failed rows can be retried precisely."""
    records = build_wiki._load_records(data_path)
    tasks: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        for task in build_wiki._record_tasks(dataset, record):
            tasks[task_key(dataset, task)] = task
    return tasks


def default_output_path(wiki_path: Path) -> Path:
    """Default to *_repaired.jsonl so the original file is never overwritten."""
    return wiki_path.with_name(f"{wiki_path.stem}_repaired{wiki_path.suffix}")


def _build_repair_runner(args: argparse.Namespace):
    """Build a wiki generator used only by the repair script."""
    client = build_client(args.base_url, timeout=args.request_timeout)

    def run(task: dict[str, Any]) -> dict[str, Any]:
        session = task["session"]
        last_error = ""

        for _attempt in range(1, args.max_retries + 1):
            try:
                raw_text = call_llm(
                    client=client,
                    model_name=args.model_name,
                    system_prompt=build_wiki.SESSION_WIKI_SYSTEM_PROMPT,
                    user_prompt=build_wiki.SESSION_WIKI_USER_PROMPT_TEMPLATE.format(
                        date_time=build_wiki._text(session.get("date_time")),
                        transcript=session["transcript_text"],
                    ),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    extra_body=DEFAULT_DISABLE_THINKING_EXTRA_BODY,
                )
                repaired = repair_json(raw_text, return_objects=True)
                if not isinstance(repaired, dict):
                    raise TypeError(f"Repaired JSON is not an object: {type(repaired).__name__}")
                wiki = build_wiki._normalize(repaired)
                return build_wiki._success_row(task, wiki)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        return build_wiki._error_row(task, last_error)

    return run


def repair_rows(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    tasks: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Retry only the rows carrying an error, then splice results back in order."""
    repaired_rows = list(rows)
    error_entries = [(index, row) for index, row in enumerate(rows) if row.get("error")]
    stats = {
        "total_rows": len(rows),
        "error_rows_before": len(error_entries),
        "repaired_success": 0,
        "repaired_still_error": 0,
        "missing_source_task": 0,
    }

    if not error_entries:
        return repaired_rows, stats

    run = _build_repair_runner(args)
    jobs: list[tuple[int, dict[str, Any]]] = []

    for index, row in error_entries:
        task = tasks.get(row_key(args.dataset, row))
        if task is None:
            stats["missing_source_task"] += 1
            continue
        jobs.append((index, task))

    if args.max_workers == 1:
        iterator = tqdm(jobs, desc="repair_wiki", unit="row")
        for index, task in iterator:
            result = run(task)
            repaired_rows[index] = result
            if result.get("error"):
                stats["repaired_still_error"] += 1
            else:
                stats["repaired_success"] += 1
        return repaired_rows, stats

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(run, task): index for index, task in jobs}
        with tqdm(total=len(future_map), desc="repair_wiki", unit="row") as progress:
            for future in as_completed(future_map):
                index = future_map[future]
                result = future.result()
                repaired_rows[index] = result
                if result.get("error"):
                    stats["repaired_still_error"] += 1
                else:
                    stats["repaired_success"] += 1
                progress.update(1)

    return repaired_rows, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the JSONL back in its original order."""
    with ensure_parent(path).open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    spec = build_wiki.DATASETS[args.dataset]
    data_path = Path(args.data_path).expanduser() if args.data_path else spec["default_data_path"]
    wiki_path = Path(args.wiki_path).expanduser() if args.wiki_path else spec["default_output_path"]
    output_path = Path(args.output_path).expanduser() if args.output_path else default_output_path(wiki_path)

    print(f"Loading wiki rows from {wiki_path}")
    rows = load_jsonl(wiki_path)
    print(f"Building source task map from {data_path}")
    tasks = build_task_map(args.dataset, data_path)

    repaired_rows, stats = repair_rows(args, rows, tasks)
    write_jsonl(output_path, repaired_rows)

    error_rows_after = sum(1 for row in repaired_rows if row.get("error"))
    stats["error_rows_after"] = error_rows_after

    print(output_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
