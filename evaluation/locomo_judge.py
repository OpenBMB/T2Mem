"""LoCoMo LLM judge: score predictions with an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

DEFAULT_BASE_URL = os.getenv("JUDGE_BASE_URL", "")
DEFAULT_MODEL_NAME = "qwen3.5-plus"

# ── LoCoMo judge prompt ─────────────────────────────────
ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoCoMo predictions with an LLM judge.")
    parser.add_argument("--input-path", required=True, help="Path to prediction JSON file.")
    parser.add_argument("--output-path", default=None, help="Path to judge JSONL output.")
    parser.add_argument("--summary-path", default=None, help="Path to summary JSON output.")
    parser.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def normalize_string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load LoCoMo predictions dict -> flat list of judge-ready rows."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for conv_id, items in data.items():
        for idx, item in enumerate(items):
            rows.append({
                "conv_id": conv_id,
                "item_idx": idx,
                "category": int(item.get("category", -1)),
                "question": normalize_string(item.get("question")),
                "answer": normalize_string(item.get("answer")),
                "generated_answer": normalize_string(item.get("response") or item.get("prediction") or ""),
            })
    return rows


def judge_one(
    row: dict[str, Any],
    client: OpenAI,
    model_name: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            prompt = ACCURACY_PROMPT.format(
                question=row["question"],
                gold_answer=row["answer"],
                generated_answer=row["generated_answer"],
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            content = resp.choices[0].message.content or ""
            label = json.loads(extract_json(content))["label"]
            return {
                **row,
                "judge_raw": content,
                "judge_label": label.lower(),
                "is_correct": 1 if label.upper() == "CORRECT" else 0,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - retry on any judge failure
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(1.0)

    return {**row, "judge_raw": "", "judge_label": "error", "is_correct": 0, "error": last_error}


def resolve_output_path(input_path: Path, output_path: str | None) -> Path:
    if output_path:
        p = Path(output_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p = input_path.with_name(f"{input_path.stem}_llm_judge_ali.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def resolve_summary_path(output_path: Path, summary_path: str | None) -> Path:
    if summary_path:
        p = Path(summary_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p = output_path.with_suffix(".summary.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def require_api_config(args: argparse.Namespace) -> None:
    """Fail early with a clear message when judge credentials are missing."""
    missing = [
        name
        for name, value in (("--api-key / JUDGE_API_KEY", args.api_key),
                            ("--base-url / JUDGE_BASE_URL", args.base_url))
        if not value
    ]
    if missing:
        raise SystemExit("Missing judge configuration: " + ", ".join(missing))


def main() -> None:
    args = parse_args()
    require_api_config(args)
    input_path = Path(args.input_path).expanduser()
    output_path = resolve_output_path(input_path, args.output_path)
    summary_path = resolve_summary_path(output_path, args.summary_path)

    rows = load_predictions(input_path)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    # Filter category 5
    rows = [r for r in rows if r["category"] != 5]

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Summary file: {summary_path}")
    print(f"Judge model: {args.model_name}")
    print(f"Rows to judge: {len(rows)}")

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
    )

    judged: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as file_obj:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futures = [
                ex.submit(judge_one, r, client, args.model_name, args.max_tokens, args.temperature, args.max_retries)
                for r in rows
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="llm-judge", unit="row"):
                result = future.result()
                judged.append(result)
                file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Aggregate
    by_cat = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
    for r in judged:
        cat = r["category"]
        by_cat[cat]["total"] += 1
        if r.get("error"):
            by_cat[cat]["errors"] += 1
        if r.get("is_correct"):
            by_cat[cat]["correct"] += 1

    total = sum(v["total"] for v in by_cat.values())
    correct = sum(v["correct"] for v in by_cat.values())
    errors = sum(v["errors"] for v in by_cat.values())

    summary = {
        "judge_model": args.model_name,
        "total_rows": total,
        "judged_rows": total - errors,
        "correct_rows": correct,
        "error_rows": errors,
        "accuracy": correct / (total - errors) if (total - errors) > 0 else 0,
        "by_category": {},
    }

    print(f"\n{'='*60}")
    print("LoCoMo LLM Judge Results")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {summary['accuracy']:.4f} ({correct}/{total - errors})")
    print()

    for cat in sorted(by_cat.keys()):
        v = by_cat[cat]
        acc = v["correct"] / v["total"] if v["total"] else 0
        summary["by_category"][str(cat)] = {
            "total": v["total"],
            "correct": v["correct"],
            "errors": v["errors"],
            "accuracy": acc,
        }
        print(f"  Category {cat}: {acc:.4f} ({v['correct']}/{v['total']})")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
