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

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.config import REPO_ROOT, ensure_parent

# ── LongMemEval judge prompts ────────────────────────────
ANSCHECK_PROMPT_SESSION = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only."""

ANSCHECK_PROMPT_TEMPORAL = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only."""

ANSCHECK_PROMPT_KUPDATE = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only."""

ANSCHECK_PROMPT_PREFERENCE = """I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {question}

Rubric: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only."""

ANSCHECK_PROMPT_ABSTAIN = """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {answer}

Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only."""

DEFAULT_DATA_PATH = REPO_ROOT / "data" / "longmemeval_processed.json"
DEFAULT_BASE_URL = os.getenv("JUDGE_BASE_URL", "")
DEFAULT_MODEL_NAME = "qwen3.5-plus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LongMemEval predictions with an LLM judge.")
    parser.add_argument("--input-path", required=True, help="Path to prediction JSONL file.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-path", default=None, help="Path to per-row judge JSONL output.")
    parser.add_argument("--summary-path", default=None, help="Path to summary JSON output.")
    parser.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def load_gold_map(data_path: Path) -> dict[str, dict[str, str]]:
    rows = load_json(data_path)
    gold_map: dict[str, dict[str, str]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        gold_map[question_id] = {
            "question_id": question_id,
            "question_type": str(row.get("question_type", "")),
            "question": str(row.get("question", "")),
            "question_date": str(row.get("question_date", "")),
            "answer": str(row.get("answer", "")),
        }
    return gold_map


def normalize_string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_yes_no(text: str) -> str:
    match = re.search(r"\b(yes|no)\b", normalize_string(text).lower())
    return match.group(1) if match else ""


def is_abstention_row(row: dict[str, Any]) -> bool:
    question_id = normalize_string(row.get("question_id")).lower()
    question_type = normalize_string(row.get("question_type")).lower()
    return ("_abs" in question_id) or (question_type in {"abstention", "unanswerable"})


def select_prompt(row: dict[str, Any]) -> tuple[str, str]:
    if is_abstention_row(row):
        return "abstention", ANSCHECK_PROMPT_ABSTAIN

    mapping = {
        "temporal-reasoning": ("temporal-reasoning", ANSCHECK_PROMPT_TEMPORAL),
        "knowledge-update": ("knowledge-update", ANSCHECK_PROMPT_KUPDATE),
        "single-session-preference": ("single-session-preference", ANSCHECK_PROMPT_PREFERENCE),
    }
    return mapping.get(normalize_string(row.get("question_type")), ("session", ANSCHECK_PROMPT_SESSION))


def merge_with_gold(row: dict[str, Any], gold_map: dict[str, dict[str, str]]) -> dict[str, Any]:
    gold = gold_map.get(normalize_string(row.get("question_id")), {})
    merged = dict(row)
    for field in ("question_type", "question", "question_date", "answer"):
        if not normalize_string(merged.get(field)):
            merged[field] = gold.get(field, "")
    return merged


def build_judge_prompt(row: dict[str, Any], prediction_field: str) -> tuple[str, str]:
    prompt_name, prompt_template = select_prompt(row)
    prediction = normalize_string(row.get(prediction_field))
    if not prediction:
        raise ValueError(f"Prediction field `{prediction_field}` is missing or empty.")

    user_prompt = prompt_template.format(
        question=normalize_string(row.get("question")),
        answer=normalize_string(row.get("answer")),
        response=prediction,
    )
    return prompt_name, user_prompt


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


def build_client(args: argparse.Namespace) -> OpenAI:
    if not args.api_key:
        raise ValueError("A judge API key is required. Set --api-key or JUDGE_API_KEY.")
    return OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
    )


def call_judge_with_retries(
    *,
    client: OpenAI,
    model_name: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> str:
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"enable_thinking": False},
            )
            content = response.choices[0].message.content
            return "" if content is None else str(content)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(1.0)
    raise RuntimeError(last_error or "Judge model call failed without error details.")


def judge_one(
    row: dict[str, Any],
    *,
    gold_map: dict[str, dict[str, str]],
    prediction_field: str,
    client: OpenAI,
    model_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    merged_row = merge_with_gold(row, gold_map)
    result: dict[str, Any] = {
        "dataset": normalize_string(merged_row.get("dataset")),
        "question_id": normalize_string(merged_row.get("question_id")),
        "question_type": normalize_string(merged_row.get("question_type")),
        "question_date": normalize_string(merged_row.get("question_date")),
        "question": normalize_string(merged_row.get("question")),
        "answer": normalize_string(merged_row.get("answer")),
        prediction_field: normalize_string(merged_row.get(prediction_field)),
    }

    try:
        prompt_name, user_prompt = build_judge_prompt(merged_row, prediction_field)
        raw_judge = call_judge_with_retries(
            client=client,
            model_name=model_name,
            user_prompt=user_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )
        judge_label = normalize_yes_no(raw_judge)
        result.update(
            {
                "judge_model_name": model_name,
                "judge_prompt_type": prompt_name,
                "judge_raw_response": raw_judge,
                "judge_label": judge_label,
                "is_correct": judge_label == "yes",
                "judge_invalid_response": not bool(judge_label),
            }
        )
        return result
    except Exception as exc:
        result.update(
            {
                "judge_model_name": model_name,
                "judge_label": "",
                "is_correct": False,
                "judge_invalid_response": False,
                "judge_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return result


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_rows": len(rows),
        "judged_rows": 0,
        "correct_rows": 0,
        "error_rows": 0,
        "invalid_judge_rows": 0,
        "accuracy": 0.0,
        "by_question_type": {},
    }

    by_question_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total_rows": 0,
            "judged_rows": 0,
            "correct_rows": 0,
            "error_rows": 0,
            "invalid_judge_rows": 0,
        }
    )

    for row in rows:
        question_type = normalize_string(row.get("question_type")) or "unknown"
        bucket = by_question_type[question_type]
        bucket["total_rows"] += 1
        summary["judged_rows"] += 1
        bucket["judged_rows"] += 1

        if row.get("is_correct"):
            summary["correct_rows"] += 1
            bucket["correct_rows"] += 1
        if row.get("judge_error"):
            summary["error_rows"] += 1
            bucket["error_rows"] += 1
        if row.get("judge_invalid_response"):
            summary["invalid_judge_rows"] += 1
            bucket["invalid_judge_rows"] += 1

    summary["accuracy"] = (
        summary["correct_rows"] / summary["judged_rows"] if summary["judged_rows"] else 0.0
    )

    for question_type, bucket in sorted(by_question_type.items()):
        judged_rows = bucket["judged_rows"] or 1
        summary["by_question_type"][question_type] = {
            **bucket,
            "accuracy": bucket["correct_rows"] / judged_rows,
        }

    return summary


def resolve_output_path(input_path: Path, output_path: str | None) -> Path:
    if output_path:
        return ensure_parent(Path(output_path).expanduser())
    return ensure_parent(input_path.with_name(f"{input_path.stem}_llm_judge_ali.jsonl"))


def resolve_summary_path(output_path: Path, summary_path: str | None) -> Path:
    if summary_path:
        return ensure_parent(Path(summary_path).expanduser())
    return ensure_parent(output_path.with_suffix(".summary.json"))


def main() -> None:
    args = parse_args()
    require_api_config(args)
    input_path = Path(args.input_path).expanduser()
    data_path = Path(args.data_path).expanduser()
    output_path = resolve_output_path(input_path, args.output_path)
    summary_path = resolve_summary_path(output_path, args.summary_path)

    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    client = build_client(args)
    gold_map = load_gold_map(data_path)

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Summary file: {summary_path}")
    print(f"Judge model: {args.model_name}")
    print(f"Rows to judge: {len(rows)}")

    judged_rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as file_obj:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = [
                executor.submit(
                    judge_one,
                    row,
                    gold_map=gold_map,
                    prediction_field=args.prediction_field,
                    client=client,
                    model_name=args.model_name,
                    args=args,
                )
                for row in rows
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="llm-judge", unit="row"):
                result = future.result()
                judged_rows.append(result)
                file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")

    judged_rows.sort(key=lambda row: normalize_string(row.get("question_id")))
    summary = summarize_rows(judged_rows)
    summary.update(
        {
            "dataset": "longmemeval",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "gold_path": str(data_path),
            "base_url": args.base_url,
            "judge_model_name": args.model_name,
            "prediction_field": args.prediction_field,
        }
    )

    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Correct rows: {summary['correct_rows']} / {summary['judged_rows']}")
    print(f"Invalid judge rows: {summary['invalid_judge_rows']}")
    print(f"Error rows: {summary['error_rows']}")


if __name__ == "__main__":
    main()
