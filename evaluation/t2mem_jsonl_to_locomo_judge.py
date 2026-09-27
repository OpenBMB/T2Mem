#!/usr/bin/env python3
from __future__ import annotations
"""Convert T2Mem JSONL output into the dict-shaped input the LoCoMo judge expects.

``locomo_judge.py`` reads a JSON object keyed by conversation id, each value
a list of QA items::

    {"conv-26": [{"category": 1, "question": "...", "answer": "...", "response": "..."}],
     "conv-41": [...]}

T2Mem writes one JSON object per line, keyed by ``sample_id`` / ``query_id``.
This script bridges the two.

Usage::

    python evaluation/t2mem_jsonl_to_locomo_judge.py \
        --input-path  outputs/locomo_answers.jsonl \
        --output-path outputs/locomo_for_judge.json
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", required=True, help="T2Mem JSONL output.")
    parser.add_argument("--output-path", required=True, help="Destination dict-shaped JSON.")
    parser.add_argument("--prediction-field", default="prediction", help="Field holding the model answer.")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Drop rows whose 'error' field is non-empty (failed runs have no prediction).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input_path).expanduser()
    dst = Path(args.output_path).expanduser()

    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    n_rows = n_skipped = 0

    with src.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            if args.skip_errors and row.get("error"):
                n_skipped += 1
                continue

            conv_id = str(row.get("sample_id") or row.get("conv_id") or "")
            if not conv_id:
                conv_id = str(row.get("query_id", "")).rsplit("_q", 1)[0]
            if not conv_id:
                raise ValueError(f"Cannot derive a conversation id from row: {row.get('query_id')!r}")

            grouped.setdefault(conv_id, []).append(
                {
                    "category": row.get("category", -1),
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "response": row.get(args.prediction_field) or row.get("prediction") or "",
                }
            )

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as handle:
        json.dump(grouped, handle, ensure_ascii=False, indent=1)

    total_items = sum(len(v) for v in grouped.values())
    print(f"conversations : {len(grouped)}")
    print(f"QA items      : {total_items}  (from {n_rows} rows, {n_skipped} skipped)")
    print(f"written       : {dst}")


if __name__ == "__main__":
    main()
