from __future__ import annotations
"""Turn the raw dataset into the model input format."""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.config import REPO_ROOT, ensure_parent
from utils.read_longmemeval import iter_session_records as iter_longmemeval_records
from utils.read_longmemeval import load_longmemeval_samples


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("longmemeval",), default="longmemeval")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--question-types", default=None)
    return parser.parse_args(argv)


def _default_data_path(dataset: str) -> Path:
    return Path(os.getenv("T2MEM_LONGMEMEVAL_RAW",
                                 str(REPO_ROOT / "data" / "raw" / "longmemeval_s_cleaned.json"))).expanduser()


def _default_output_path(dataset: str) -> Path:
    return Path(os.getenv("T2MEM_LME_DATA",
                                 str(REPO_ROOT / "data" / "longmemeval_processed.json"))).expanduser()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    data_path = Path(args.data_path).expanduser() if args.data_path else _default_data_path(args.dataset)
    output_path = ensure_parent(
        Path(args.output_path).expanduser() if args.output_path else _default_output_path(args.dataset)
    )

    rows = iter_longmemeval_records(
        load_longmemeval_samples(data_path),
        question_types=args.question_types,
    )
    row_id_key = "question_id"
    row_name = "query"

    count = 0
    with output_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("[\n")
        for row in rows:
            if count > 0:
                file_obj.write(",\n")
            file_obj.write(json.dumps(row, ensure_ascii=False, indent=2))
            file_obj.flush()
            count += 1
            print(f"saved {row_name} {count}: {row[row_id_key]}")
        file_obj.write("\n]\n")
        file_obj.flush()

    print(output_path)
    print(f"total queries saved: {count}")

if __name__ == "__main__":
    main()
