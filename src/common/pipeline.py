"""Configuration, preflight checks, and orchestration for the benchmark CLIs.

The benchmark modules own the research logic. This module only resolves inputs,
launches those modules, and verifies their artifacts before advancing a stage.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("preprocess", "index", "retrieve", "infer", "evaluate")
RAW_NAMES = {
    "locomo": ("locomo10.json",),
    "longmemeval": (
        "longmemeval_s_cleaned.json",
        "longmemeval_m_cleaned.json",
        "longmemeval_oracle.json",
    ),
}


class PipelineError(Exception):
    """An actionable configuration, input, or stage failure."""


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def positive_int(value: str) -> int:
    parsed = nonnegative_int(value)
    if not parsed:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=tuple(RAW_NAMES),
        help="omit to discover and run datasets in data/",
    )
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--env-file", type=Path, help="defaults to the repository .env")
    parser.add_argument("--data-dir", type=Path, help="raw and derived data directory")
    parser.add_argument(
        "--data-path", type=Path, help="explicit raw dataset; requires a dataset name"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="prediction directory (default: outputs/)"
    )
    parser.add_argument(
        "--eval-dir", type=Path, help="judge output directory (default: eval-results/)"
    )
    parser.add_argument(
        "--predictions", type=Path, help="prediction file for --stage evaluate"
    )
    parser.add_argument("--variant", choices=("structured", "summary"))
    parser.add_argument(
        "--read-mode", choices=("full", "wiki-only", "raw-only"), default="full"
    )
    parser.add_argument("--embed-backend", choices=("api", "local"))
    parser.add_argument("--retrieval-top-k", type=positive_int)
    parser.add_argument("--top-k", type=positive_int, help="initial reader candidates")
    parser.add_argument("--workers", type=positive_int)
    parser.add_argument("--max-steps", type=positive_int)
    parser.add_argument(
        "--temperature",
        type=float,
        help="generation temperature for indexing, inference, and evaluation",
    )
    parser.add_argument(
        "--limit", type=positive_int, help="inference/evaluation query limit"
    )
    parser.add_argument("--output-budget-tokens", type=nonnegative_int)
    parser.add_argument("--reasoning-budget", type=nonnegative_int)
    parser.add_argument("--dump-prompts", action="store_true")
    parser.add_argument("--answer-max-tokens", type=positive_int)
    parser.add_argument("--controller-model-name")
    parser.add_argument("--answer-model-name")
    parser.add_argument(
        "--skip-eval", action="store_true", help="run all stages through inference"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate configuration and inputs without model calls",
    )
    args = parser.parse_args(argv)
    if args.temperature is not None and (
        not math.isfinite(args.temperature) or args.temperature < 0
    ):
        parser.error("--temperature must be a nonnegative finite number")
    if args.data_path and not args.dataset:
        parser.error("--data-path requires locomo or longmemeval")
    if args.predictions and (args.stage != "evaluate" or not args.dataset):
        parser.error("--predictions requires a dataset name and --stage evaluate")
    if args.skip_eval and args.stage != "all":
        parser.error("--skip-eval is only used with --stage all")
    return args


def load_config(path: Path | None) -> dict[str, str]:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise PipelineError(
            "Install dependencies first: python -m pip install -r requirements.txt"
        ) from exc
    env_path = path.resolve() if path else ROOT / ".env"
    if path and not env_path.is_file():
        raise PipelineError(f"Configuration file not found: {env_path}")
    # Parse a specific file without executing shell code; process environment wins.
    load_dotenv(env_path, override=False)
    return dict(os.environ)


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def setting(env: dict[str, str], *keys: str, default: str = "") -> str:
    return next((env[key] for key in keys if env.get(key)), default)


def integer_setting(
    value: int | None, env: dict[str, str], *keys: str, default: int
) -> int:
    try:
        return (
            value
            if value is not None
            else positive_int(setting(env, *keys, default=str(default)))
        )
    except argparse.ArgumentTypeError as exc:
        raise PipelineError(f"{keys[0]} {exc}") from exc


def raw_candidates(data_dir: Path, dataset: str) -> list[Path]:
    return [
        data_dir / subdir / name
        for name in RAW_NAMES[dataset]
        for subdir in ("", "raw")
        if (data_dir / subdir / name).is_file()
    ]


def read_records(path: Path, dataset: str, *, raw: bool = False) -> list[dict]:
    if not path.is_file():
        raise PipelineError(f"Dataset file not found: {path}")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise PipelineError(f"Invalid JSON dataset: {path}") from exc
    if not isinstance(rows, list) or not rows:
        raise PipelineError(f"Dataset must be a nonempty JSON list: {path}")
    fields = {
        "locomo": ("sample_id", "conversation", "qa")
        if raw
        else ("sample_id", "sessions", "qa"),
        "longmemeval": (
            "question_id",
            "question_type",
            "question",
            "question_date",
            "answer",
            "haystack_session_ids",
            "haystack_dates",
            "haystack_sessions",
        )
        if raw
        else ("question_id", "sessions", "question", "answer"),
    }
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or any(key not in row for key in fields[dataset]):
            raise PipelineError(
                f"Invalid {dataset} record {index} in {path}; required fields: {fields[dataset]}"
            )
        if raw and dataset == "longmemeval":
            haystacks = [
                row[key]
                for key in (
                    "haystack_session_ids",
                    "haystack_dates",
                    "haystack_sessions",
                )
            ]
            if (
                any(not isinstance(values, list) for values in haystacks)
                or len({len(values) for values in haystacks}) != 1
            ):
                raise PipelineError(
                    f"Unaligned LongMemEval haystack fields in record {index}: {path}"
                )
        if dataset == "locomo":
            if not isinstance(row["qa"], list) or any(
                not isinstance(query, dict) or "question" not in query
                for query in row["qa"]
            ):
                raise PipelineError(
                    f"LoCoMo qa must contain question objects in record {index}: {path}"
                )
            if raw and not isinstance(row["conversation"], dict):
                raise PipelineError(
                    f"LoCoMo conversation must be an object in record {index}: {path}"
                )
        if not raw and (not isinstance(row["sessions"], list) or not row["sessions"]):
            raise PipelineError(f"No sessions in record {index}: {path}")
    id_key = "sample_id" if dataset == "locomo" else "question_id"
    identifiers = [str(row[id_key]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise PipelineError(f"Duplicate {id_key} values in {path}")
    return rows


def inspect_jsonl(
    path: Path, *, errors: tuple[str, ...] = ("error",), expected: int | None = None
) -> int:
    if not path.is_file():
        raise PipelineError(f"Artifact not found: {path}")
    count = failures = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("row must be an object")
                count += 1
                failures += int(any(row.get(field) for field in errors))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PipelineError(f"Invalid JSONL artifact: {path}") from exc
    if not count:
        raise PipelineError(f"Artifact is empty: {path}")
    if failures:
        raise PipelineError(
            f"{failures}/{count} rows failed in {path}; inspect the artifact/log and rerun this stage"
        )
    if expected is not None and count != expected:
        raise PipelineError(f"Expected {expected} rows, found {count}: {path}")
    return count


@dataclass
class Experiment:
    dataset: str
    raw: Path | None
    processed: Path
    wiki: Path
    retrieval: Path
    predictions: Path
    eval_dir: Path
    variant: str
    mode: str
    backend: str
    env: dict[str, str]
    args: argparse.Namespace
    stages: tuple[str, ...]

    @property
    def source(self) -> Path:
        return ROOT / "src" / self.dataset

    def query_count(self, *, reading: bool) -> int:
        records = read_records(self.processed, self.dataset)
        count = (
            len(records)
            if self.dataset == "longmemeval"
            else sum(
                1
                for row in records
                for query in row["qa"]
                if not reading or str(query.get("category")) != "5"
            )
        )
        return min(count, self.args.limit) if reading and self.args.limit else count

    def preflight(self) -> None:
        stages = self.stages
        if "preprocess" in stages:
            if self.raw is None:
                names = ", ".join(RAW_NAMES[self.dataset])
                raise PipelineError(
                    f"Place {names} under {self.processed.parent} (or its raw/ subfolder)"
                )
            read_records(self.raw, self.dataset, raw=True)
        else:
            if (
                any(stage in stages for stage in ("index", "retrieve", "infer"))
                or self.dataset == "longmemeval"
            ):
                read_records(self.processed, self.dataset)
        if stages[0] == "retrieve" and self.mode != "raw-only":
            inspect_jsonl(self.wiki)
        if stages[0] == "infer":
            inspect_jsonl(self.wiki)
            inspect_jsonl(self.retrieval, expected=self.query_count(reading=False))
        if stages[0] == "evaluate":
            inspect_jsonl(self.predictions, errors=())
        controller_options = any(
            (
                self.args.output_budget_tokens is not None,
                self.args.reasoning_budget is not None,
                self.args.dump_prompts,
            )
        )
        if controller_options and (
            self.dataset != "locomo" or self.variant != "structured"
        ):
            raise PipelineError(
                "Budget and prompt-dump options require structured LoCoMo."
            )
        if self.variant == "summary" and self.mode != "full":
            raise PipelineError("Summary readers support --read-mode full only.")
        modules = []
        if any(stage in stages for stage in ("index", "retrieve", "infer", "evaluate")):
            modules += [
                ("openai", "requirements.txt"),
                ("tqdm", "requirements.txt"),
                ("json_repair", "requirements.txt"),
            ]
        if "retrieve" in stages:
            modules.append(("numpy", "requirements.txt"))
            if self.backend == "local":
                modules.append(("sentence_transformers", "requirements-local.txt"))
        for module, requirements in modules:
            if importlib.util.find_spec(module) is None:
                raise PipelineError(
                    f"Missing {module}; run: python -m pip install -r {requirements}"
                )
        if any(stage in stages for stage in ("index", "infer")):
            require_url(self.env, "BASE_URL", "T2MEM_LLM_BASE_URL")
            require_model(self.env, "MODEL", "T2MEM_LLM_MODEL_NAME")
        if "retrieve" in stages and self.backend == "api":
            require_url(self.env, "EMBED_BASE_URL", "T2MEM_EMBEDDING_BASE_URL")
            require_model(self.env, "EMBED_MODEL", "T2MEM_EMBEDDING_MODEL_NAME")
        if "evaluate" in stages:
            require_url(self.env, "JUDGE_BASE_URL")
            if not self.env.get("JUDGE_API_KEY"):
                raise PipelineError(
                    "Set JUDGE_API_KEY in .env, or use --skip-eval for inference only."
                )
            require_model(self.env, "JUDGE_MODEL")

    def command(self, stage: str) -> tuple[list[str], Path]:
        args = self.args
        env = self.env
        llm = [
            "--base-url",
            setting(env, "BASE_URL", "T2MEM_LLM_BASE_URL"),
            "--api-key",
            setting(env, "API_KEY", "T2MEM_LLM_API_KEY"),
            "--model-name",
            setting(env, "MODEL", "T2MEM_LLM_MODEL_NAME"),
        ]
        workers = integer_setting(args.workers, env, "WORKERS", default=8)
        timeout = setting(env, "T2MEM_REQUEST_TIMEOUT", default="120")
        if stage == "preprocess":
            entry = self.source / "wiki/preprocess_data.py"
            flags = ["--data-path", str(self.raw), "--output-path", str(self.processed)]
            output = self.processed
        elif stage == "index":
            entry = (
                self.source
                / "wiki"
                / (
                    "build_summary_wiki.py"
                    if self.variant == "summary"
                    else "build_wiki.py"
                )
            )
            flags = [
                "--data-path",
                str(self.processed),
                "--output-path",
                str(self.wiki),
                "--max-workers",
                str(workers),
                "--request-timeout",
                timeout,
                *llm,
            ]
            output = self.wiki
        elif stage == "retrieve":
            representation = (
                "raw"
                if self.mode == "raw-only"
                else ("summary_wiki" if self.variant == "summary" else "wiki")
            )
            entry = (
                self.source
                / "retrieval"
                / f"{self.dataset}_retrieval_{representation}.py"
            )
            top_k = integer_setting(
                args.retrieval_top_k, env, "T2MEM_RETRIEVAL_TOP_K", default=10
            )
            flags = [
                "--data-path",
                str(self.processed),
                "--output-path",
                str(self.retrieval),
                "--backend",
                self.backend,
                "--top-k",
                str(top_k),
            ]
            if representation != "raw":
                flags += ["--wiki-path", str(self.wiki)]
            if self.backend == "api":
                flags += [
                    "--base-url",
                    setting(env, "EMBED_BASE_URL", "T2MEM_EMBEDDING_BASE_URL"),
                    "--model-name",
                    setting(env, "EMBED_MODEL", "T2MEM_EMBEDDING_MODEL_NAME"),
                    "--api-key",
                    setting(
                        env,
                        "EMBED_API_KEY",
                        "T2MEM_EMBEDDING_API_KEY",
                        "OPENAI_API_KEY",
                        default="EMPTY",
                    ),
                    "--request-timeout",
                    timeout,
                ]
            else:
                flags += [
                    "--model-path",
                    setting(
                        env,
                        "EMBED_MODEL_PATH",
                        "T2MEM_EMBEDDING_MODEL",
                        default="BAAI/bge-m3",
                    ),
                ]
            output = self.retrieval
        elif stage == "infer":
            entry = (
                self.source
                / "t2mem"
                / (
                    "t2mem_summary_answer.py"
                    if self.variant == "summary"
                    else "t2mem_answer.py"
                )
            )
            flags = [
                "--data-path",
                str(self.processed),
                "--wiki-path",
                str(self.wiki),
                "--retrieval-path",
                str(self.retrieval),
                "--output-path",
                str(self.predictions),
                "--max-workers",
                str(workers),
                "--request-timeout",
                timeout,
                "--top-k",
                str(integer_setting(args.top_k, env, "T2MEM_READ_TOP_K", default=5)),
                "--max-steps",
                str(integer_setting(args.max_steps, env, "MAX_STEPS", default=5)),
                *llm,
            ]
            if not (self.variant == "summary" and self.dataset == "locomo"):
                flags += ["--dataset", self.dataset]
            if self.variant == "structured":
                flags += ["--read-mode", self.mode]
            for flag, value in (
                (
                    "--controller-model-name",
                    args.controller_model_name
                    or env.get("T2MEM_CONTROLLER_MODEL_NAME"),
                ),
                (
                    "--answer-model-name",
                    args.answer_model_name or env.get("T2MEM_ANSWER_MODEL_NAME"),
                ),
            ):
                if value:
                    flags += [flag, value]
            if args.limit:
                flags += ["--limit", str(args.limit)]
            for flag, value in (
                ("--output-budget-tokens", args.output_budget_tokens),
                ("--reasoning-budget", args.reasoning_budget),
                ("--answer-max-tokens", args.answer_max_tokens),
            ):
                if value is not None:
                    flags += [flag, str(value)]
            if args.dump_prompts:
                flags += ["--dump-prompts"]
            output = self.predictions
        else:
            # Convert LoCoMo input with the existing adapter before launching its judge.
            entry = ROOT / "evaluation" / f"{self.dataset}_judge.py"
            judge_input = (
                self.predictions
                if self.dataset == "longmemeval"
                else self.eval_dir / f"judge_{self.predictions.stem}_input.json"
            )
            output = self.eval_dir / f"judge_{self.predictions.stem}.jsonl"
            flags = [
                "--input-path",
                str(judge_input),
                "--output-path",
                str(output),
                "--summary-path",
                str(output.with_suffix(".summary.json")),
                "--model-name",
                env["JUDGE_MODEL"],
                "--max-workers",
                str(workers),
                "--request-timeout",
                timeout,
            ]
            if self.dataset == "longmemeval":
                flags += ["--data-path", str(self.processed)]
            if args.limit:
                flags += ["--limit", str(args.limit)]
        if args.temperature is not None and stage in ("index", "infer", "evaluate"):
            flags += ["--temperature", str(args.temperature)]
        return [sys.executable, str(entry), *flags], output

    def run(self) -> None:
        for index, stage in enumerate(self.stages, start=1):
            command, output = self.command(stage)
            print(
                f"\n[{self.dataset}] {index}/{len(self.stages)} {stage} -> {output}",
                flush=True,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            log_path = output.with_suffix(".log")
            if stage == "evaluate" and self.dataset == "locomo":
                adapter = [
                    sys.executable,
                    str(ROOT / "evaluation/t2mem_jsonl_to_locomo_judge.py"),
                    "--input-path",
                    str(self.predictions),
                    "--output-path",
                    str(self.eval_dir / f"judge_{self.predictions.stem}_input.json"),
                ]
                execute(adapter, self.env, log_path)
            execute(
                command,
                self.env,
                log_path,
                append=(stage == "evaluate" and self.dataset == "locomo"),
            )
            if stage == "preprocess":
                read_records(output, self.dataset)
            elif stage == "index":
                expected = sum(
                    len(row["sessions"])
                    for row in read_records(self.processed, self.dataset)
                )
                inspect_jsonl(output, expected=expected)
            elif stage == "retrieve":
                inspect_jsonl(output, expected=self.query_count(reading=False))
            elif stage == "infer":
                inspect_jsonl(output, expected=self.query_count(reading=True))
            else:
                inspect_jsonl(
                    output, errors=("error", "judge_error", "judge_invalid_response")
                )
        print(f"\n[{self.dataset}] Completed: {', '.join(self.stages)}", flush=True)


def require_url(env: dict[str, str], *keys: str) -> None:
    value = setting(env, *keys)
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise PipelineError(f"Set {keys[-1]} to an HTTP(S) API URL in .env.")


def require_model(env: dict[str, str], *keys: str) -> None:
    value = setting(env, *keys)
    if not value or value in ("your-served-model", "your-controller", "your-answer"):
        raise PipelineError(f"Set {keys[-1]} to your served model name in .env.")


def execute(
    command: list[str], env: dict[str, str], log_path: Path, *, append: bool = False
) -> None:
    # Never print command arguments: legacy CLIs accept API keys as flags.
    with (
        log_path.open("a" if append else "w", encoding="utf-8") as log,
        subprocess.Popen(
            command,
            cwd=ROOT,
            env={**env, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process,
    ):
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise PipelineError(
            f"{Path(command[1]).name} exited with status {code}; see {log_path}"
        )


def build_experiments(
    args: argparse.Namespace, env: dict[str, str]
) -> list[Experiment]:
    data_dir = repo_path(args.data_dir or env.get("DATA_ROOT") or ROOT / "data")
    output_dir = repo_path(args.output_dir or ROOT / "outputs")
    eval_dir = repo_path(args.eval_dir or env.get("OUT_DIR") or ROOT / "eval-results")
    stages = STAGES if args.stage == "all" else (args.stage,)
    try:
        timeout = float(setting(env, "T2MEM_REQUEST_TIMEOUT", default="120"))
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError
    except ValueError as exc:
        raise PipelineError(
            "T2MEM_REQUEST_TIMEOUT must be a positive finite number."
        ) from exc
    if args.skip_eval:
        stages = stages[:-1]
    datasets = (
        [args.dataset]
        if args.dataset
        else [
            name
            for name in RAW_NAMES
            if raw_candidates(data_dir, name)
            or (data_dir / f"{name}_processed.json").is_file()
        ]
    )
    if not datasets:
        raise PipelineError(
            f"No dataset found in {data_dir}; add locomo10.json or longmemeval_s_cleaned.json."
        )
    variant = args.variant or env.get("VARIANT") or "structured"
    if variant not in ("structured", "summary"):
        raise PipelineError("VARIANT must be structured or summary.")
    backend = args.embed_backend or env.get("EMBED_BACKEND") or "api"
    if backend not in ("api", "local"):
        raise PipelineError("EMBED_BACKEND must be api or local.")
    experiments = []
    for dataset in datasets:
        raw_key = "T2MEM_LOCOMO_RAW" if dataset == "locomo" else "T2MEM_LONGMEMEVAL_RAW"
        explicit = args.data_path or env.get(raw_key)
        candidates = raw_candidates(data_dir, dataset)
        if explicit:
            raw = repo_path(explicit)
        elif "preprocess" in stages and len(candidates) > 1:
            raise PipelineError(
                f"Multiple {dataset} datasets found: {candidates}; select one with --data-path."
            )
        else:
            raw = candidates[0] if candidates else None
        wiki_dir = data_dir / dataset
        suffix = "summary_wiki" if variant == "summary" else "wiki"
        unit = "raw" if args.read_mode == "raw-only" else suffix
        prediction = (
            repo_path(args.predictions)
            if args.predictions
            else output_dir
            / f"{dataset}_{variant}_t2mem_{args.read_mode.replace('-', '_')}_answers.jsonl"
        )
        experiment_env = dict(env)
        experiment_env.setdefault("JUDGE_MODEL", "qwen3.5-plus")
        experiment = Experiment(
            dataset,
            raw,
            data_dir / f"{dataset}_processed.json",
            wiki_dir / f"{dataset}_{suffix}.jsonl",
            wiki_dir / f"{dataset}_retrieval_{unit}.jsonl",
            prediction,
            eval_dir,
            variant,
            args.read_mode,
            backend,
            experiment_env,
            args,
            stages,
        )
        # Validate all stage-specific numeric settings before the first model call.
        for stage in stages:
            experiment.command(stage)
        experiment.preflight()
        experiments.append(experiment)
    return experiments


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        env = load_config(args.env_file)
        experiments = build_experiments(args, env)
        if args.check:
            for experiment in experiments:
                print(
                    f"{experiment.dataset}: configuration and inputs OK; stages={', '.join(experiment.stages)}"
                )
            return 0
        for experiment in experiments:
            experiment.run()
        return 0
    except (PipelineError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the desired stage with --stage.", file=sys.stderr)
        return 130
