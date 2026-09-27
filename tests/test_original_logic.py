"""Guard research logic against the pre-organization source baseline.

The baseline contains hashes from the original archive, not a second source copy.
AST comparison ignores comments, docstrings, type annotations, the method rename,
embedding construction/loader plumbing, and summary-client construction. Prompts
are compared as evaluated string literals, including whitespace. CLI defaults and
choices are checked separately. No dataset, API credential, or model is required.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).parent / "fixtures/original_logic.json"


def rename_source(source: str) -> str:
    for old, new in (
        ("dualreading_summary_answer", "t2mem_summary_answer"),
        ("dualreading_answer", "t2mem_answer"),
        ("DUALREAD", "T2MEM"),
        ("DualRead", "T2Mem"),
        ("dualread", "t2mem"),
        ("proceed_data", "preprocess_data"),
    ):
        source = source.replace(old, new)
    return source


class LogicNormalizer(ast.NodeTransformer):
    def __init__(self, path: str):
        self.retrieval = "/retrieval/" in path
        self.summary_builder = path.endswith("wiki/build_summary_wiki.py")

    def visit_FunctionDef(self, node):
        node.returns = None
        return self.generic_visit(node)

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Try(self, node):
        node = self.generic_visit(node)
        # Only the optional SentenceTransformer import differs in these modules.
        if self.retrieval and not node.body:
            return None
        return node

    def visit_Assign(self, node):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if self.retrieval and names == ["model"]:
            return None
        if (self.summary_builder or self.retrieval) and names == ["client"]:
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if self.retrieval and isinstance(node.test, ast.UnaryOp):
            value = node.test.operand
            if (
                isinstance(node.test.op, ast.Not)
                and isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "args"
                and value.attr in ("api_key", "base_url")
            ):
                return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        if (
            self.retrieval
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
            and node.value.args
            and isinstance(node.value.args[0], ast.JoinedStr)
            and isinstance(node.value.args[0].values[0], ast.Constant)
            and node.value.args[0]
            .values[0]
            .value.startswith(("Loading embedding model", "Embedding backend"))
        ):
            return None
        return self.generic_visit(node)


def digest(value: ast.AST) -> str:
    def canonical(node):
        if isinstance(node, ast.AST):
            return {
                "node": type(node).__name__,
                **{
                    name: canonical(item)
                    for name, item in ast.iter_fields(node)
                    if name != "type_params" and item is not None
                },
            }
        if isinstance(node, list):
            return [canonical(item) for item in node]
        return node

    payload = json.dumps(canonical(value), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def snapshot(path: str, source: str) -> dict:
    tree = ast.parse(rename_source(source))
    functions = {}
    constants = {}
    literal_constants = {}
    options = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name not in (
                "parse_args",
                "load_sentence_transformer",
                "_default_data_path",
                "_default_output_path",
            ):
                functions[node.name] = digest(LogicNormalizer(path).visit(node))
            else:
                for call in ast.walk(node):
                    if (
                        not isinstance(call, ast.Call)
                        or not isinstance(call.func, ast.Attribute)
                        or call.func.attr != "add_argument"
                        or not call.args
                    ):
                        continue
                    name = ast.literal_eval(call.args[0])
                    values = {}
                    for keyword in call.keywords:
                        if keyword.arg not in ("default", "choices", "action"):
                            continue
                        # Deployment-specific endpoints/model names are configuration.
                        if name in ("--base-url", "--api-key", "--model-name"):
                            continue
                        try:
                            values[keyword.arg] = ast.literal_eval(keyword.value)
                        except (ValueError, TypeError):
                            pass
                    if values:
                        options[name] = values
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "DEFAULT_LLM_MODEL_NAME":
                        continue
                    constants[target.id] = hashlib.sha256(
                        node.value.value.encode()
                    ).hexdigest()
        elif isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    literal_constants[target.id] = digest(node.value)
    return json.loads(
        json.dumps(
            {
                "functions": functions,
                "strings": constants,
                "literal_constants": literal_constants,
                "cli_options": options,
            }
        )
    )


class OriginalLogicTests(unittest.TestCase):
    def test_research_logic_and_prompt_strings_match_original(self):
        baseline = json.loads(BASELINE.read_text())
        for path, expected in baseline["files"].items():
            with self.subTest(path=path):
                actual = snapshot(path, (ROOT / path).read_text())
                self.assertEqual(actual["functions"], expected["functions"], path)
                self.assertEqual(actual["strings"], expected["strings"], path)
                self.assertEqual(
                    actual["literal_constants"], expected["literal_constants"], path
                )

    def test_original_cli_defaults_and_choices_remain_available(self):
        baseline = json.loads(BASELINE.read_text())
        for path, expected in baseline["files"].items():
            actual = snapshot(path, (ROOT / path).read_text())
            for name, values in expected["cli_options"].items():
                with self.subTest(path=path, option=name):
                    self.assertEqual(actual["cli_options"].get(name), values)


class RunnerDefaultsTests(unittest.TestCase):
    def make_experiment(self, dataset, variant, args=None, env=None):
        from src.common.pipeline import Experiment, parse_args

        return Experiment(
            dataset=dataset,
            raw=ROOT / "data/raw.json",
            processed=ROOT / "data/processed.json",
            wiki=ROOT / "data/wiki.jsonl",
            retrieval=ROOT / "data/retrieval.jsonl",
            predictions=ROOT / "outputs/predictions.jsonl",
            eval_dir=ROOT / "eval-results",
            variant=variant,
            mode="full",
            backend="api",
            env={"JUDGE_MODEL": "test-judge", **(env or {})},
            args=args or parse_args([]),
            stages=("index", "retrieve", "infer", "evaluate"),
        )

    def test_defaults_do_not_override_original_timeout_or_sampling(self):
        for dataset in ("locomo", "longmemeval"):
            for variant in ("structured", "summary"):
                experiment = self.make_experiment(dataset, variant)
                for stage, workers in (
                    ("index", "16"),
                    ("infer", "8"),
                    ("evaluate", "4"),
                ):
                    with self.subTest(dataset=dataset, variant=variant, stage=stage):
                        command, _ = experiment.command(stage)
                        self.assertEqual(
                            command[command.index("--max-workers") + 1], workers
                        )
                        self.assertNotIn("--request-timeout", command)
                        self.assertNotIn("--temperature", command)
                        self.assertNotIn("--answer-max-tokens", command)

    def test_explicit_overrides_are_forwarded(self):
        from src.common.pipeline import parse_args

        args = parse_args(
            ["--workers", "2", "--temperature", "0.0", "--answer-max-tokens", "1024"]
        )
        experiment = self.make_experiment(
            "locomo", "structured", args, {"T2MEM_REQUEST_TIMEOUT": "240"}
        )
        for stage in ("index", "retrieve", "infer", "evaluate"):
            command, _ = experiment.command(stage)
            self.assertEqual(command[command.index("--request-timeout") + 1], "240")
            if stage != "retrieve":
                self.assertEqual(command[command.index("--max-workers") + 1], "2")
                self.assertEqual(command[command.index("--temperature") + 1], "0.0")
            if stage == "infer":
                self.assertEqual(
                    command[command.index("--answer-max-tokens") + 1], "1024"
                )
        experiment.backend = "local"
        self.assertNotIn("--request-timeout", experiment.command("retrieve")[0])

    def test_longmemeval_preprocess_preserves_original_output_alias(self):
        entry = ROOT / "src/longmemeval/wiki/preprocess_data.py"
        code = (
            "import runpy, sys; "
            "module = runpy.run_path(sys.argv[1]); "
            "print(module['_default_output_path']('longmemeval'))"
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("T2MEM_LONGMEMEVAL_DATA", "T2MEM_LME_DATA")
        }
        for settings, expected in (
            ({"T2MEM_LONGMEMEVAL_DATA": "original.json"}, "original.json"),
            ({"T2MEM_LME_DATA": "alias.json"}, "alias.json"),
            (
                {
                    "T2MEM_LONGMEMEVAL_DATA": "original.json",
                    "T2MEM_LME_DATA": "alias.json",
                },
                "original.json",
            ),
        ):
            with self.subTest(settings=settings):
                result = subprocess.run(
                    [sys.executable, "-c", code, str(entry)],
                    env={**env, **settings},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertEqual(result.stdout.strip(), expected)


if __name__ == "__main__":
    unittest.main()
