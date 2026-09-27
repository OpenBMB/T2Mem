# Experiment settings

## Unified entry point

`python run.py` loads the repository `.env`, discovers datasets directly in `data/` or `data/raw/`, and runs preprocess → index → retrieve → infer → evaluate. Both benchmarks support API embeddings with `requirements.txt` alone.

```bash
python run.py --check                         # no model calls
python run.py                                # all discovered benchmarks
python run.py locomo --skip-eval              # through inference
python run.py longmemeval --stage preprocess
python run.py longmemeval --stage index
python run.py longmemeval --stage retrieve
python run.py longmemeval --stage infer
python run.py longmemeval --stage evaluate
```

Use `--variant` and `--read-mode` consistently across stages. `--data-path` selects a custom raw file; `--predictions` selects an evaluation input. `--data-dir`, `--output-dir`, and `--eval-dir` relocate artifacts. Multiple matching raw files require explicit selection.

CLI options override environment settings; existing environment variables override `.env`. Optional `T2MEM_CONTROLLER_MODEL_NAME` and `T2MEM_ANSWER_MODEL_NAME` select separate models on the same LLM endpoint.

`--retrieval-top-k` / `T2MEM_RETRIEVAL_TOP_K` controls ranked candidates (default 10). `--top-k` / `T2MEM_READ_TOP_K` controls the initial observation (default 5). `--limit` restricts inference and evaluation, not indexing or retrieval. `WORKERS`, `MAX_STEPS`, and `T2MEM_REQUEST_TIMEOUT` configure concurrency, rounds, and request timeout (default 120 seconds).

Preflight checks inputs, configuration, dependencies, and supported settings before model calls. Each stage verifies its output. Empty artifacts, failed rows, and unexpected query counts stop the pipeline with a nonzero status. Logs and partial artifacts remain for diagnosis. Evaluation can be run explicitly on failed predictions; the adapter retains them as empty responses.

Repeated stages replace their generated outputs; prior files are not silently reused. Select an individual stage to continue an experiment. Use `python run.py --help` for all options.

## Supported configurations

| Entry point | `full` / `wiki-only` / `raw-only` | Output budget | Reasoning budget | Prompt dump | Token summary |
|---|---|---|---|---|---|
| Structured LoCoMo | Yes | Yes | Yes | Yes | Yes |
| Structured LongMemEval | Yes | No | No | No | No |
| Summary LoCoMo | `full` only | No | No | No | No |
| Summary LongMemEval | `full` only | No | No | No | No |

Controller and answer prompts are benchmark-specific. Unsupported parameter combinations are rejected before execution.

## Memory format

| `VARIANT` | Index builder | Inference entry point | Index filename | Retrieval filename |
|---|---|---|---|---|
| `structured` (default) | `wiki/build_wiki.py` | `t2mem/t2mem_answer.py` | `<dataset>_wiki.jsonl` | `<dataset>_retrieval_wiki.jsonl` |
| `summary` | `wiki/build_summary_wiki.py` | `t2mem/t2mem_summary_answer.py` | `<dataset>_summary_wiki.jsonl` | `<dataset>_retrieval_summary_wiki.jsonl` |

Summary entries contain a single natural-language summary per session. Their reader uses the full controller action set and requires `--read-mode full`.

## Structured reading modes

| Mode | Initial observation | Allowed controller actions | Retrieval representation |
|---|---|---|---|
| `full` | Episode indexes | `SEARCH_WIKI`, `READ_RAW_FROM_WIKI`, `ANSWER` | Structured index |
| `wiki-only` | Episode indexes | `SEARCH_WIKI`, `ANSWER` | Structured index |
| `raw-only` | Raw session transcripts | `SEARCH_RAW`, `ANSWER` | Raw session |

`SEARCH_WIKI` and `SEARCH_RAW` advance over the precomputed ranked candidate list. `READ_RAW_FROM_WIKI` resolves a selected index ID to its raw source session. An action outside the permitted set falls back to `ANSWER` and increments `action_fallback_count`.

Wiki and raw candidates are ranked using different representations. Comparing these modes therefore changes both retrieval and reading. Even `raw-only` entry points currently load the wiki file during initialization, so retain the structured index artifact when running that ablation.

## Embedding backends

API and local backends support structured indexes, summary indexes, and raw sessions on both benchmarks. Retrieval uses benchmark-specific text representations and normalized embedding similarity.

`run.py` defaults to API embeddings. Configure the embedding endpoint, served model, and key in `.env`; only `requirements.txt` is needed. Local embeddings require `requirements-local.txt` and a sentence-transformers model directory or ID in `T2MEM_EMBEDDING_MODEL`.

Shell runners default to: API for structured LoCoMo wiki retrieval, local otherwise. Set `EMBED_BACKEND` explicitly; the example `.env` selects API.

## Shell runner configuration

The `scripts/*.sh` entry points run individual experiment stages. Export `.env` first with `set -a; source .env; set +a`. All shell runners accept additional Python CLI arguments after their positional arguments. Use the relevant Python entry point's `--help` for its complete flag list. Extra CLI flags override flags supplied earlier by the runner.

| Variable | Default / purpose |
|---|---|
| `PYTHON` | `python`; may be an absolute interpreter path |
| `DATA_ROOT` | `<repo>/data` |
| `RAW_DATA` | Raw file selected by the preprocessing runner |
| `DATA` | `<DATA_ROOT>/<dataset>_processed.json` |
| `WIKI_ROOT` | `<DATA_ROOT>/<dataset>` |
| `WIKI` | Structured or summary index chosen by `VARIANT` |
| `RETRIEVAL` | Representation selected by the reader |
| `OUT` | Index/retrieval output or prediction output, depending on runner |
| `OUT_DIR` | `<repo>/eval-results`, used by evaluation |
| `VARIANT` | `structured`; alternative: `summary` |
| `BASE_URL`, `API_KEY`, `MODEL` | Override `T2MEM_LLM_*`; fallback URL is localhost and fallback model is `Qwen3.5-27B` |
| `WORKERS` | 16 for indexing, 8 for inference |
| `TOP_K` | 10 for retrieval, 5 initially observed by inference |
| `MAX_STEPS` | 5 controller rounds |
| `EMBED_BACKEND` | API for structured LoCoMo wiki; local otherwise |
| `JUDGE_MODEL` | Overrides the judge model; otherwise `qwen3.5-plus` |

`OUT` determines the runner's log path. Prefer environment overrides for paths; if an extra `--output-path` is used, the Python artifact moves but the wrapper log remains at the `OUT` location.

Direct Python entry points also accept `T2MEM_LOCOMO_DATA`, `T2MEM_LME_DATA`, `T2MEM_LOCOMO_RAW`, and `T2MEM_LONGMEMEVAL_RAW`. Use `DATA` / `RAW_DATA` when configuring shell runners. `T2MEM_ROOT` overrides the evaluation config's repository root.

## Reading parameters

| Flag | Default | Availability |
|---|---|---|
| `--top-k` | 5 | All readers |
| `--max-steps` | 5 | All readers |
| `--temperature` | 0.0 | All readers |
| `--answer-max-tokens` | 1024; summary LoCoMo: 512 | All readers |
| `--controller-model-name`, `--answer-model-name` | Same as `--model-name` | All readers |
| `--limit` | All queries | All readers |
| `--output-budget-tokens` | Unlimited | Structured LoCoMo |
| `--reasoning-budget` | Unlimited reasoning words | Structured LoCoMo |
| `--dump-prompts` | Disabled | Structured LoCoMo |

The LoCoMo controller-output budget gates each call with `min_call_tokens = 50` and `answer_reserve = 0`. A budget below 50 cannot fund the first controller call. If no evidence is produced or truncated output cannot be parsed, the fallback uses the initial retrieved wiki snippets. Answer tokens are outside this controller budget. Different endpoints may report token usage differently; inspect the recorded usage and stop reason.

Controller-output-budget sweep:

```bash
for budget in 50 100 200 400 800 1200; do
  OUT="outputs/locomo_budget_${budget}.jsonl" \
    bash scripts/run_t2mem.sh locomo wiki-only --output-budget-tokens "$budget"
done
```

## Outputs

Default inference output:

```text
outputs/<dataset>_<variant>_t2mem_<mode-with-underscores>_answers.jsonl
```

Each row contains `prediction`, query metadata, `controller_steps`, `accumulated_evidence`, `steps_taken`, and `stop_reason`. A failed query contains an empty prediction and a nonempty `error`. Structured LoCoMo additionally records stage-specific token usage and writes an aggregate `<output-stem>_token_usage_summary.json`. Summary LoCoMo has row-level controller usage but no aggregate summary; LongMemEval readers do not currently report aggregate token usage.

Preserve model versions, dataset versions, endpoint settings, index artifacts, embedding configuration, candidate count, and reader/judge arguments when archiving an experiment. `--dump-prompts` on structured LoCoMo records the exact inputs for analysis.

## Evaluation conventions

- LoCoMo category 5 is excluded by its readers and judge.
- The LoCoMo adapter represents failed predictions as empty responses. `--skip-errors` excludes those rows when explicitly requested; the pipeline does not enable this option.
- LoCoMo overall `accuracy` is computed over successful judge responses. Its category accuracy includes judge-error rows in the denominator. Inspect `total_rows`, `judged_rows`, and `error_rows` before reporting results.
- LongMemEval uses task-specific judge prompts and abstention handling. Its accuracy denominator includes submitted rows, including invalid or failed judgments; inspect those counts.
- Neither evaluator reconstructs missing queries. Supply the full prediction file for a full benchmark evaluation, and report any limits or missing rows.
- The reported metrics use the LLM-judge conventions described above.
