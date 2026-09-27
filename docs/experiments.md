# Experiment settings

## Supported configurations

| Entry point | `full` / `wiki-only` / `raw-only` | Output budget | Reasoning budget | Prompt dump | Token summary |
|---|---|---|---|---|---|
| Structured LoCoMo | Yes | Yes | Yes | Yes | Yes |
| Structured LongMemEval | Yes | No | No | No | No |
| Summary LoCoMo | `full` only | No | No | No | No |
| Summary LongMemEval | `full` only | No | No | No | No |

Each benchmark keeps its original controller and answer prompts. Unsupported flags fail in the Python argument parser; runner combinations are also checked before launching work.

## Memory format

| `VARIANT` | Index builder | Inference entry point | Index filename | Retrieval filename |
|---|---|---|---|---|
| `structured` (default) | `wiki/build_wiki.py` | `t2mem/t2mem_answer.py` | `<dataset>_wiki.jsonl` | `<dataset>_retrieval_wiki.jsonl` |
| `summary` | `wiki/build_summary_wiki.py` | `t2mem/t2mem_summary_answer.py` | `<dataset>_summary_wiki.jsonl` | `<dataset>_retrieval_summary_wiki.jsonl` |

Summary entries contain a single natural-language summary per session. Their reader uses the existing full controller action set; the runner requires `full` rather than silently ignoring a different read mode.

## Structured reading modes

| Mode | Initial observation | Allowed controller actions | Retrieval representation |
|---|---|---|---|
| `full` | Episode indexes | `SEARCH_WIKI`, `READ_RAW_FROM_WIKI`, `ANSWER` | Structured index |
| `wiki-only` | Episode indexes | `SEARCH_WIKI`, `ANSWER` | Structured index |
| `raw-only` | Raw session transcripts | `SEARCH_RAW`, `ANSWER` | Raw session |

`SEARCH_WIKI` and `SEARCH_RAW` advance over the precomputed ranked candidate list. `READ_RAW_FROM_WIKI` resolves a selected index ID to its raw source session. An action outside the permitted set falls back to `ANSWER` and increments `action_fallback_count`.

Wiki and raw candidates are ranked using different representations. Comparing these modes therefore changes both retrieval and reading. Even `raw-only` entry points currently load the wiki file during initialization, so retain the structured index artifact when running that ablation.

## Embedding backends

| Benchmark and representation | API | Local | Default |
|---|---|---|---|
| Structured LoCoMo index | Yes | Yes | API |
| Raw LoCoMo session | No | Yes | Local |
| Summary LoCoMo index | No | Yes | Local |
| Structured LongMemEval index | No | Yes | Local |
| Raw LongMemEval session | No | Yes | Local |
| Summary LongMemEval index | No | Yes | Local |

Use `python -m pip install -r requirements-local.txt` for the local backend. `EMBED_MODEL_PATH` or `T2MEM_EMBEDDING_MODEL` must identify a sentence-transformers model. API requests use `EMBED_BASE_URL`, `EMBED_MODEL`, and `EMBED_API_KEY`, falling back to their `T2MEM_EMBEDDING_*` equivalents. Unsupported API combinations fail explicitly instead of switching backends.

## Runner configuration

All runners accept additional Python CLI arguments after their positional arguments. Use the relevant Python entry point's `--help` for its complete flag list. Extra CLI flags override flags supplied earlier by the runner.

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
| `EMBED_BACKEND` | As shown in the embedding matrix |
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

The LoCoMo controller-output budget gates each call with `min_call_tokens = 50` and `answer_reserve = 0`. A budget below 50 cannot fund the first controller call. If no evidence is produced or truncated output cannot be parsed, the existing fallback uses the initial retrieved wiki snippets. Answer tokens are outside this controller budget. Different endpoints may report token usage differently; inspect the recorded usage and stop reason.

Example sweep (the numbers are settings, not reproduced result claims):

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

- LoCoMo category 5 is excluded by its readers and judge, preserving the supplied experiment scope.
- LoCoMo's adapter now retains failed prediction rows by default as empty responses. The adapter's explicit `--skip-errors` option remains available for reproducing previously filtered evaluations; the evaluation runner does not enable it.
- LoCoMo overall `accuracy` is computed over successful judge responses. Its category accuracy includes judge-error rows in the denominator. Inspect `total_rows`, `judged_rows`, and `error_rows` before reporting results.
- LongMemEval preserves its task-specific judge prompts and abstention handling. Its accuracy uses valid judge labels; inspect invalid and failed judgment counts.
- Neither evaluator reconstructs missing queries. Supply the full prediction file for a full benchmark evaluation, and report any limits or missing rows.
- These are the LLM-judge conventions in the supplied experiment code. They should not be presented as an independent implementation of every upstream benchmark metric.

## Changes from the supplied snapshot

The method is consistently named **T2Mem** in paths, imports, function names, environment variables, and output filenames. Deployment-specific defaults were removed from the summary branches. Runner paths now match index/retrieval filenames, summary entry points receive only supported arguments, and retrieval receives the selected wiki path. The local environment includes NumPy. Unsupported backend combinations fail explicitly. The LoCoMo adapter no longer silently filters failed predictions during evaluation.

The original input formats, active benchmark prompts, controller actions, candidate ranking formulas, evidence accumulation, and answer-generation logic are retained. An original snapshot was backed up locally before organizing this repository.
