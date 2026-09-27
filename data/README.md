# Data

Datasets are **not** shipped with this repository. Obtain them from the benchmark authors:

- [LoCoMo repository](https://github.com/snap-research/locomo)
- [LongMemEval repository](https://github.com/xiaowu0162/LongMemEval) and [cleaned dataset release](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)

From the repository root:

```bash
mkdir -p data
curl -fL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o data/locomo10.json
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  -o data/longmemeval_s_cleaned.json
python run.py locomo --stage preprocess
python run.py longmemeval --stage preprocess
```

Keep the downloaded version or checksum with your experiment record. The repository's MIT license does not replace the benchmark's data terms.

Put `locomo10.json` and/or `longmemeval_s_cleaned.json` directly in `data/`. After installing `requirements.txt` and configuring `.env`, run:

```bash
python run.py --check
python run.py
```

The runner discovers available datasets and creates derived files automatically. The existing `data/raw/` location also works. If multiple raw files match one benchmark, select one with `--data-path`. Raw inputs remain unchanged; repeating a stage replaces its generated outputs.

```bash
python run.py locomo --stage preprocess
python run.py locomo --stage index
python run.py locomo --stage retrieve
python run.py locomo --stage infer
python run.py locomo --stage evaluate
```

Use `longmemeval` for the other benchmark. Use `--data-dir` or `DATA_ROOT` to change
the unified runner's data directory. The legacy shell runners also accept
individual `DATA` / `WIKI_ROOT` / `RETRIEVAL` / `WIKI` overrides.

```
data/
├── locomo10.json                  raw LoCoMo data (not distributed)
├── longmemeval_s_cleaned.json      raw LongMemEval data (not distributed)
├── locomo_processed.json
├── locomo/
│   ├── locomo_wiki.jsonl
│   ├── locomo_summary_wiki.jsonl
│   ├── locomo_retrieval_summary_wiki.jsonl
│   ├── locomo_retrieval_wiki.jsonl
│   └── locomo_retrieval_raw.jsonl
├── longmemeval_processed.json
└── longmemeval/
    ├── longmemeval_wiki.jsonl
    ├── longmemeval_summary_wiki.jsonl
    ├── longmemeval_retrieval_summary_wiki.jsonl
    ├── longmemeval_retrieval_wiki.jsonl
    └── longmemeval_retrieval_raw.jsonl
```

## 1. Processed datasets

### LoCoMo — `locomo_processed.json`

A JSON list with one record per conversation:

| Field | Used for |
|---|---|
| `sample_id` | conversation id, e.g. `conv-26` |
| `sessions` | that conversation's sessions |
| `qa` | questions, reference answers, evidence, and categories |

Each session needs `wiki_id`, `source_session_id`, `date_time` and `transcript_text`.

Queries come from the retrieval file rather than the dataset, so every retrieval row must
carry `sample_id`, `query_id`, `question`, `answer`, `category` and `topk_sessions`.

Rows with `category == 5` are skipped by the LoCoMo readers and judge, preserving the supplied experiment scope.

### LongMemEval — `longmemeval_processed.json`

A JSON list with one record per question:

| Field | Used for |
|---|---|
| `question_id` | query key |
| `question_type`, `question_date`, `answer`, `answer_session_ids` | copied into the output |
| `sessions` | that question's own haystack (same per-session fields as above) |

Every question has its own haystack and `wiki_id` never crosses questions, so the wiki
lookup is scoped per question rather than global.

## 2. Derived files

| File | Contents |
|---|---|
| `<name>_wiki.jsonl` | one wiki entry per session |
| `<name>_retrieval_wiki.jsonl` | per query: top-k sessions ranked by **query × wiki** similarity |
| `<name>_retrieval_raw.jsonl` | per query: top-k sessions ranked by **query × raw session** similarity |

Retrieval rows look like:

```json
{"sample_id": "conv-26", "query_id": "conv-26_q1", "question": "...",
 "answer": "...", "category": 1,
 "topk_sessions": [{"source_session_id": "conv-26_3", "wiki_id": "conv-26_session_3", "score": 0.675}]}
```

Wiki entries carry the fields rendered into the controller's context: `wiki_id`,
`source_session_id`, `source_session_date_time`, `summary`, `event_core`, `atomic_facts`,
`entities`, `temporal_clues`, `resolved_temporal_facts`, `ongoing_threads`. LongMemEval
entries additionally carry `question_id`.

Which retrieval file is used depends on the read mode:

| `--read-mode` | Retrieval file |
|---|---|
| `full`, `wiki-only` | `*_retrieval_wiki.jsonl` |
| `raw-only` | `*_retrieval_raw.jsonl` |

All derived files are produced by this repository:

```bash
bash scripts/run_build_wiki.sh <dataset>              # -> <name>_wiki.jsonl
bash scripts/run_retrieval.sh <dataset> wiki          # -> <name>_retrieval_wiki.jsonl
bash scripts/run_retrieval.sh <dataset> raw           # -> <name>_retrieval_raw.jsonl
```

The wiki depends only on the LLM used to build it; the retrieval files depend only on the
embedding model and the dataset. Record the exact models and settings used to build them. Pre-built files can be reused independently of the model used to read them, but LLM-generated indexes may vary across endpoint versions and runs.
