# Data

Datasets are **not** shipped with this repository. Obtain them from the benchmark authors:

- [LoCoMo repository](https://github.com/snap-research/locomo)
- [LongMemEval repository](https://github.com/xiaowu0162/LongMemEval) and [cleaned dataset release](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)

From the repository root:

```bash
mkdir -p data/raw
curl -fL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o data/raw/locomo10.json
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  -o data/raw/longmemeval_s_cleaned.json
bash scripts/preprocess_data.sh locomo
bash scripts/preprocess_data.sh longmemeval
```

Keep the downloaded version or checksum with your experiment record. The repository's MIT license does not replace the benchmark's data terms.

By default the runner scripts look under `data/` using the layout below; override
`DATA_ROOT` (or the individual `DATA` / `WIKI_ROOT` / `RETRIEVAL` / `WIKI` variables)
to point somewhere else.

```
data/
├── raw/                       raw benchmark files (not distributed)
│   ├── locomo10.json
│   └── longmemeval_s_cleaned.json
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
