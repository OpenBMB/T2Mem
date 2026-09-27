<div align="center">

# T2Mem: Test-Time Memory Construction

[![Code](https://img.shields.io/badge/Code-T2Mem-181717?logo=github&logoColor=white)](https://github.com/OpenBMB/T2Mem)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](requirements.txt)
[![License](https://img.shields.io/badge/License-MIT-22863A)](LICENSE)
[![LoCoMo](https://img.shields.io/badge/Benchmark-LoCoMo-7B61FF)](https://github.com/snap-research/locomo)
[![LongMemEval](https://img.shields.io/badge/Benchmark-LongMemEval-0078D4)](https://github.com/xiaowu0162/LongMemEval)

</div>

## 1. Install

Use Python **3.10 or newer**.

```bash
git clone https://github.com/OpenBMB/T2Mem.git
cd T2Mem
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Configure APIs

```bash
cp .env.example .env
```

Edit `.env` with the endpoints, API keys, and served model names for the **LLM**, **embedding model**, and **evaluation judge**. All three use OpenAI-compatible APIs. `run.py` loads `.env` automatically.

## 3. Prepare data

Download either or both datasets and place the files in `data/`:

| Dataset | Download | Filename |
|---|---|---|
| LoCoMo | [Official data](https://github.com/snap-research/locomo/tree/main/data) | `locomo10.json` |
| LongMemEval | [Cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | `longmemeval_s_cleaned.json` |

```text
data/
├── locomo10.json
└── longmemeval_s_cleaned.json
```

## 4. Run

Check the configuration, then run all datasets found in `data/`:

```bash
python run.py --check --temperature 0.0 --answer-max-tokens 1024
python run.py --temperature 0.0 --answer-max-tokens 1024
```

To run a single dataset, add `locomo` or `longmemeval` after `run.py`. To generate predictions without evaluation, add `--skip-eval`.

Alternatively, run the stages in order:

```bash
python run.py locomo --stage preprocess
python run.py locomo --stage index --temperature 0.0
python run.py locomo --stage retrieve
python run.py locomo --stage infer --temperature 0.0 --answer-max-tokens 1024
python run.py locomo --stage evaluate
```

Replace `locomo` with `longmemeval` for LongMemEval.

## 5. View outputs

- `outputs/`: model predictions.
- `eval-results/`: judgments and accuracy summaries.
- `data/`: processed data, indexes, and retrieval files.
