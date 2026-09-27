# Contributing

Use Python 3.10 or newer and install `requirements.txt`. Add `requirements-local.txt` only when working with local embeddings.

Keep changes scoped to their pipeline stage. Shared configuration/orchestration and embedding adapters live in `src/common`; benchmark modules own prompts, text representations, and research logic. The LoCoMo and LongMemEval trees are independent CLI implementations; run each in its own process because they use dataset-local `utils`, `prompt`, and `t2mem` imports.

Before submitting a change:

```bash
python -m unittest discover -s tests -v
python -m compileall -q run.py src evaluation tests
```

Offline integration tests use a mock OpenAI-compatible server and a local embedding stub. A change to model behavior or scoring also needs an actual experiment with the relevant dataset and endpoint; include the configuration and limitations with its results.

Use focused Conventional Commits, for example:

```text
feat(indexing): add an episode-index field
fix(inference): resolve raw-session IDs for LongMemEval
fix(evaluation): retain failed predictions in judge input
docs(usage): clarify benchmark reproduction settings
```

Do not commit API keys, `.env`, datasets, model weights, generated predictions, logs, or benchmark outputs. Do not change prompts or evaluation conventions as part of a cosmetic cleanup. Document intentional changes to reproduction settings or scoring behavior.
