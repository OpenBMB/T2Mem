#!/usr/bin/env bash
# Evaluate T2Mem predictions with benchmark-specific LLM judges.
# Usage: bash scripts/run_eval.sh <locomo|longmemeval> <predictions.jsonl> [extra args]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
case "${1:-}" in
  -h|--help|"") echo "Usage: bash scripts/run_eval.sh <locomo|longmemeval> <predictions.jsonl> [extra args]"; exit 0 ;;
esac
select_dataset "$1"
PRED="${2:?missing predictions path}"
shift 2
[ -f "$PRED" ] || fail "predictions file does not exist: $PRED"
PRED="$(cd "$(dirname "$PRED")" && pwd)/$(basename "$PRED")"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/eval-results}"
mkdir -p "$OUT_DIR"
STEM="judge_$(basename "${PRED%.jsonl}")"
JUDGE="$REPO_ROOT/evaluation/${DATASET}_judge.py"
if [ "$DATASET" = locomo ]; then
  INPUT="$OUT_DIR/${STEM}_input.json"
  # Preserve the original LoCoMo runner's failed-prediction filtering.
  "$PYTHON" "$REPO_ROOT/evaluation/t2mem_jsonl_to_locomo_judge.py" \
    --input-path "$PRED" --output-path "$INPUT" --skip-errors
  EXTRA=(--model-name "${JUDGE_MODEL:-qwen3.5-plus}")
else
  INPUT="$PRED"
  EXTRA=(--data-path "$DATA" --model-name "${JUDGE_MODEL:-qwen3.5-plus}")
fi
"$PYTHON" "$JUDGE" --input-path "$INPUT" \
  --output-path "$OUT_DIR/${STEM}.jsonl" --summary-path "$OUT_DIR/${STEM}.summary.json" \
  "${EXTRA[@]}" "$@"
echo "Judge summary: $OUT_DIR/${STEM}.summary.json"
