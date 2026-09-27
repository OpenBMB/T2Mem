#!/usr/bin/env bash
# Run T2Mem test-time memory construction and answering.
# Usage: bash scripts/run_t2mem.sh <locomo|longmemeval> <full|wiki-only|raw-only> [extra args]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
case "${1:-}" in
  -h|--help|"") echo "Usage: bash scripts/run_t2mem.sh <locomo|longmemeval> <full|wiki-only|raw-only> [extra args]"; exit 0 ;;
esac
select_dataset "$1"
READ_MODE="${2:?missing read-mode}"
shift 2
case "$READ_MODE" in full|wiki-only|raw-only) ;; *) fail "unknown read-mode '$READ_MODE'" ;; esac
select_variant
configure_llm
WORKERS="${WORKERS:-8}"
TOP_K="${TOP_K:-5}"
MAX_STEPS="${MAX_STEPS:-5}"
ENTRY_ARGS=(--top-k "$TOP_K")
if [ "$VARIANT" = summary ]; then
  [ "$READ_MODE" = full ] || fail "summary entry points support full mode only"
  ENTRY="$SRC_DIR/t2mem/t2mem_summary_answer.py"
  RETRIEVAL="${RETRIEVAL:-$WIKI_ROOT/${DATASET}_retrieval_summary_wiki.jsonl}"
  if [ "$DATASET" = longmemeval ]; then ENTRY_ARGS+=(--dataset "$DATASET"); fi
else
  ENTRY="$SRC_DIR/t2mem/t2mem_answer.py"
  if [ "$READ_MODE" = raw-only ]; then RETRIEVAL_UNIT=raw; else RETRIEVAL_UNIT=wiki; fi
  RETRIEVAL="${RETRIEVAL:-$WIKI_ROOT/${DATASET}_retrieval_${RETRIEVAL_UNIT}.jsonl}"
  ENTRY_ARGS+=(--dataset "$DATASET" --read-mode "$READ_MODE")
fi
TAG="${READ_MODE//-/_}"
OUT="${OUT:-$REPO_ROOT/outputs/${DATASET}_${VARIANT}_t2mem_${TAG}_answers.jsonl}"
mkdir -p "$(dirname "$OUT")"
echo "T2Mem: dataset=$DATASET variant=$VARIANT read-mode=$READ_MODE model=$MODEL output=$OUT"
"$PYTHON" "$ENTRY" "${ENTRY_ARGS[@]}" \
  --data-path "$DATA" --retrieval-path "$RETRIEVAL" --wiki-path "$WIKI" --output-path "$OUT" \
  --base-url "$BASE_URL" --api-key "$API_KEY" --model-name "$MODEL" \
  --max-workers "$WORKERS" --max-steps "$MAX_STEPS" \
  "$@" 2>&1 | tee "${OUT%.jsonl}.log"
