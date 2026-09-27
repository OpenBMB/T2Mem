#!/usr/bin/env bash
# Build episode indexes; VARIANT=summary selects the summary ablation.
# Usage: bash scripts/run_build_wiki.sh <locomo|longmemeval> [extra Python args]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
case "${1:-}" in
  -h|--help|"") echo "Usage: bash scripts/run_build_wiki.sh <locomo|longmemeval> [extra args]"; exit 0 ;;
esac
select_dataset "$1"
shift
select_variant
configure_llm
WORKERS="${WORKERS:-16}"
OUT="${OUT:-$WIKI}"
mkdir -p "$(dirname "$OUT")"
ENTRY_ARGS=(--data-path "$DATA" --output-path "$OUT")
if [ "$VARIANT" = summary ]; then
  BUILDER="$SRC_DIR/wiki/build_summary_wiki.py"
  if [ "$DATASET" = longmemeval ]; then ENTRY_ARGS+=(--dataset "$DATASET"); fi
else
  BUILDER="$SRC_DIR/wiki/build_wiki.py"
  ENTRY_ARGS+=(--dataset "$DATASET")
fi
echo "Indexing: dataset=$DATASET variant=$VARIANT model=$MODEL output=$OUT"
"$PYTHON" "$BUILDER" "${ENTRY_ARGS[@]}" \
  --base-url "$BASE_URL" --api-key "$API_KEY" --model-name "$MODEL" \
  --max-workers "$WORKERS" "$@" 2>&1 | tee "${OUT%.jsonl}.log"
