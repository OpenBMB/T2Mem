#!/usr/bin/env bash
# Build candidate lists with API or optional local embeddings.
# Usage: bash scripts/run_retrieval.sh <locomo|longmemeval> <wiki|raw> [extra args]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
case "${1:-}" in
  -h|--help|"") echo "Usage: bash scripts/run_retrieval.sh <locomo|longmemeval> <wiki|raw> [extra args]"; exit 0 ;;
esac
select_dataset "$1"
UNIT="${2:?missing retrieval unit (wiki|raw)}"
shift 2
case "$UNIT" in wiki|raw) ;; *) fail "unknown unit '$UNIT'; choose wiki or raw" ;; esac
select_variant
if [ "$UNIT" = raw ] && [ "$VARIANT" = summary ]; then
  fail "summary retrieval uses wiki; select VARIANT=structured for raw retrieval"
fi
if [ "$UNIT" = wiki ]; then REPRESENTATION="$WIKI_SUFFIX"; else REPRESENTATION=raw; fi
TOP_K="${TOP_K:-10}"
OUT="${OUT:-$WIKI_ROOT/${DATASET}_retrieval_${REPRESENTATION}.jsonl}"
DEFAULT_BACKEND=local
if [ "$DATASET" = locomo ] && [ "$VARIANT" = structured ] && [ "$UNIT" = wiki ]; then
  DEFAULT_BACKEND=api
fi
EMBED_BACKEND="${EMBED_BACKEND:-$DEFAULT_BACKEND}"
EXTRA=()
case "$EMBED_BACKEND" in
  api)
    SCRIPT="$SRC_DIR/retrieval/${DATASET}_retrieval_${REPRESENTATION}_api.py"
    [ -f "$SCRIPT" ] || fail "API embeddings are supported only for structured LoCoMo wiki retrieval; use EMBED_BACKEND=local and requirements-local.txt for this combination"
    EMBED_BASE_URL="${EMBED_BASE_URL:-${T2MEM_EMBEDDING_BASE_URL:-}}"
    EMBED_API_KEY="${EMBED_API_KEY:-${T2MEM_EMBEDDING_API_KEY:-${OPENAI_API_KEY:-EMPTY}}}"
    EMBED_MODEL="${EMBED_MODEL:-${T2MEM_EMBEDDING_MODEL_NAME:-bge-m3}}"
    EXTRA+=(--base-url "$EMBED_BASE_URL" --api-key "$EMBED_API_KEY" --model-name "$EMBED_MODEL")
    ;;
  local)
    SCRIPT="$SRC_DIR/retrieval/${DATASET}_retrieval_${REPRESENTATION}.py"
    EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-${T2MEM_EMBEDDING_MODEL:-bge-m3}}"
    EXTRA+=(--model-path "$EMBED_MODEL_PATH")
    ;;
  *) fail "unknown EMBED_BACKEND '$EMBED_BACKEND'; choose api or local" ;;
esac
if [ "$UNIT" = wiki ]; then EXTRA+=(--wiki-path "$WIKI"); fi
mkdir -p "$(dirname "$OUT")"
echo "Retrieval: dataset=$DATASET representation=$REPRESENTATION backend=$EMBED_BACKEND output=$OUT"
"$PYTHON" "$SCRIPT" --data-path "$DATA" --output-path "$OUT" --top-k "$TOP_K" \
  "${EXTRA[@]}" "$@" 2>&1 | tee "${OUT%.jsonl}.log"
