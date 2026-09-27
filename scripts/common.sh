#!/usr/bin/env bash
# Shared paths and validation for the experiment runners.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"

fail() { echo "ERROR: $*" >&2; exit 2; }
select_dataset() {
  DATASET="$1"
  case "$DATASET" in
    locomo|longmemeval) ;;
    *) fail "unknown dataset '$DATASET'; choose locomo or longmemeval" ;;
  esac
  SRC_DIR="$REPO_ROOT/src/$DATASET"
  DATA="${DATA:-$DATA_ROOT/${DATASET}_processed.json}"
  WIKI_ROOT="${WIKI_ROOT:-$DATA_ROOT/$DATASET}"
}
select_variant() {
  VARIANT="${VARIANT:-structured}"
  case "$VARIANT" in
    structured) WIKI_SUFFIX=wiki ;;
    summary) WIKI_SUFFIX=summary_wiki ;;
    *) fail "unknown VARIANT '$VARIANT'; choose structured or summary" ;;
  esac
  WIKI="${WIKI:-$WIKI_ROOT/${DATASET}_${WIKI_SUFFIX}.jsonl}"
}
configure_llm() {
  BASE_URL="${BASE_URL:-${T2MEM_LLM_BASE_URL:-http://127.0.0.1:8010/v1}}"
  API_KEY="${API_KEY:-${T2MEM_LLM_API_KEY:-}}"
  MODEL="${MODEL:-${T2MEM_LLM_MODEL_NAME:-Qwen3.5-27B}}"
}
