#!/usr/bin/env bash
# Usage: bash scripts/preprocess_data.sh <locomo|longmemeval> [extra Python args]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
case "${1:-}" in
  -h|--help|"") echo "Usage: bash scripts/preprocess_data.sh <locomo|longmemeval> [extra args]"; exit 0 ;;
esac
select_dataset "$1"
shift
case "$DATASET" in
  locomo) RAW_DATA="${RAW_DATA:-${T2MEM_LOCOMO_RAW:-$DATA_ROOT/raw/locomo10.json}}" ;;
  longmemeval) RAW_DATA="${RAW_DATA:-${T2MEM_LONGMEMEVAL_RAW:-$DATA_ROOT/raw/longmemeval_s_cleaned.json}}" ;;
esac
"$PYTHON" "$SRC_DIR/wiki/preprocess_data.py" --data-path "$RAW_DATA" --output-path "$DATA" "$@"
