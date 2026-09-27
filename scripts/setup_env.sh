#!/bin/bash
# Create (or verify) the conda environment required by T2Mem.
#
#   bash scripts/setup_env.sh              # create the env if missing
#   bash scripts/setup_env.sh --check      # only verify, do not create
set -euo pipefail

ENV_NAME="t2mem"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK_ONLY=0
case "${1:-}" in
  "") ;;
  --check) CHECK_ONLY=1 ;;
  -h|--help) echo "Usage: bash scripts/setup_env.sh [--check]"; exit 0 ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' already exists."
else
  if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "ERROR: environment '$ENV_NAME' does not exist (run without --check to create it)." >&2
    exit 1
  fi
  echo "Creating environment '$ENV_NAME' from environment.yml ..."
  (cd "$REPO_ROOT" && conda env create -f environment.yml)
fi

conda activate "$ENV_NAME"

echo
echo "Verifying dependencies ..."
python - <<'PY'
import sys
missing = []
for mod, pkg in (("openai", "openai"), ("tqdm", "tqdm"), ("json_repair", "json-repair"), ("numpy", "numpy")):
    try:
        __import__(mod)
    except ImportError:
        missing.append(pkg)
print(f"  python {sys.version.split()[0]}")
if missing:
    print("  MISSING: " + ", ".join(missing))
    raise SystemExit(1)
print("  openai / tqdm / json_repair / numpy  OK")
PY

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
