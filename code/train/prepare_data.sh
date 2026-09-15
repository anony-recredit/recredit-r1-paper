#!/usr/bin/env bash
# Unpack Recredit-R1 pack and convert to verl parquet (Stage I / Stage II).
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
CONDA_ROOT="$BASE/miniconda3"
RECIPE="$BASE/code/recredit_r1"
PACK_TGZ="${1:-}"
PACK_DIR="$BASE/data/raw/recredit_r1_er_autofill_20260824_121338"
JSON="$PACK_DIR/annotations/recredit_r1_autofill_local.json"
OUT="$BASE/data/parquet"

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONPATH="$RECIPE:${PYTHONPATH:-}"
cd "$RECIPE"

if [[ -n "$PACK_TGZ" ]]; then
  mkdir -p "$BASE/data/raw"
  tar -xzf "$PACK_TGZ" -C "$BASE/data/raw"
fi

if [[ ! -f "$JSON" ]]; then
  echo "missing $JSON" >&2
  exit 1
fi

python convert_recredit_to_parquet.py \
  --json "$JSON" \
  --data-root "$PACK_DIR" \
  --out-dir "$OUT" \
  --stage stage1 \
  --max-horizon 8 \
  --val-frac 0.2 \
  --seed 1

python convert_recredit_to_parquet.py \
  --json "$JSON" \
  --data-root "$PACK_DIR" \
  --out-dir "$OUT" \
  --stage stage2 \
  --val-frac 0.2 \
  --seed 1

ls -lh "$OUT"
