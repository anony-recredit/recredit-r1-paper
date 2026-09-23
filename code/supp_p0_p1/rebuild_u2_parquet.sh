#!/usr/bin/env bash
# Rebuild U2 parquet from Geo using REAL perc/reas+span token mass (Qwen3 tokenizer).
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
SRC="${SRC:-$BASE/data/parquet_p123_shared/P3_recredit_p25}"
OUT="${OUT:-$BASE/data/parquet_p123_shared/P3_recredit_p25_u2_mass}"
TOK="${TOK:-$BASE/models/Qwen3-VL-8B-Instruct}"

# retire broken q_t-matched build if present
if [[ -d "$OUT" ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  mv "$OUT" "${OUT}.BAD_q_t_match_${ts}"
  echo "moved broken U2 dir -> ${OUT}.BAD_q_t_match_${ts}"
fi

"$PY" "$ABL/code/ablations/build_mass_matched_uniform_stage1_parquet.py" \
  --src-dir "$SRC" \
  --out-dir "$OUT" \
  --tokenizer "$TOK" \
  --atol 1e-5

echo "==== U2 rebuild OK ===="
cat "$OUT/U2_BUILD_REPORT.json"
