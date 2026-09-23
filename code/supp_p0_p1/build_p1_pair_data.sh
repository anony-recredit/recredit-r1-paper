#!/usr/bin/env bash
set -euo pipefail

BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
P1_ROOT="${P1_ROOT:?set P1_ROOT}"
PY="${PYTHON:-$BASE/miniconda3/envs/recredit/bin/python}"
OUT="${PAIR_OUT:-$P1_ROOT/data/pairs}"

"$PY" "$ABL/code/ablations/build_p1_pairs.py" \
  --source-freeze "$ABL/freeze/P1_SOURCES.json" \
  --out-dir "$OUT"

"$PY" "$ABL/code/ablations/audit_p1_pairs.py" \
  --attr "$OUT/attr_unweighted.parquet" \
  --type-agnostic "$OUT/type_agnostic_unweighted.parquet" \
  --candidate-pool "$OUT/candidates_sanitized.parquet" \
  --source-freeze "$ABL/freeze/P1_SOURCES.json" \
  --forbidden-eval "$BASE/data/eval/test_C_105.json" \
  --forbidden-eval /data1/dataset/embodied_reasoner/test_809.json \
  --forbidden-eval "$BASE/data/eval/test_809_n806_no_emulator3.json" \
  --require-images \
  --out "$OUT/P1_PAIR_MANIFEST.json"
