#!/usr/bin/env bash
# ER multiturn → recredit_r1.v1 with validation gate.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ER_ROOT="${ER_ROOT:-${EMBODIED_REASONER_ROOT}}"
MODE="${MODE:-local}"   # local | full

if [[ ! -f "$ER_ROOT/train_multiturn_9390.json" ]]; then
  echo "missing train_multiturn_9390.json under ER_ROOT=$ER_ROOT" >&2
  exit 1
fi

# Engine package must sit next to the ER data root for `python -m recredit_data_engine.*`
if [[ ! -d "$ER_ROOT/recredit_data_engine" ]]; then
  echo "install engine -> $ER_ROOT/recredit_data_engine"
  cp -a "$SKILL_DIR/recredit_data_engine" "$ER_ROOT/"
fi

cd "$ER_ROOT"
export PYTHONPATH="$ER_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$MODE" == "full" ]]; then
  OUT="recredit_r1_autofill_full.json"
  python3 -m recredit_data_engine.batch_autofill_from_train \
    --er-root "$ER_ROOT" \
    --out "$OUT"
else
  OUT="recredit_r1_autofill_local.json"
  python3 -m recredit_data_engine.batch_autofill_local
fi

python3 -m recredit_data_engine.validate_v1 "$OUT" --er-root "$ER_ROOT"
echo "PASS: $OUT"
