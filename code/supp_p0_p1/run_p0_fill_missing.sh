#!/usr/bin/env bash
# Fill missing n781 identities for one or more P0 result dirs (keeps completed tasks).
# Env:
#   NAME=p0_full_geo_s1          # single dir
#   NAMES="p0_full_geo_s1 ..."   # space-separated list
#   If neither set: all p0_full_* dirs under RESULTS
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
RESULTS="${RESULTS:-$BASE/results/icra_p0}"
MANIFEST="${MANIFEST:-$ABL/freeze/TASK_MANIFEST.json}"
FILL_SH="$BASE/code/recredit_r1/fill_v27_n781_missing.sh"
mkdir -p "$RESULTS" "$RESULTS/logs"

if [[ -n "${NAME:-}" ]]; then
  NAMES=("$NAME")
elif [[ -n "${NAMES:-}" ]]; then
  # shellcheck disable=SC2206
  NAMES=($NAMES)
else
  NAMES=()
  for d in "$RESULTS"/p0_full_*; do
    [[ -d "$d" ]] || continue
    NAMES+=("$(basename "$d")")
  done
fi

echo "==== $(date -Is) P0 fill-missing start n=${#NAMES[@]} ===="
for name in "${NAMES[@]}"; do
  dir="$RESULTS/$name"
  [[ -d "$dir" ]] || { echo "SKIP missing dir $dir"; continue; }
  if [[ ! "$name" =~ ^p0_full_(geo|u2)_s([0-9]+)(_r[0-9]+)?$ ]]; then
    echo "SKIP unrecognized name=$name"
    continue
  fi
  arm="${BASH_REMATCH[1]}"
  seed="${BASH_REMATCH[2]}"
  model="$BASE/ckpts/icra_p0/full_${arm}_s${seed}/best"
  [[ -f "$model/config.json" ]] || { echo "FATAL missing $model"; exit 1; }

  n=$("$PY" - <<PY
from pathlib import Path
import json
ids=set()
for p in Path("$dir").rglob("result.json"):
    try:
        ids.add(str(json.loads(p.read_text()).get("identity")))
    except Exception:
        pass
print(len(ids))
PY
)
  if [[ "$n" -ge 781 ]]; then
    echo "==== $name already n=$n; strict rescore only ===="
  else
    echo "==== fill $name have=$n/781 model=$model ===="
    tmp=$(mktemp)
    sed 's/^MODEL_TYPE=.*/MODEL_TYPE=qwen3_vl/' "$FILL_SH" > "$tmp"
    chmod +x "$tmp"
    set +e
    EVAL_NAME="$name" EVAL_MODEL="$model" RESULTS="$RESULTS" bash "$tmp"
    rc=$?
    set -e
    rm -f "$tmp"
    [[ "$rc" -eq 0 ]] || { echo "FATAL fill failed $name rc=$rc"; exit 1; }
  fi

  "$PY" "$ABL/code/ablations/score_n781_strict.py" \
    --results-dir "$dir" \
    --manifest "$MANIFEST" \
    --name "$name" \
    --model "$model" \
    --out-json "$dir/N781_SR.json" \
    --out-task-csv "$dir/task_success.csv"
  echo "==== scored $name ===="
  cat "$dir/N781_SR.json"
done
echo "==== $(date -Is) P0 fill-missing finished ===="
