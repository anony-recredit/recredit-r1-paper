#!/usr/bin/env bash
# Fill+strict-score every incomplete p0_full_* / p1_* result dir under RESULTS roots.
set -uo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
P1_ROOT="${P1_ROOT:?set P1_ROOT}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
MANIFEST="$ABL/freeze/TASK_MANIFEST.json"
FILL_SH="$BASE/code/recredit_r1/fill_v27_n781_missing.sh"
export MAX_SHARDS="${MAX_SHARDS:-8}"

is_strict_done() {
  local sr="$1"
  [[ -f "$sr" ]] || return 1
  "$PY" - <<PY
import json, sys
m=json.load(open("$sr"))
sys.exit(0 if m.get("policy")=="strict_manifest_metrics_success" and m.get("n_completed")==781 else 1)
PY
}

fill_and_score() {
  local results_root="$1" name="$2" model="$3"
  local dir="$results_root/$name"
  [[ -d "$dir" ]] || return 0
  if is_strict_done "$dir/N781_SR.json"; then
    echo "==== SKIP strict-done $name ===="
    return 0
  fi
  local n
  n=$("$PY" -c "from pathlib import Path; print(len(list(Path('$dir').rglob('result.json'))))")
  echo "==== $(date -Is) fill/rescore $name have=$n model=$model ===="
  if [[ ! -f "$model/config.json" ]]; then
    echo "WARN missing model $model; skip $name"
    return 1
  fi
  if [[ "$n" -lt 781 ]]; then
    tmp=$(mktemp)
    sed 's/^MODEL_TYPE=.*/MODEL_TYPE=qwen3_vl/' "$FILL_SH" > "$tmp"
    chmod +x "$tmp"
    EVAL_NAME="$name" EVAL_MODEL="$model" RESULTS="$results_root" bash "$tmp" || echo "WARN fill rc=$? $name"
    rm -f "$tmp"
  fi
  "$PY" "$ABL/code/ablations/score_n781_strict.py" \
    --results-dir "$dir" \
    --manifest "$MANIFEST" \
    --name "$name" --model "$model" \
    --out-json "$dir/N781_SR.json" \
    --out-task-csv "$dir/task_success.csv" \
    && cat "$dir/N781_SR.json" \
    || echo "WARN strict failed $name"
}

echo "==== $(date -Is) fill-all-incomplete start ===="

# P0
for d in "$BASE/results/icra_p0"/p0_full_*; do
  [[ -d "$d" ]] || continue
  name=$(basename "$d")
  if [[ "$name" =~ ^p0_full_(geo|u2)_s([0-9]+)(_r[0-9]+)?$ ]]; then
    arm="${BASH_REMATCH[1]}"
    seed="${BASH_REMATCH[2]}"
    model="$BASE/ckpts/icra_p0/full_${arm}_s${seed}/best"
    fill_and_score "$BASE/results/icra_p0" "$name" "$model"
  fi
done

# P1
for d in "$BASE/results/icra_p1"/p1_*; do
  [[ -d "$d" ]] || continue
  name=$(basename "$d")
  if [[ "$name" =~ ^p1_(attr|type_agnostic)_s([0-9]+)_r([0-9]+)$ ]]; then
    arm="${BASH_REMATCH[1]}"
    seed="${BASH_REMATCH[2]}"
    model="$P1_ROOT/ckpts/p1_${arm}_s${seed}/best"
    fill_and_score "$BASE/results/icra_p1" "$name" "$model"
  fi
done

echo "==== $(date -Is) fill-all-incomplete finished ===="
