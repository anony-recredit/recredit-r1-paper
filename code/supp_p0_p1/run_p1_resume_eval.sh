#!/usr/bin/env bash
# Resume P1 n781 evals: fill missing on partial dirs, skip completed, continue queue.
# Default REPS=1 2 (matches the interrupted waiter). set -uo so one failure does not abort.
set -uo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
P1_ROOT="${P1_ROOT:?set P1_ROOT}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
RESULTS="${RESULTS:-$BASE/results/icra_p1}"
MANIFEST="$ABL/freeze/TASK_MANIFEST.json"
FILL_SH="$BASE/code/recredit_r1/fill_v27_n781_missing.sh"
REPS=(${REPS:-1 2})
SEEDS=(${SEEDS:-1 17 31})
ARMS=(${ARMS:-attr type_agnostic})
export MAX_SHARDS="${MAX_SHARDS:-8}"
mkdir -p "$RESULTS" "$RESULTS/logs"

have_n() {
  local d="$1"
  [[ -d "$d" ]] || { echo 0; return; }
  "$PY" - <<PY
from pathlib import Path
print(len(list(Path("$d").rglob("result.json"))))
PY
}

strict_score() {
  local name="$1" arm="$2" seed="$3"
  local model="$P1_ROOT/ckpts/p1_${arm}_s${seed}/best"
  "$PY" "$ABL/code/ablations/score_n781_strict.py" \
    --results-dir "$RESULTS/$name" \
    --manifest "$MANIFEST" \
    --name "$name" --model "$model" \
    --out-json "$RESULTS/$name/N781_SR.json" \
    --out-task-csv "$RESULTS/$name/task_success.csv"
  echo "==== scored $name ===="
  cat "$RESULTS/$name/N781_SR.json"
}

fill_one() {
  local name="$1" arm="$2" seed="$3"
  local model="$P1_ROOT/ckpts/p1_${arm}_s${seed}/best"
  echo "==== $(date -Is) fill-missing $name ===="
  tmp=$(mktemp)
  sed 's/^MODEL_TYPE=.*/MODEL_TYPE=qwen3_vl/' "$FILL_SH" > "$tmp"
  chmod +x "$tmp"
  EVAL_NAME="$name" EVAL_MODEL="$model" RESULTS="$RESULTS" bash "$tmp"
  local rc=$?
  rm -f "$tmp"
  return "$rc"
}

eval_one() {
  local arm="$1" seed="$2" rep="$3"
  local name="p1_${arm}_s${seed}_r${rep}"
  local model="$P1_ROOT/ckpts/p1_${arm}_s${seed}/best"
  local n

  [[ -f "$model/config.json" ]] || { echo "FATAL missing $model"; return 1; }

  if [[ -f "$RESULTS/$name/N781_SR.json" ]]; then
    local completed
    completed=$("$PY" -c "import json; print(json.load(open('$RESULTS/$name/N781_SR.json')).get('n_completed',0))")
    if [[ "$completed" == "781" ]]; then
      echo "==== $(date -Is) SKIP done $name ===="
      return 0
    fi
  fi

  n=$(have_n "$RESULTS/$name")
  if [[ "$n" -ge 781 ]]; then
    echo "==== $(date -Is) have $n results; strict score $name ===="
    strict_score "$name" "$arm" "$seed" && return 0
    echo "WARN strict failed with n=$n; attempting fill"
    fill_one "$name" "$arm" "$seed" || true
    strict_score "$name" "$arm" "$seed" || return 1
    return 0
  fi

  if [[ "$n" -gt 0 ]]; then
    echo "==== $(date -Is) PARTIAL $name have=$n/781 -> fill ===="
    fill_one "$name" "$arm" "$seed" || true
    n=$(have_n "$RESULTS/$name")
    if [[ "$n" -ge 781 ]]; then
      strict_score "$name" "$arm" "$seed" || return 1
      return 0
    fi
    echo "WARN still incomplete after fill $name n=$n; full re-eval"
  fi

  echo "==== $(date -Is) FULL eval $name ===="
  ARM="$arm" SEED="$seed" REP="$rep" RESULTS="$RESULTS" \
    bash "$ABL/code/ablations/run_p1_eval_n781.sh"
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "WARN eval/score failed $name rc=$rc; fill+rescore"
    fill_one "$name" "$arm" "$seed" || true
    strict_score "$name" "$arm" "$seed" || return 1
  fi
  return 0
}

echo "==== $(date -Is) P1 resume-eval start REPS=${REPS[*]} ===="
FAIL=0
for rep in "${REPS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
      if ! eval_one "$arm" "$seed" "$rep"; then
        echo "ERROR failed $arm s$seed r$rep"
        FAIL=1
      fi
    done
  done
done

echo "==== $(date -Is) aggregate if possible ===="
set +e
RESULTS="$RESULTS" bash "$ABL/code/ablations/run_p1_aggregate_stats.sh"
set -e

echo "==== $(date -Is) P1 resume-eval finished FAIL=$FAIL ===="
exit "$FAIL"
