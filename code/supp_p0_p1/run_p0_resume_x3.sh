#!/usr/bin/env bash
# Resume P0 3-rep queue after stall:
#   1) fill missing on partial dirs (no wipe)
#   2) run remaining full evals; on failure fill+continue (do not abort queue)
#   3) final fill+strict score + X3 summary
set -uo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
RESULTS="${RESULTS:-$BASE/results/icra_p0}"
LOGDIR="${LOGDIR:-$BASE/logs/icra_p0}"
MANIFEST="${MANIFEST:-$ABL/freeze/TASK_MANIFEST.json}"
export MAX_SHARDS="${MAX_SHARDS:-8}"
SEEDS=(${SEEDS:-1 17 31})
ARMS=(${ARMS:-geo u2})
REPS=(${REPS:-1 2})
mkdir -p "$RESULTS" "$LOGDIR"

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
  local model="$BASE/ckpts/icra_p0/full_${arm}_s${seed}/best"
  "$PY" "$ABL/code/ablations/score_n781_strict.py" \
    --results-dir "$RESULTS/$name" \
    --manifest "$MANIFEST" \
    --name "$name" \
    --model "$model" \
    --out-json "$RESULTS/$name/N781_SR.json" \
    --out-task-csv "$RESULTS/$name/task_success.csv"
  echo "==== scored $name ===="
  cat "$RESULTS/$name/N781_SR.json"
}

fill_one() {
  local name="$1"
  echo "==== $(date -Is) fill-missing $name ===="
  NAME="$name" RESULTS="$RESULTS" bash "$ABL/code/ablations/run_p0_fill_missing.sh" || {
    echo "WARN fill failed $name rc=$?; continue"
    return 1
  }
}

eval_one() {
  local arm="$1" seed="$2" rep="$3"
  local name="p0_full_${arm}_s${seed}_r${rep}"
  local model="$BASE/ckpts/icra_p0/full_${arm}_s${seed}/best"
  local n
  n=$(have_n "$RESULTS/$name")
  if [[ "$n" -ge 781 ]]; then
    echo "==== $(date -Is) SKIP done $name n=$n ===="
    [[ -f "$RESULTS/$name/N781_SR.json" ]] || strict_score "$name" "$arm" "$seed"
    return 0
  fi
  if [[ "$n" -gt 0 ]]; then
    echo "==== $(date -Is) PARTIAL $name have=$n/781 -> fill only ===="
    fill_one "$name"
    n=$(have_n "$RESULTS/$name")
    if [[ "$n" -ge 781 ]]; then
      return 0
    fi
    echo "WARN still incomplete after fill $name n=$n; leave for final pass"
    return 1
  fi
  echo "==== $(date -Is) FULL eval $name ===="
  set +e
  ARM="$arm" SEED="$seed" REP="$rep" MODEL="$model" RESULTS="$RESULTS" \
    bash "$ABL/code/ablations/run_p0_eval_n781.sh"
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "WARN eval failed $name rc=$rc; attempting fill-missing"
    fill_one "$name" || true
    return "$rc"
  fi
  return 0
}

echo "==== $(date -Is) P0 resume x3 start ===="

# Drop stale lock if any
rm -f "$LOGDIR/wait_chain_then_x3.lock"

# Phase 1: known partial first
fill_one "p0_full_u2_s1_r1" || true

# Phase 2: remaining queue in same order as run_p0_eval_x3.sh
FAILS=()
for REP in "${REPS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for ARM in "${ARMS[@]}"; do
      if ! eval_one "$ARM" "$SEED" "$REP"; then
        FAILS+=("p0_full_${ARM}_s${SEED}_r${REP}")
      fi
    done
  done
done

# Phase 3: final fill+score all x3 dirs
echo "==== $(date -Is) final fill-missing pass ===="
X3_NAMES=()
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for rep in "${REPS[@]}"; do
      X3_NAMES+=("p0_full_${arm}_s${seed}_r${rep}")
    done
  done
done
NAMES="${X3_NAMES[*]}" RESULTS="$RESULTS" bash "$ABL/code/ablations/run_p0_fill_missing.sh" || true

# Phase 4: summary (only rows with all 3 N781_SR)
"$PY" - <<PY
import json
from pathlib import Path
results = Path("$RESULTS")
reps = [int(x) for x in "${REPS[*]}".split()]
seeds = "${SEEDS[*]}".split()
arms = "${ARMS[*]}".split()
rows = []
incomplete = []
for arm in arms:
    for seed in seeds:
        srs = []
        paths = []
        ok = True
        for rep in reps:
            name = f"p0_full_{arm}_s{seed}_r{rep}"
            p = results / name / "N781_SR.json"
            if not p.exists():
                ok = False
                incomplete.append(name)
                continue
            meta = json.loads(p.read_text())
            srs.append(float(meta["sr_pct"]))
            paths.append(str(p))
        if ok and srs:
            rows.append({
                "arm": arm,
                "seed": int(seed),
                "reps": reps,
                "sr_pct": srs,
                "sr_mean": round(sum(srs) / len(srs), 4),
                "n781_sr": paths,
            })
out = {
    "protocol": "each Full ckpt evaluated 3 times on frozen n781; mean of strict SR",
    "rows": rows,
    "incomplete": incomplete,
    "eval_failures": """${FAILS[*]}""".split(),
}
dest = results / "P0_EVAL_X3_SUMMARY.json"
dest.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
print(f"wrote {dest}")
PY

echo "==== $(date -Is) P0 resume x3 finished fails=${FAILS[*]:-none} ===="
