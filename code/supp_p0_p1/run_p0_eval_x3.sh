#!/usr/bin/env bash
# Three independent n781 evals per Full ckpt (geo/u2 × seeds 1/17/31).
# Writes p0_full_${ARM}_s${SEED}_r{1,2,3} so it does not clobber an in-flight single chain.
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
SEEDS=(${SEEDS:-1 17 31})
ARMS=(${ARMS:-geo u2})
REPS=(${REPS:-1 2 3})
RESULTS="${RESULTS:-$BASE/results/icra_p0}"
LOGDIR="${LOGDIR:-$BASE/logs/icra_p0}"
mkdir -p "$RESULTS" "$LOGDIR"

echo "==== $(date -Is) P0 eval x3 start reps=${REPS[*]} ===="
for REP in "${REPS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for ARM in "${ARMS[@]}"; do
      MODEL="$BASE/ckpts/icra_p0/full_${ARM}_s${SEED}/best"
      [[ -f "$MODEL/config.json" ]] || { echo "FATAL missing $MODEL"; exit 1; }
      echo "======== eval ARM=$ARM SEED=$SEED REP=$REP model=$MODEL ========"
      ARM="$ARM" SEED="$SEED" REP="$REP" MODEL="$MODEL" RESULTS="$RESULTS" \
        bash "$ABL/code/ablations/run_p0_eval_n781.sh"
    done
  done
done

"$BASE/miniconda3/envs/recredit/bin/python" - <<PY
import json
from pathlib import Path
results = Path("$RESULTS")
reps = [int(x) for x in "${REPS[*]}".split()]
seeds = "${SEEDS[*]}".split()
arms = "${ARMS[*]}".split()
rows = []
for arm in arms:
    for seed in seeds:
        srs = []
        paths = []
        for rep in reps:
            name = f"p0_full_{arm}_s{seed}_r{rep}"
            p = results / name / "N781_SR.json"
            meta = json.loads(p.read_text())
            srs.append(float(meta["sr_pct"]))
            paths.append(str(p))
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
}
dest = results / "P0_EVAL_X3_SUMMARY.json"
dest.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
print(f"wrote {dest}")
PY
echo "==== $(date -Is) P0 eval x3 finished ===="
