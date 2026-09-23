#!/usr/bin/env bash
# Wait until the in-flight single eval chain exits, then:
#   1) fill missing n781 ids on chain results + strict score
#   2) run 3-replicate evals
#   3) fill missing on x3 results + strict score
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
CHAIN_PID="${CHAIN_PID:?set CHAIN_PID to the running run_p0_eval_all.sh pid}"
RESULTS="${RESULTS:-$BASE/results/icra_p0}"
LOGDIR="${LOGDIR:-$BASE/logs/icra_p0}"
mkdir -p "$LOGDIR"

echo "==== $(date -Is) waiting for eval chain PID=$CHAIN_PID ===="
while kill -0 "$CHAIN_PID" 2>/dev/null; do
  sleep 60
done
while pgrep -f '/run_eval_qwen3vl_n781.sh' >/dev/null; do
  echo "==== $(date -Is) chain pid gone but n781 eval still running; waiting ===="
  sleep 30
done

echo "==== $(date -Is) chain done; fill missing on single-chain results ===="
NAMES="p0_full_geo_s1 p0_full_geo_s17 p0_full_geo_s31 p0_full_u2_s1 p0_full_u2_s17 p0_full_u2_s31" \
  RESULTS="$RESULTS" bash "$ABL/code/ablations/run_p0_fill_missing.sh"

echo "==== $(date -Is) starting 3-rep evals ===="
bash "$ABL/code/ablations/run_p0_eval_x3.sh"

echo "==== $(date -Is) fill missing on x3 results ===="
X3_NAMES=()
for arm in geo u2; do
  for seed in 1 17 31; do
    for rep in 1 2 3; do
      X3_NAMES+=("p0_full_${arm}_s${seed}_r${rep}")
    done
  done
done
NAMES="${X3_NAMES[*]}" RESULTS="$RESULTS" bash "$ABL/code/ablations/run_p0_fill_missing.sh"

echo "==== $(date -Is) wait+fill+x3+fill done ===="
