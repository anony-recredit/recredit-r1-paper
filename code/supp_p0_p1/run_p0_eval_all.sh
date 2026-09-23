#!/usr/bin/env bash
# After P0 train: n781 eval + strict score for each Full arm×seed, then Geo vs U2 stats.
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
SEEDS=(${SEEDS:-1 17 31})
ARMS=(${ARMS:-geo u2})
RESULTS="${RESULTS:-$BASE/results/icra_p0}"
LOGDIR="${LOGDIR:-$BASE/logs/icra_p0}"
mkdir -p "$RESULTS" "$LOGDIR"

echo "==== $(date -Is) P0 eval-all start ===="
for SEED in "${SEEDS[@]}"; do
  for ARM in "${ARMS[@]}"; do
    MODEL="$BASE/ckpts/icra_p0/full_${ARM}_s${SEED}/best"
    [[ -f "$MODEL/config.json" ]] || { echo "FATAL missing $MODEL"; exit 1; }
    echo "======== eval ARM=$ARM SEED=$SEED model=$MODEL ========"
    ARM="$ARM" SEED="$SEED" MODEL="$MODEL" RESULTS="$RESULTS" \
      bash "$ABL/code/ablations/run_p0_eval_n781.sh"
  done
done

RESULTS="$RESULTS" bash "$ABL/code/ablations/run_p0_aggregate_stats.sh"
echo "==== $(date -Is) P0 eval-all finished ===="
cat "$RESULTS/P0_GEO_VS_U2_STATS.json"
