#!/usr/bin/env bash
# P0 scheduler: for each seed in {1,17,31}: Geo-S1 → Full, U2-S1 → Full.
# Independent output dirs under ckpts/icra_p0/. Same Full CE parquet for both arms.
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
SEEDS=(${SEEDS:-1 17 31})
NPROC="${NPROC:-8}"

for SEED in "${SEEDS[@]}"; do
  echo "======== P0 seed=$SEED Geo ========"
  ARM=geo SEED="$SEED" NPROC="$NPROC" bash "$ABL/code/ablations/run_p0_stage1.sh"
  ARM=geo SEED="$SEED" NPROC="$NPROC" bash "$ABL/code/ablations/run_p0_stage2_full.sh"
  echo "======== P0 seed=$SEED U2 ========"
  ARM=u2 SEED="$SEED" NPROC="$NPROC" bash "$ABL/code/ablations/run_p0_stage1.sh"
  ARM=u2 SEED="$SEED" NPROC="$NPROC" bash "$ABL/code/ablations/run_p0_stage2_full.sh"
done
echo "==== P0 all seeds train finished ===="

# Auto-chain n781 eval for each Full ckpt + paired Geo vs U2 stats.
echo "==== $(date -Is) chaining P0 eval-all ===="
bash "$ABL/code/ablations/run_p0_eval_all.sh"
echo "==== P0 train+eval pipeline finished ===="
