#!/usr/bin/env bash
# After P1 training finishes: 2× n781 eval per ckpt, then aggregate stats.
# Does not start while train/eval GPUs are still busy.
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
TRAIN_PID="${TRAIN_PID:?set TRAIN_PID to run_p1_all.sh}"
REPS="${REPS:-1 2}"
LOGDIR="${LOGDIR:-$BASE/logs/icra_p1}"
mkdir -p "$LOGDIR"

echo "==== $(date -Is) waiting for P1 train PID=$TRAIN_PID (then REPS=$REPS) ===="
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done
echo "==== $(date -Is) train PID=$TRAIN_PID exited; drain GPU workers ===="
while pgrep -f 'torchrun.*trainer.py|evaluate.py|local_deploy.py|run_eval_qwen3vl_n781.sh' >/dev/null 2>&1; do
  echo "still busy..."
  sleep 30
done

# require six best ckpts
for seed in 1 17 31; do
  for arm in attr type_agnostic; do
    m="${P1_ROOT}/ckpts/p1_${arm}_s${seed}/best/config.json"
    [[ -f "$m" ]] || { echo "FATAL missing $m"; exit 1; }
  done
done

echo "==== $(date -Is) starting P1 eval-all REPS=$REPS ===="
REPS="$REPS" bash "$ABL/code/ablations/run_p1_eval_all.sh"
echo "==== $(date -Is) aggregate stats REPS=$REPS ===="
REPS="$REPS" bash "$ABL/code/ablations/run_p1_aggregate_stats.sh"
echo "==== $(date -Is) P1 train→eval×2 pipeline finished ===="
ls -la "$BASE/results/icra_p1"/P1_*_STATS.json 2>/dev/null || true
