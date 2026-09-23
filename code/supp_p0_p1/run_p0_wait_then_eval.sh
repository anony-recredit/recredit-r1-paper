#!/usr/bin/env bash
# Wait for an already-running P0 train PID, then run eval-all (for mid-flight attach).
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
TRAIN_PID="${TRAIN_PID:?set TRAIN_PID=...}"
LOGDIR="${LOGDIR:-$BASE/logs/icra_p0}"
mkdir -p "$LOGDIR"

echo "==== $(date -Is) waiting for TRAIN_PID=$TRAIN_PID ===="
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done
echo "==== $(date -Is) TRAIN_PID=$TRAIN_PID exited; starting eval-all ===="
# If train crashed without ckpts, eval-all will fail loudly.
bash "$ABL/code/ablations/run_p0_eval_all.sh"
echo "==== $(date -Is) wait+eval done ===="
