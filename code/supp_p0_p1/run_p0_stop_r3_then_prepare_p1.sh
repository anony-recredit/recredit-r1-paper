#!/usr/bin/env bash
# After the in-flight P0 eval job finishes: kill the r3 resume queue, then prepare P1 only.
# Does NOT flip P1_GO / does NOT start run_p1_all.sh.
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
LOGDIR="${LOGDIR:-$BASE/logs/icra_p0}"
RESUME_PID="${RESUME_PID:?}"
EVAL_PID="${EVAL_PID:?}"
mkdir -p "$LOGDIR"

echo "==== $(date -Is) freeze resume PID=$RESUME_PID; wait eval PID=$EVAL_PID ===="
if kill -0 "$RESUME_PID" 2>/dev/null; then
  kill -STOP "$RESUME_PID" || true
  echo "SIGSTOP sent to resume PID=$RESUME_PID"
else
  echo "resume PID already gone"
fi

while [[ "$EVAL_PID" != 0 ]] && kill -0 "$EVAL_PID" 2>/dev/null; do
  sleep 30
done
echo "==== $(date -Is) eval PID=$EVAL_PID exited ===="

# Stop the resume queue before it can launch r3.
if kill -0 "$RESUME_PID" 2>/dev/null; then
  kill -CONT "$RESUME_PID" 2>/dev/null || true
  sleep 1
  kill "$RESUME_PID" 2>/dev/null || true
  sleep 2
  kill -9 "$RESUME_PID" 2>/dev/null || true
  echo "killed resume PID=$RESUME_PID"
fi
# Also clear any stale wait_chain / resume helpers that could relaunch r3.
pkill -f '/run_p0_resume_x3.sh' 2>/dev/null || true
pkill -f '/run_p0_wait_chain_then_x3.sh' 2>/dev/null || true
pkill -f '/run_p0_eval_x3.sh' 2>/dev/null || true

echo "==== $(date -Is) waiting for GPU eval/train processes to drain ===="
while pgrep -f 'run_eval_qwen3vl_n781.sh|evaluate.py|local_deploy.py|torchrun.*trainer.py|fill_v27_n781_missing' >/dev/null 2>&1; do
  echo "still busy: $(pgrep -af 'run_eval_qwen3vl|evaluate.py|local_deploy|torchrun.*trainer|fill_v27' | head -3)"
  sleep 30
done

# Confirm P1_GO remains HOLD before prepare (prepare itself does not train).
GO="$ABL/freeze/P1_GO.json"
status=$("$BASE/miniconda3/envs/recredit/bin/python" -c "import json; print(json.load(open('$GO'))['status'])")
echo "P1_GO status=$status (expected HOLD; will not start run_p1_all.sh)"

echo "==== $(date -Is) starting prepare_p1_all.sh ===="
cd "$ABL"
bash "$ABL/code/ablations/prepare_p1_all.sh"
echo "==== $(date -Is) prepare_p1_all finished; review ref manifests then set P1_GO=GO manually ===="
