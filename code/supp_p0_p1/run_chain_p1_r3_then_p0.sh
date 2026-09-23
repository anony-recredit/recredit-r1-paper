#!/usr/bin/env bash
# After current P1 resume finishes:
#   1) P1 resume REPS=1 2 3 (skip done; adds r3 for all 6 ckpts; fill partials)
#   2) P0 resume x3 (fill u2_s31_r2 + remaining r3)
# Does not kill in-flight evals.
set -uo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
LOGDIR_P1="${LOGDIR_P1:-$BASE/logs/icra_p1}"
LOGDIR_P0="${LOGDIR_P0:-$BASE/logs/icra_p0}"
mkdir -p "$LOGDIR_P1" "$LOGDIR_P0"

wait_gpu_idle() {
  local tag="$1"
  while pgrep -f 'torchrun.*trainer.py|/run_eval_qwen3vl_n781.sh|evaluate.py --model_name|local_deploy.py --frame' >/dev/null 2>&1; do
    echo "==== $(date -Is) $tag waiting GPU/eval workers ===="
    sleep 60
  done
}

wait_pid_or_gone() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "==== $(date -Is) pid $pid already gone ===="
    return 0
  fi
  echo "==== $(date -Is) waiting for pid=$pid ===="
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  echo "==== $(date -Is) pid $pid exited ===="
}

echo "==== $(date -Is) chain start ===="

# Prefer explicit WAIT_PID; else current resume_eval.lock; else any run_p1_resume_eval
WAIT_PID="${WAIT_PID:-}"
if [[ -z "$WAIT_PID" && -f "$LOGDIR_P1/resume_eval.lock" ]]; then
  WAIT_PID=$(cat "$LOGDIR_P1/resume_eval.lock" || true)
fi
if [[ -z "$WAIT_PID" ]]; then
  WAIT_PID=$(pgrep -f '/run_p1_resume_eval\.sh' | head -1 || true)
fi
wait_pid_or_gone "$WAIT_PID"
wait_gpu_idle "post-p1-resume"

# --- Phase A: P1 all ckpts through r3 ---
echo "==== $(date -Is) PHASE A: P1 resume REPS=1 2 3 ===="
export MAX_SHARDS="${MAX_SHARDS:-8}"
export REPS="1 2 3"
set +e
REPS="1 2 3" MAX_SHARDS=8 bash "$ABL/code/ablations/run_p1_resume_eval.sh"
rc_p1=$?
set -e
echo "==== $(date -Is) PHASE A done rc=$rc_p1 ===="
wait_gpu_idle "post-p1-r3"

# --- Phase B: P0 complete unfinished ---
echo "==== $(date -Is) PHASE B: P0 resume x3 ===="
set +e
MAX_SHARDS=8 bash "$ABL/code/ablations/run_p0_resume_x3.sh"
rc_p0=$?
set -e
echo "==== $(date -Is) PHASE B done rc=$rc_p0 ===="

echo "==== $(date -Is) chain finished p1_rc=$rc_p1 p0_rc=$rc_p0 ===="
exit 0
