#!/usr/bin/env bash
# After p1_type_agnostic_s31_r2 reaches strict n=781:
#   1) kill other hung schedulers (resume / wait_workers / r3 chains / p0 resume)
#   2) drain GPU workers
#   3) immediately fill+strict all incomplete P0/P1 eval dirs
set -uo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
TARGET="$BASE/results/icra_p1/p1_type_agnostic_s31_r2/N781_SR.json"
LOGDIR="$BASE/logs/icra_p1"
mkdir -p "$LOGDIR"

target_done() {
  [[ -f "$TARGET" ]] || return 1
  "$PY" - <<PY
import json, sys
m=json.load(open("$TARGET"))
sys.exit(0 if m.get("policy")=="strict_manifest_metrics_success" and m.get("n_completed")==781 else 1)
PY
}

echo "==== $(date -Is) waiting for TA_s31_r2 strict done ($TARGET) ===="
while ! target_done; do
  # If r3 already started, stop immediately (REPS=1 2 3 race).
  if pgrep -f 'evaluate.py --model_name p1_.*_r3' >/dev/null 2>&1 \
     || [[ -d "$BASE/results/icra_p1/p1_attr_s1_r3" ]]; then
    echo "==== $(date -Is) DETECTED r3 start; aborting schedulers now ===="
    break
  fi
  cur=$(ps -eo cmd | awk '/evaluate.py --model_name /{print; exit}' || true)
  echo "==== $(date -Is) not yet; current=$cur ===="
  sleep 5
done
echo "==== $(date -Is) TA_s31_r2 DONE; stopping other hung schedulers ===="

# Kill schedulers that would continue to r3 / p0 x3 / chains.
# Do NOT match this script itself.
for pat in \
  '/run_p1_resume_eval.sh' \
  '/run_wait_workers_then_complete.sh' \
  '/run_p0_resume_x3.sh' \
  '/run_p0_wait_chain_then_x3.sh' \
  '/run_p0_stop_r3_then_prepare_p1.sh' \
  '/run_chain_p1_r3_then_p0' \
  '/run_p1_wait_train_then_eval.sh' \
  '/run_p1_eval_all.sh'
do
  pids=$(ps -eo pid,cmd | awk -v p="$pat" 'index($0,p) && !/awk/ && !/run_after_ta_s31_r2/ {print $1}')
  for pid in $pids; do
    echo "kill scheduler pid=$pid ($pat)"
    kill "$pid" 2>/dev/null || true
  done
done
sleep 2
# force leftovers
pkill -f '/run_p1_resume_eval.sh' 2>/dev/null || true
pkill -f '/run_wait_workers_then_complete.sh' 2>/dev/null || true
pkill -f '/run_p0_resume_x3.sh' 2>/dev/null || true

echo "==== $(date -Is) drain GPU eval/fill workers ===="
# stop any in-flight full evals that a killed parent left behind
pkill -f '/run_eval_qwen3vl_n781.sh' 2>/dev/null || true
pkill -f 'evaluate.py --model_name' 2>/dev/null || true
pkill -f 'local_deploy.py' 2>/dev/null || true
sleep 5
for i in $(seq 1 60); do
  if ! pgrep -f 'evaluate.py --model_name|local_deploy.py|/run_eval_qwen3vl_n781.sh|fill_v27_n781_missing' >/dev/null 2>&1; then
    break
  fi
  echo "still draining ($i)"
  sleep 10
done

echo "==== $(date -Is) immediate fill-all-incomplete ===="
bash "$ABL/code/ablations/run_fill_all_incomplete_evals.sh"
echo "==== $(date -Is) after-TA_s31_r2 stop+fill done ===="
