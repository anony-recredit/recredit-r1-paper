#!/usr/bin/env bash
# Launch waiter: after in-flight run_p1_all.sh → 2-rep n781 eval + aggregate.
set -euo pipefail
BASE=${RECREDIT_ROOT}
ABL=${RECREDIT_ABLATIONS}
LOG=$BASE/logs/icra_p1/wait_train_then_eval.nohup.log
LOCK=$BASE/logs/icra_p1/wait_train_then_eval.lock
mkdir -p $BASE/logs/icra_p1

if [[ -f "$LOCK" ]]; then
  old=$(cat "$LOCK" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    cmd=$(ps -p "$old" -o cmd= || true)
    if [[ "$cmd" == *run_p1_wait_train_then_eval.sh* ]]; then
      echo "already waiting pid=$old"; exit 0
    fi
  fi
fi

TRAIN_PID=$(ps -eo pid,cmd | awk '/\/run_p1_all\.sh/ && !/awk/ {print $1; exit}')
if [[ -z "${TRAIN_PID:-}" ]]; then
  echo "no run_p1_all; start eval×2 now"
  nohup env REPS="1 2" bash "$ABL/code/ablations/run_p1_eval_all.sh" >>"$LOG" 2>&1 &
  echo $! >"$LOCK"
  echo "EVAL_PID=$! LOG=$LOG"
  # still aggregate after — better run full wait script with TRAIN_PID=0
  exit 0
fi

echo "TRAIN_PID=$TRAIN_PID"
nohup env TRAIN_PID="$TRAIN_PID" REPS="1 2" \
  bash "$ABL/code/ablations/run_p1_wait_train_then_eval.sh" >>"$LOG" 2>&1 &
echo $! >"$LOCK"
echo "WAIT_PID=$! TRAIN_PID=$TRAIN_PID REPS=1 2 LOG=$LOG"
sleep 2
tail -10 "$LOG" || true
ps -p "$TRAIN_PID" "$(cat $LOCK)" -o pid,etime,cmd
