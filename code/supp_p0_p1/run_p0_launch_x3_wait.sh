#!/usr/bin/env bash
# Replace/launch waiter: after in-flight eval-all → fill missing → x3 → fill again.
set -euo pipefail
BASE=${RECREDIT_ROOT}
ABL=${RECREDIT_ABLATIONS}
LOG=$BASE/logs/icra_p0/wait_chain_then_x3.nohup.log
LOCK=$BASE/logs/icra_p0/wait_chain_then_x3.lock
mkdir -p $BASE/logs/icra_p0

# stop previous waiter only (never the eval chain)
if [[ -f "$LOCK" ]]; then
  old=$(cat "$LOCK" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    cmd=$(ps -p "$old" -o cmd= || true)
    case "$cmd" in
      *run_p0_wait_chain_then_x3.sh*)
        echo "killing old waiter pid=$old"
        kill "$old" 2>/dev/null || true
        sleep 1
        ;;
      *)
        echo "lock pid=$old is not waiter ($cmd); abort"
        exit 1
        ;;
    esac
  fi
fi

CHAIN_PID=$(ps -eo pid,cmd | awk '/\/run_p0_eval_all\.sh/ && !/awk/ {print $1; exit}')
if [[ -z "${CHAIN_PID:-}" ]]; then
  echo "no in-flight eval-all; fill then x3 now"
  nohup bash -lc "
    set -euo pipefail
    NAMES='p0_full_geo_s1 p0_full_geo_s17 p0_full_geo_s31 p0_full_u2_s1 p0_full_u2_s17 p0_full_u2_s31' \
      bash $ABL/code/ablations/run_p0_fill_missing.sh
    bash $ABL/code/ablations/run_p0_eval_x3.sh
    NAMES=''
    for arm in geo u2; do for seed in 1 17 31; do for rep in 1 2 3; do
      NAMES=\"\$NAMES p0_full_\${arm}_s\${seed}_r\${rep}\"
    done; done; done
    NAMES=\$NAMES bash $ABL/code/ablations/run_p0_fill_missing.sh
  " >>"$LOG" 2>&1 &
  echo $! > "$LOCK"
  echo "PIPELINE_PID=$! LOG=$LOG"
  exit 0
fi

echo "CHAIN_PID=$CHAIN_PID"
nohup env CHAIN_PID="$CHAIN_PID" bash "$ABL/code/ablations/run_p0_wait_chain_then_x3.sh" >>"$LOG" 2>&1 &
echo $! > "$LOCK"
echo "WAIT_PID=$! CHAIN_PID=$CHAIN_PID LOG=$LOG"
sleep 1
tail -5 "$LOG" || true
ps -p "$CHAIN_PID" -o pid,etime,cmd
ps -p "$(cat "$LOCK")" -o pid,etime,cmd
