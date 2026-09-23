#!/usr/bin/env bash
set -uo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
LOGP1="$BASE/logs/icra_p1"
PY="$BASE/miniconda3/envs/recredit/bin/python"
REPORT="$LOGP1/watchdog_last.json"
mkdir -p "$LOGP1" "$BASE/logs/icra_p0"

train_busy() { pgrep -f 'torchrun.*trainer.py' >/dev/null 2>&1; }

active_eval_name() {
  # first evaluate.py --model_name X
  pgrep -af 'evaluate.py --model_name' | grep -v pgrep | head -1 | sed -n 's/.*--model_name \([^ ]*\).*/\1/p'
}

have_eval_workers() {
  pgrep -f 'evaluate.py --model_name|local_deploy.py --frame' >/dev/null 2>&1
}

have_orchestrator() {
  pgrep -f "run_p1_resume_eval.sh|run_p0_resume_x3.sh|run_chain_p1_r3_then_p0.sh|run_p1_eval_n781.sh|run_wait_workers_then_complete.sh" >/dev/null 2>&1
}

"$PY" - <<'PY' > "$REPORT"
import json, time
from pathlib import Path
now=time.time()
out={"ts":time.strftime("%Y-%m-%dT%H:%M:%S%z"), "p1":[], "p0":[]}
def scan(root, prefix, arms, seeds, reps):
  rows=[]
  for rep in reps:
    for seed in seeds:
      for arm in arms:
        name=f"{prefix}_{arm}_s{seed}_r{rep}"
        d=root/name
        n=len(list(d.rglob("result.json"))) if d.exists() else 0
        last=None
        if n:
          last=max(p.stat().st_mtime for p in d.rglob("result.json"))
        st="done" if n>=781 else ("partial" if n>0 else "pending")
        idle=(now-last)/3600 if last else None
        rows.append({"name":name,"n":n,"status":st,"idle_h":None if idle is None else round(idle,2)})
  return rows
out["p1"]=scan(Path(__import__("os").environ["RECREDIT_ROOT"])/"results/icra_p1","p1",["attr","type_agnostic"],[1,17,31],[1,2,3])
out["p0"]=scan(Path(__import__("os").environ["RECREDIT_ROOT"])/"results/icra_p0","p0_full",["geo","u2"],[1,17,31],[1,2,3])
print(json.dumps(out))
PY

echo "==== $(date -Is) watchdog ===="
"$PY" - <<'PY'
import json
from pathlib import Path
j=json.loads(Path(__import__("os").environ["RECREDIT_ROOT"])/"logs/icra_p1/watchdog_last.json".read_text())
for label in ("p1","p0"):
  rows=j[label]
  print(label,"done",sum(1 for x in rows if x["status"]=="done"),
        "partial",sum(1 for x in rows if x["status"]=="partial"),
        "pending",sum(1 for x in rows if x["status"]=="pending"))
  for x in rows:
    if x["status"]=="partial":
      print(" partial",x)
PY

if train_busy; then
  echo "train busy; skip"
  exit 0
fi

ACTIVE=$(active_eval_name || true)
echo "active_eval=$ACTIVE"

# If workers alive: only kill if the ACTIVE name is stale (>=1.5h no new results)
if have_eval_workers; then
  if [[ -n "$ACTIVE" ]]; then
    idle=$("$PY" - <<PY
import json
from pathlib import Path
j=json.loads(Path(__import__("os").environ["RECREDIT_ROOT"])/"logs/icra_p1/watchdog_last.json".read_text())
for x in j["p1"]+j["p0"]:
  if x["name"]=="$ACTIVE":
    print(x.get("idle_h") if x.get("idle_h") is not None else 0)
    break
else:
  print(0)
PY
)
    echo "active_idle_h=$idle"
    # bc compare
    stale_active=$("$PY" -c "print(1 if float('$idle' or 0)>=1.5 else 0)")
    if [[ "$stale_active" == "1" ]]; then
      echo "ACTIVE STALE $ACTIVE idle=${idle}h -> kill workers and relaunch chain"
      pkill -f 'evaluate.py --model_name' 2>/dev/null || true
      pkill -f 'local_deploy.py' 2>/dev/null || true
      pkill -f 'run_eval_qwen3vl_n781.sh' 2>/dev/null || true
      pkill -f 'run_p1_resume_eval.sh' 2>/dev/null || true
      pkill -f 'run_p0_resume_x3.sh' 2>/dev/null || true
      pkill -f 'run_chain_p1_r3_then_p0.sh' 2>/dev/null || true
      sleep 5
    else
      # healthy workers; ensure orchestrator exists (chain waiting is ok)
      if have_orchestrator; then
        echo "workers+orchestrator healthy"
        exit 0
      fi
      echo "workers alive but no orchestrator; attach wait-then-complete"
      if ! pgrep -f "/run_wait_workers_then_complete\.sh" >/dev/null; then
        LOG="$LOGP1/wait_workers_then_complete.nohup.log"
        nohup bash "$ABL/run_wait_workers_then_complete.sh" >> "$LOG" 2>&1 &
        echo $! > "$LOGP1/wait_workers_then_complete.pid"
        echo "attached orch pid=$(cat $LOGP1/wait_workers_then_complete.pid)"
      fi
      exit 0
    fi
  else
    echo "workers without parseable model_name; leave alone"
    exit 0
  fi
fi

# No eval workers: if unfinished, start chain immediately
need=$("$PY" - <<'PY'
import json
from pathlib import Path
j=json.loads(Path(__import__("os").environ["RECREDIT_ROOT"])/"logs/icra_p1/watchdog_last.json".read_text())
print(1 if any(x["status"]!="done" for x in j["p1"]+j["p0"]) else 0)
PY
)
if [[ "$need" != "1" ]]; then
  echo "all done"
  exit 0
fi

if pgrep -f '/run_chain_p1_r3_then_p0\.sh|/run_p1_resume_eval\.sh|/run_p0_resume_x3\.sh' >/dev/null; then
  echo "orchestrator already running"
  exit 0
fi

echo "IDLE + unfinished -> launch chain"
LOG=$LOGP1/chain_p1_r3_then_p0.nohup.log
nohup env WAIT_PID= MAX_SHARDS=8 bash "$ABL/run_chain_p1_r3_then_p0.sh" >> "$LOG" 2>&1 &
echo $! > $LOGP1/chain_p1_r3_then_p0.pid
echo "launched $(cat $LOGP1/chain_p1_r3_then_p0.pid)"
