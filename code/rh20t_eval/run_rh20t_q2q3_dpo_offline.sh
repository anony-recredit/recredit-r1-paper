#!/usr/bin/env bash
# Offline RH20T cfg3_pc_n100 next-action eval for Q2/Q3 Attr-DPO checkpoints.
# Not closed-loop SR — teacher-forced history + current RGB → verb match.
#
# Env:
#   ARM=q2_dpo|q3_dpo|q2_full|q3_full|dpo|full|both
#   WAIT_FILL=0/1
#   N_SHARDS=8  OUT_SUFFIX=_ol  OVERWRITE=1
set -euo pipefail

BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
CONDA="$BASE/miniconda3"
ER_CODE="$BASE/code/Embodied-Omni/embodied_reasoner"
PACK="$BASE/data/rh20t_recredit_cfg3_pc_n100"
RESULTS="$BASE/results/rh20t_cfg3_pc_n100"
CODE_DIR="$BASE/code/recredit_r1"
PY_EVAL="$CODE_DIR/eval_rh20t_offline_action.py"
LOGDIR="$RESULTS/logs"
ARM="${ARM:-both}"
WAIT_FILL="${WAIT_FILL:-0}"
N_SHARDS="${N_SHARDS:-8}"
VLM_BASE_PORT="${VLM_BASE_PORT:-10001}"
MODEL_TYPE_Q2="${MODEL_TYPE_Q2:-qwen2_vl}"
MODEL_TYPE_Q3="${MODEL_TYPE_Q3:-qwen3_vl}"
OVERWRITE="${OVERWRITE:-1}"
# v2 = open-loop multiturn, n781-aligned sr_pct
OUT_SUFFIX="${OUT_SUFFIX:-_ol}"

Q2_DPO="$BASE/ckpts/wt_ablation_v5/full_dpo/best"
Q3_DPO="$BASE/ckpts/wt_ablation_v5_qwen3/full_dpo/best"
Q2_FULL="$BASE/ckpts/wt_ablation_v5/full/best"
Q3_FULL="$BASE/ckpts/wt_ablation_v5_qwen3/full/best"
Q2_S1="$BASE/ckpts/wt_ablation_v5/stage1_p25/best"
Q3_S1="$BASE/ckpts/wt_ablation_v5_qwen3/stage1_p25/best"
# Match n781: Uniform uses the worse epoch_0 ckpt
Q2_UNI_E0="$BASE/ckpts/wt_ablation_v5/uniform/epoch_0"
Q3_UNI_E0="${Q3_UNI_E0:-$BASE/ckpts/wt_ablation_v5_qwen3/uniform/epoch_0}"
Q2_INV="$BASE/ckpts/wt_ablation_v5/invert/best"
# backward-compat aliases
Q2_MODEL="${Q2_MODEL:-$Q2_DPO}"
Q3_MODEL="${Q3_MODEL:-$Q3_DPO}"

mkdir -p "$RESULTS" "$LOGDIR" "$CODE_DIR"
exec > >(tee -a "$LOGDIR/run_rh20t_offline.log") 2>&1
echo "==== $(date -Is) RH20T offline ARM=$ARM ===="

[[ -f "$PY_EVAL" ]] || { echo "ERROR missing $PY_EVAL"; exit 1; }
[[ -f "$PACK/all.json" ]] || { echo "ERROR missing pack $PACK"; exit 1; }

source "$CONDA/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONUNBUFFERED=1
export IMAGE_RESOLUTION=351232 MIN_PIXELS=3136 MAX_PIXELS=351232
export BAN_TOKEN_220="${BAN_TOKEN_220:-1}"
export SELF_CONSISTENCY_K=1

kill_port(){ fuser -k "${1}/tcp" 2>/dev/null || true; }

wait_ports_free() {
  local base="$1" n="$2"
  for i in $(seq 0 $((n-1))); do
    local p=$((base+i))
    if ss -lntp 2>/dev/null | grep -q ":$p "; then
      return 1
    fi
  done
  return 0
}

wait_fill_done() {
  [[ "$WAIT_FILL" == "1" ]] || return 0
  echo "[wait] Q3 dpo_r1 n781 fill / free VLM ports..."
  local uni="$BASE/data/eval/test_n781_SMT_C78.json"
  local root="$BASE/results/wt_ablation_v5_qwen3/dpo_r1"
  for t in $(seq 1 720); do
    local miss fill_alive
    miss=$(python3 - <<PY
import json
from pathlib import Path
uni=json.load(open("$uni"))
tasks=uni if isinstance(uni,list) else (uni.get("examples") or [])
by={str(e.get("identity")) for e in tasks}
have=set()
root=Path("$root")
if root.exists():
  for p in root.rglob("result.json"):
    try: have.add(str(json.loads(p.read_text()).get("identity")))
    except Exception: pass
print(len(by-have))
PY
)
    fill_alive=0
    pgrep -f "fill_qwen3_n781_missing|fill_v27_n781_missing" >/dev/null && fill_alive=1
    pgrep -f "evaluate.py --model_name dpo_r1" >/dev/null && fill_alive=1

    if [[ "$miss" == "0" && "$fill_alive" == "0" ]]; then
      if wait_ports_free "$VLM_BASE_PORT" 8; then
        echo "[wait] fill complete and ports free (t=${t})"
        return 0
      fi
      echo "[wait] fill complete; stopping leftover VLM servers on ${VLM_BASE_PORT}.."
      for i in $(seq 0 7); do kill_port $((VLM_BASE_PORT+i)); done
      # also kill orphan local_deploy tied to those ports
      pkill -f "inference/local_deploy.py" 2>/dev/null || true
      sleep 8
      if wait_ports_free "$VLM_BASE_PORT" 8; then
        return 0
      fi
    fi
    if (( t % 12 == 0 )); then
      echo "[wait] still ... miss=$miss fill_alive=$fill_alive t=${t} $(date -Is)"
    fi
    sleep 10
  done
  echo "ERROR timeout waiting for fill/ports"
  exit 1
}

start_vlm_fleet() {
  local model="$1" mtype="$2" n="$3" tag="$4"
  cd "$ER_CODE"
  local pids=()
  for i in $(seq 0 $((n-1))); do
    local port=$((VLM_BASE_PORT+i))
    local gpu=$i
    kill_port "$port"; sleep 1
    CUDA_VISIBLE_DEVICES=$gpu nohup python ./inference/local_deploy.py \
      --frame hf --model_type "$mtype" --model_name "$model" --port "$port" \
      >"$LOGDIR/vlm_${tag}_g${gpu}_p${port}.log" 2>&1 &
    pids+=($!)
    echo $! >"$LOGDIR/vlm_${tag}_g${gpu}.pid"
  done
  echo "[serve] started ${#pids[@]} servers model=$model type=$mtype"
  for i in $(seq 0 $((n-1))); do
    local port=$((VLM_BASE_PORT+i))
    local ok=0
    for t in $(seq 1 180); do
      if nc -z localhost "$port" 2>/dev/null; then ok=1; break; fi
      sleep 2
    done
    [[ "$ok" == "1" ]] || { echo "ERROR server port $port failed"; exit 1; }
    echo "[serve] port $port ready"
  done
}

stop_vlm_fleet() {
  local n="$1"
  for i in $(seq 0 $((n-1))); do
    kill_port $((VLM_BASE_PORT+i))
  done
  sleep 3
}

ports_csv() {
  local n="$1"
  local out=""
  for i in $(seq 0 $((n-1))); do
    local p=$((VLM_BASE_PORT+i))
    if [[ -z "$out" ]]; then out="$p"; else out="$out,$p"; fi
  done
  echo "$out"
}

run_arm() {
  local name="$1" model="$2" mtype="$3"
  local out_name="${name}${OUT_SUFFIX}"
  local out="$RESULTS/$out_name"
  mkdir -p "$out"
  echo "==== $(date -Is) eval arm=$out_name model=$model overwrite=$OVERWRITE ===="
  [[ -f "$model/config.json" ]] || { echo "ERROR missing $model"; exit 1; }

  stop_vlm_fleet "$N_SHARDS"
  start_vlm_fleet "$model" "$mtype" "$N_SHARDS" "$out_name"
  local ports
  ports=$(ports_csv "$N_SHARDS")

  local ow_flag=()
  [[ "$OVERWRITE" == "1" ]] && ow_flag=(--overwrite)

  local pids=()
  for s in $(seq 0 $((N_SHARDS-1))); do
    python "$PY_EVAL" \
      --pack "$PACK" --out "$out" --split all \
      --ports $((VLM_BASE_PORT+s)) \
      --shard "$s" --n-shards "$N_SHARDS" \
      "${ow_flag[@]}" \
      >"$LOGDIR/eval_${out_name}_shard${s}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done

  # fill missing / incomplete (no v2 protocol yet)
  python3 - <<PY
import json, subprocess
from pathlib import Path
pack=Path("$PACK")
out=Path("$out")
exs=json.loads((pack/"all.json").read_text())["examples"]
miss=[]
for e in exs:
    rj=out/str(e["id"])/"result.json"
    ok=False
    if rj.exists():
        try:
            d=json.loads(rj.read_text())
            ok=("metrics" in d and "success" in d["metrics"]
                and str(d.get("protocol","")).startswith("openloop_multiturn"))
        except Exception:
            ok=False
    if not ok:
        miss.append(e)
print("fill_missing", len(miss))
Path("$RESULTS/${out_name}_missing.json").write_text(json.dumps([e["id"] for e in miss], indent=2))
if not miss:
    raise SystemExit(0)
ids=",".join(e["id"] for e in miss)
cmd=["python","$PY_EVAL","--pack","$PACK","--out",str(out),"--split","all",
     "--ports","$ports","--ids",ids,"--n-shards","1","--shard","0","--overwrite"]
print(" ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd))
PY
  fill_rc=$?
  [[ "$fill_rc" == "0" ]] || rc=1

  python3 - <<PY
import json
from pathlib import Path
import importlib.util
from datetime import datetime
spec=importlib.util.spec_from_file_location("ev","$PY_EVAL")
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
pack=Path("$PACK"); out=Path("$out")
score=ev.summarize(out, ev.load_examples(pack,"all"), arm="$out_name", model="$model")
train_ids={e["id"] for e in ev.load_examples(pack,"train")}
test_ids={e["id"] for e in ev.load_examples(pack,"test")}
rows=[]
for p in out.iterdir():
  rj=p/"result.json"
  if rj.exists():
    try: rows.append(json.loads(rj.read_text()))
    except Exception: pass

def agg(ids):
  sub=[r for r in rows if r["id"] in ids]
  if not sub: return {"n":0,"succ":0,"sr_pct":None}
  succ=sum(1 for r in sub if int((r.get("metrics") or {}).get("success",0)))
  return {"n":len(sub),"succ":succ,"sr_pct":round(100.0*succ/len(sub),2)}

score["train"]=agg(train_ids)
score["test"]=agg(test_ids)
score["ended"]=datetime.now().isoformat(timespec="seconds")
Path("$RESULTS/score_${out_name}.json").write_text(json.dumps(score, indent=2))
# also symlink-style primary name without forcing
print(json.dumps(score, indent=2))
PY

  echo "==== $(date -Is) arm=$out_name done rc=$rc ===="
  return $rc
}

wait_fill_done

rc=0
case "$ARM" in
  q3_dpo)
    run_arm q3_dpo "$Q3_DPO" "$MODEL_TYPE_Q3" || rc=1
    ;;
  q2_dpo)
    run_arm q2_dpo "$Q2_DPO" "$MODEL_TYPE_Q2" || rc=1
    ;;
  q3_full)
    run_arm q3_full "$Q3_FULL" "$MODEL_TYPE_Q3" || rc=1
    ;;
  q2_full)
    run_arm q2_full "$Q2_FULL" "$MODEL_TYPE_Q2" || rc=1
    ;;
  q3_stage1|q3_s1)
    run_arm q3_stage1 "$Q3_S1" "$MODEL_TYPE_Q3" || rc=1
    ;;
  q2_stage1|q2_s1)
    run_arm q2_stage1 "$Q2_S1" "$MODEL_TYPE_Q2" || rc=1
    ;;
  q3_uniform_e0|q3_uni)
    run_arm q3_uniform_e0 "$Q3_UNI_E0" "$MODEL_TYPE_Q3" || rc=1
    ;;
  q2_uniform_e0|q2_uni)
    run_arm q2_uniform_e0 "$Q2_UNI_E0" "$MODEL_TYPE_Q2" || rc=1
    ;;
  q2_invert)
    run_arm q2_invert "$Q2_INV" "$MODEL_TYPE_Q2" || rc=1
    ;;
  dpo|both)
    run_arm q3_dpo "$Q3_DPO" "$MODEL_TYPE_Q3" || rc=1
    run_arm q2_dpo "$Q2_DPO" "$MODEL_TYPE_Q2" || rc=1
    ;;
  full)
    run_arm q3_full "$Q3_FULL" "$MODEL_TYPE_Q3" || rc=1
    run_arm q2_full "$Q2_FULL" "$MODEL_TYPE_Q2" || rc=1
    ;;
  # Remaining options: stage1 only (NO Q2 uniform_e0 / invert)
  options|opts)
    run_arm q3_stage1 "$Q3_S1" "$MODEL_TYPE_Q3" || rc=1
    run_arm q2_stage1 "$Q2_S1" "$MODEL_TYPE_Q2" || rc=1
    ;;
  all)
    run_arm q3_full "$Q3_FULL" "$MODEL_TYPE_Q3" || rc=1
    run_arm q2_full "$Q2_FULL" "$MODEL_TYPE_Q2" || rc=1
    run_arm q3_dpo "$Q3_DPO" "$MODEL_TYPE_Q3" || rc=1
    run_arm q2_dpo "$Q2_DPO" "$MODEL_TYPE_Q2" || rc=1
    ;;
  *)
    echo "ERROR unknown ARM=$ARM"; exit 1
    ;;
esac

stop_vlm_fleet "$N_SHARDS"
echo "==== $(date -Is) ALL DONE rc=$rc ===="
exit $rc
