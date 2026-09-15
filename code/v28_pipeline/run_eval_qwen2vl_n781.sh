#!/usr/bin/env bash
# Qwen2-VL family n781 eval ONLY (n806 minus 25 long-range C).
# HARD RULE: never accept n806 / test_809 full as INPUT.
#
# Env: EVAL_MODEL, EVAL_NAME, RESULTS, BAN_TOKEN_220 (default 1)
# Optional: INPUT_PATH (must be the n781 pack), N_SHARDS
set -euo pipefail

BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ER_CODE="$BASE/code/Embodied-Omni/embodied_reasoner"
ER_DATA="${ER_ROOT:-${EMBODIED_REASONER_ROOT}}"
CONDA="$BASE/miniconda3"
RESULTS="${RESULTS:-$BASE/results/ablation_v28}"
LOGDIR="$RESULTS/logs"
N_SHARDS="${N_SHARDS:-8}"
EMBED_PORT="${EMBED_PORT:-20000}"
VLM_BASE_PORT="${VLM_BASE_PORT:-10001}"

NAME="${EVAL_NAME:?EVAL_NAME required}"
MODEL="${EVAL_MODEL:?EVAL_MODEL required}"
INPUT_PATH="${INPUT_PATH:-$BASE/data/eval/test_n781_SMT_C78.json}"
BAN_TOKEN_220="${BAN_TOKEN_220:-1}"

mkdir -p "$RESULTS" "$LOGDIR" "$ER_CODE/data" "$ER_CODE/data/result"
exec >>"$LOGDIR/eval_${NAME}.log" 2>&1
echo "==== $(date -Is) qwen2vl n781-ONLY name=$NAME model=$MODEL ===="

[[ -f "$MODEL/config.json" ]] || { echo "ERROR missing $MODEL"; exit 1; }
[[ -f "$INPUT_PATH" ]] || { echo "ERROR missing INPUT_PATH=$INPUT_PATH"; exit 1; }

# Refuse n806 / full 809 by path name and by count
case "$INPUT_PATH" in
  *n806*|*/test_809.json|*test_809_n806*)
    echo "FATAL: n806/test809 input forbidden in n781 eval script: $INPUT_PATH"
    exit 2
    ;;
esac

python3 - <<PY
import json
from pathlib import Path
p = Path("$INPUT_PATH")
x = json.loads(p.read_text())
tasks = x if isinstance(x, list) else (x.get("examples") or x.get("data") or [])
assert len(tasks) == 781, f"expected n781 got {len(tasks)} path={p}"
# long-range C identities dropped from n806→n781
dropped = {str(i) for i in range(785, 810)}
ids = {str(e.get("identity")) for e in tasks if e.get("identity") is not None}
leak = ids & dropped
assert not leak, f"long-range C still in pack: {sorted(leak)[:10]}"
# also must not be full n806 size
assert len(tasks) != 806, "refusing n806-sized input"
print(f"input OK n=781 path={p}")
PY

source "$CONDA/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONUNBUFFERED=1
export IMAGE_RESOLUTION=351232
export MIN_PIXELS=3136
export MAX_PIXELS=351232
export THOR_PLATFORM="${THOR_PLATFORM:-Linux64}"
export EVAL_C_FIRST="${EVAL_C_FIRST:-1}"
export MAX_MODEL_INFER_COUNT=3
export MAX_SAME_ACTION=3
export MAX_STEP_SCALE=1.0
export SELF_CONSISTENCY_K=1
export BAN_TOKEN_220
unset INFER_CONSTRAINTS INFER_RECOVERY MIN_INTERACT_BEFORE_END END_WARN_CAP \
      RECOVERY_AFTER_FAILS REPEAT_RECOVERY_AT SOFT_REQUERY_MAX || true

echo "BAN_TOKEN_220=$BAN_TOKEN_220 MAX_MODEL_INFER_COUNT=$MAX_MODEL_INFER_COUNT INPUT=$INPUT_PATH"

# Omni evaluate.py reads data/test_809.json — wire n781 pack there (name is historical).
ln -sfn "$INPUT_PATH" "$ER_CODE/data/test_809.json"
echo "wired data/test_809.json -> $INPUT_PATH (n781)"
ln -sfn "$ER_DATA/agent_positions.json" "$ER_CODE/data/agent_positions.json"

if ! pgrep -f "Xorg.*:0" >/dev/null; then
  ai2thor-xorg start 0 || true
  sleep 3
fi

kill_port() { fuser -k "${1}/tcp" 2>/dev/null || true; }

start_embedding() {
  kill_port "$EMBED_PORT"; sleep 1
  nohup python ./inference/local_deploy.py --embedding 1 --port "$EMBED_PORT" \
    >"$LOGDIR/embed_${NAME}.log" 2>&1 &
  echo $! >"$LOGDIR/embed_${NAME}.pid"
  for i in $(seq 1 120); do nc -z localhost "$EMBED_PORT" && return 0; sleep 2; done
  echo "ERROR embed failed"; exit 1
}

start_vlm_shard() {
  local gpu="$1" port="$2"
  kill_port "$port"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python ./inference/local_deploy.py \
    --frame hf --model_type qwen2_vl --model_name "$MODEL" --port "$port" \
    >"$LOGDIR/vlm_${NAME}_g${gpu}.log" 2>&1 &
  echo $! >"$LOGDIR/vlm_${NAME}_g${gpu}.pid"
}

wait_port() {
  local port="$1"
  for i in $(seq 1 240); do nc -z localhost "$port" && return 0; sleep 2; done
  return 1
}

stop_vlms() {
  for f in "$LOGDIR"/vlm_${NAME}_g*.pid; do
    [[ -f "$f" ]] && kill "$(cat "$f")" 2>/dev/null || true
  done
  for p in $(seq 0 $((N_SHARDS - 1))); do kill_port $((VLM_BASE_PORT + p)); done
}

cd "$ER_CODE"
rm -rf "$RESULTS/$NAME"
mkdir -p "$RESULTS/$NAME"
rm -rf "$ER_CODE/data/$NAME"
ln -sfn "$RESULTS/$NAME" "$ER_CODE/data/$NAME"

start_embedding
for shard in $(seq 0 $((N_SHARDS - 1))); do
  start_vlm_shard "$shard" $((VLM_BASE_PORT + shard))
done
for shard in $(seq 0 $((N_SHARDS - 1))); do
  wait_port $((VLM_BASE_PORT + shard)) || {
    echo "VLM shard $shard failed"
    tail -40 "$LOGDIR/vlm_${NAME}_g${shard}.log" || true
    exit 1
  }
done

PIDS=()
for shard in $(seq 0 $((N_SHARDS - 1))); do
  port=$((VLM_BASE_PORT + shard))
  cur=$((shard + 1))
  (
    export CUDA_VISIBLE_DEVICES="$shard"
    export THOR_X_DISPLAY="0.${shard}"
    export BAN_TOKEN_220
    export MAX_MODEL_INFER_COUNT=3
    export MAX_SAME_ACTION=3
    python ./evaluate/evaluate.py \
      --model_name "$NAME" \
      --input_path "data/test_809.json" \
      --batch_size 200 \
      --cur_count "$cur" \
      --total_count "$N_SHARDS" \
      --port "$port" \
      --no_dashboard
  ) >"$LOGDIR/${NAME}_shard${shard}.log" 2>&1 &
  PIDS+=("$!")
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
stop_vlms
kill "$(cat "$LOGDIR/embed_${NAME}.pid" 2>/dev/null)" 2>/dev/null || true
kill_port "$EMBED_PORT"
[[ "$FAIL" -eq 0 ]] || { echo "eval shards failed"; exit 1; }

python ./evaluate/show_result.py --model_name "$NAME" | tee "$RESULTS/$NAME/show_result.txt" || true
[[ -f data/result/result.csv ]] && cp -f data/result/result.csv "$RESULTS/$NAME/result.csv" || true
n=$(find "$RESULTS/$NAME" -name result.json 2>/dev/null | wc -l)
echo "==== $(date -Is) eval done $NAME n=$n/781 ===="

python3 - <<PY
import json
from pathlib import Path
root = Path("$RESULTS/$NAME")
ok2 = tot2 = 0
for p in root.rglob("result.json"):
    try:
        j = json.loads(p.read_text())
    except Exception:
        continue
    tot2 += 1
    s = j.get("success", j.get("task_success", j.get("is_success")))
    if s is True or s == 1 or s == "true":
        ok2 += 1
sr = (ok2 / tot2 * 100) if tot2 else 0.0
meta = {
    "name": "$NAME",
    "universe": "n781_SMT_C78",
    "n_completed": tot2,
    "n_success": ok2,
    "sr_pct": round(sr, 4),
    "policy": "completed_only_n781",
    "input": "$INPUT_PATH",
    "model": "$MODEL",
}
(root / "N781_SR.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
PY

ATTR="$BASE/code/recredit_r1/error_attribution.py"
[[ -f "$ATTR" ]] && "$CONDA/envs/recredit/bin/python" "$ATTR" "$RESULTS/$NAME" || echo "WARN attribution skipped"
printf '%s' "{\"phase\":\"eval_done\",\"name\":\"$NAME\",\"n\":$n,\"universe\":\"n781\"}" > "$RESULTS/${NAME}_STATUS.json"
echo "==== $(date -Is) n781 eval exit $NAME ===="
