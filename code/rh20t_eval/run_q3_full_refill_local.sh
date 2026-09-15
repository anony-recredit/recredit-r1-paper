#!/usr/bin/env bash
# Local GPU refill of Q3 Full RH20T request failures. Same offline eval protocol.
set -euo pipefail
ROOT="${RECREDIT_ROOT}"
CODE="$ROOT/code/rh20t_eval"
EVAL="$ROOT/evaluation/rh20t"
PACK="$ROOT/data/rh20t_recredit_cfg3_pc_n100"
OUT="$ROOT/results/rh20t_cfg3_pc_n100/q3_full_ol"
FAIL="$ROOT/results/rh20t_metrics/q3_full_ol_FAILED_STEPS.json"
CKPT="${RECREDIT_ROOT}/ckpts/wt_ablation_v5_qwen3/full/best"
ER="${RECREDIT_ROOT}/code/Embodied-Omni/embodied_reasoner"
PY="${RECREDIT_PYTHON:-python}"
PORT="${PORT:-18001}"
LOGDIR="$ROOT/results/rh20t_metrics/logs_local_refill"
mkdir -p "$LOGDIR"

export PYTHONUNBUFFERED=1
export IMAGE_RESOLUTION=351232 MIN_PIXELS=3136 MAX_PIXELS=351232
export BAN_TOKEN_220=1 SELF_CONSISTENCY_K=1
export THINKING_MAX_NEW_TOKENS="${THINKING_MAX_NEW_TOKENS:-2048}"

[[ -f "$CKPT/model.safetensors" || -f "$CKPT/config.json" ]] || { echo "missing ckpt $CKPT"; exit 1; }
[[ -f "$FAIL" ]] || { echo "missing fail list"; exit 1; }
[[ -f "$PACK/all.json" ]] || { echo "missing pack"; exit 1; }

if ! nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
  echo "==== $(date -Is) start qwen3_vl on :$PORT ===="
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  cd "$ER"
  nohup "$PY" ./inference/local_deploy.py \
    --frame hf --model_type qwen3_vl --model_name "$CKPT" --port "$PORT" \
    >"$LOGDIR/vlm_q3_full_ol_refill_p${PORT}.log" 2>&1 &
  echo $! >"$LOGDIR/vlm.pid"
  ok=0
  for t in $(seq 1 180); do
    if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then ok=1; break; fi
    sleep 2
  done
  [[ "$ok" == 1 ]] || { echo "server failed"; tail -50 "$LOGDIR/vlm_q3_full_ol_refill_p${PORT}.log"; exit 1; }
  echo "server ready pid=$(cat "$LOGDIR/vlm.pid")"
else
  echo "reuse existing server :$PORT"
fi

echo "==== $(date -Is) refill failed steps ===="
"$PY" "$CODE/refill_failed_steps.py" \
  --pack "$PACK" --out "$OUT" --fail-list "$FAIL" --port "$PORT" \
  | tee "$LOGDIR/refill.log"

echo "==== $(date -Is) rescore six variants ===="
cd "$EVAL"
"$PY" run_six_variants.py | tee "$LOGDIR/rescore.log"

echo "==== $(date -Is) rebuild fail list / gate-D sample ===="
"$PY" "$EVAL/export_q3_full_failed_steps.py" || true
"$PY" "$CODE/build_gate_d_sample.py" | tee "$LOGDIR/gate_d.log"
echo "==== $(date -Is) local refill pipeline done ===="
