#!/usr/bin/env bash
# Train one failctrl_v3 GR arm from Stage-I. Does not use the live CE tree.
set -euo pipefail
ARM="${ARM:?neg_uniform|type_neg|type_neg_invert}"
SEED="${SEED:?}"
ETA="${ETA:?}"
PHASE="${PHASE:?pilot|formal}"
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
V3="$BASE/code/failctrl_v3_gr_mass_matched"
LIVE_SRC="$V3/live_src"
S1="${S1_BEST:-$BASE/ckpts/wt_ablation_v5/stage1_p25/best}"
ETA_TAG=$(python3 - <<PY
eta=float("$ETA")
print(f"eta{int(round(eta*100)):03d}")
PY
)
DATA="$BASE/data/parquet_p123_shared/P3_recredit_failctrl_v3/${ETA_TAG}/${ARM}"
CKPT="$BASE/ckpts/failctrl_v3_gr/${ETA_TAG}/${ARM}_s${SEED}_${PHASE}"
LOGDIR="$BASE/logs/failctrl_v3_gr/${ETA_TAG}/${ARM}_s${SEED}_${PHASE}"
mkdir -p "$CKPT" "$LOGDIR"
export PYTHONPATH="$LIVE_SRC:${PYTHONPATH:-}"
export FAILCTRL_V3=1
export FAIL_SPAN_ROUTE=grounding_think_action
export NEG_CREDIT_MODE=unlikelihood
export FAILCTRL_FAIL_FILES="$DATA/failure_train.parquet"
export BAN_TOKEN_220=1
export KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
if [[ "$PHASE" == "pilot" ]]; then
  # U from Full success stream: 1794 steps/epoch * 2 = 3588; U_p = ceil(0.25U)=897
  export FAILCTRL_MAX_STEPS="${FAILCTRL_MAX_STEPS:-897}"
else
  unset FAILCTRL_MAX_STEPS || true
fi
NPROC="${NPROC:-8}"
export MASTER_PORT="${MASTER_PORT:-29571}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
cd "$LIVE_SRC"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" trainer.py \
  --config-path "$LIVE_SRC" --config-name config \
  trainer.seed="$SEED" \
  trainer.total_epochs=2 \
  trainer.save_dir="$CKPT" \
  trainer.stage=stage2 \
  model.path="$S1" \
  model.precision=bf16 \
  model.attn_implementation=sdpa \
  model.gradient_checkpointing=true \
  data.train_files="$DATA/success_train.parquet" \
  data.val_files="$DATA/success_val.parquet" \
  data.train_batch_size=2 \
  data.micro_batch_size_per_gpu=2 \
  data.max_length=8192 \
  data.max_image_side=448 \
  data.max_history_images=3 \
  data.truncation=left \
  optim.lr=5.0e-6 \
  optim.betas=[0.9,0.95] \
  optim.weight_decay=0.01 \
  optim.warmup_steps_ratio=0.03 \
  optim.clip_grad=1.0 \
  dpo.enabled=false \
  2>&1 | tee -a "$LOGDIR/train.log"
echo "DONE $ARM seed=$SEED eta=$ETA phase=$PHASE" | tee -a "$LOGDIR/done.txt"
