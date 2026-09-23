#!/usr/bin/env bash
# P0 Stage-I: Geo or U2 × one seed. Explicit trainer.seed required.
set -euo pipefail
ARM="${ARM:?set ARM=geo|u2}"
SEED="${SEED:?set SEED=1|17|31}"
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
LIVE="$BASE/code/recredit_r1_qwen3"
BASE_MODEL="${BASE_MODEL:-$BASE/models/Qwen3-VL-8B-Instruct}"
NPROC="${NPROC:-8}"
S1_EPOCHS="${S1_EPOCHS:-1}"
LR="${LR:-1.0e-5}"

case "$ARM" in
  geo) S1_PQ="${S1_PQ:-$BASE/data/parquet_p123_shared/P3_recredit_p25}" ;;
  u2)  S1_PQ="${S1_PQ:-$BASE/data/parquet_p123_shared/P3_recredit_p25_u2_mass}" ;;
  *) echo "ARM must be geo|u2"; exit 2 ;;
esac

OUT="$BASE/ckpts/icra_p0/s1_${ARM}_s${SEED}"
LOGDIR="$BASE/logs/icra_p0/s1_${ARM}_s${SEED}"
mkdir -p "$OUT" "$LOGDIR"

[[ -f "$BASE_MODEL/config.json" ]] || { echo "missing model $BASE_MODEL"; exit 1; }
[[ -f "$S1_PQ/stage1_train.parquet" ]] || { echo "missing $S1_PQ/stage1_train.parquet"; exit 1; }
[[ "$SEED" == "1" || "$SEED" == "17" || "$SEED" == "31" ]] || { echo "SEED must be 1|17|31"; exit 2; }

FREEZE_S1="${FREEZE_S1:-$ABL/freeze/STAGE1_P0.json}"
FREEZE_CODE="${FREEZE_CODE:-$ABL/freeze/TRAIN_CODE_FREEZE.json}"
PY="${RECREDIT_PYTHON:-$BASE/miniconda3/envs/recredit/bin/python}"

source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONPATH="$LIVE:$BASE/code/recredit_data_engine_autofill:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export BAN_TOKEN_220="${BAN_TOKEN_220:-1}"

echo "==== Stage-I SHA + train-code freeze preflight ====" | tee -a "$LOGDIR/train.log"
"$PY" "$ABL/code/ablations/preflight_stage1_sha.py" \
  --arm "$ARM" --data-root "$S1_PQ" --freeze "$FREEZE_S1" | tee -a "$LOGDIR/train.log"
"$PY" "$ABL/code/ablations/preflight_train_code_freeze.py" \
  --freeze "$FREEZE_CODE" | tee -a "$LOGDIR/train.log"

cd "$LIVE"
# parquet stores relative paths `images/...`; resolve via ER data root
ER_DATA="${ER_DATA:-/data1/dataset/embodied_reasoner/data}"
[[ -d "$ER_DATA/images" ]] || { echo "missing ER images at $ER_DATA/images"; exit 1; }
ln -sfn "$ER_DATA/images" "$LIVE/images"
[[ -e "$LIVE/images/pickup_and_put" ]] || { echo "images symlink broken: $LIVE/images"; exit 1; }

echo "==== $(date -Is) P0 Stage-I arm=$ARM seed=$SEED pq=$S1_PQ out=$OUT ====" | tee -a "$LOGDIR/train.log"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  trainer.py \
  data.train_files="$S1_PQ/stage1_train.parquet" \
  data.val_files="$S1_PQ/stage1_val.parquet" \
  data.train_batch_size=2 data.micro_batch_size_per_gpu=2 \
  data.max_length=8192 data.truncation=left data.max_history_images=3 \
  model.path="$BASE_MODEL" \
  trainer.stage=stage1 trainer.seed="$SEED" \
  trainer.save_dir="$OUT" trainer.total_epochs="$S1_EPOCHS" \
  optim.lr="$LR" \
  dpo.enabled=false \
  2>&1 | tee -a "$LOGDIR/train.log"

# prefer epoch_0 then final
if [[ -d "$OUT/epoch_0" ]]; then
  ln -sfn "$OUT/epoch_0" "$OUT/best"
elif [[ -d "$OUT/final" ]]; then
  ln -sfn "$OUT/final" "$OUT/best"
else
  echo "FATAL no epoch_0/final under $OUT"; exit 1
fi
printf '%s\n' "{\"arm\":\"s1_$ARM\",\"seed\":$SEED,\"best\":\"$OUT/best\",\"pq\":\"$S1_PQ\"}" > "$OUT/RUN.json"
echo "==== done Stage-I $ARM seed=$SEED best=$OUT/best ===="
