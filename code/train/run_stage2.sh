#!/usr/bin/env bash
# Stage II: outcome-guided recrediting (A_perc / A_reas_hat) from Stage I checkpoint.
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
CONDA_ROOT="$BASE/miniconda3"
RECIPE="$BASE/code/recredit_r1"
VERL="$BASE/code/verl"
NPROC="${NPROC:-8}"
STAGE1="${STAGE1_CKPT:-$BASE/ckpts/stage1/final}"
if [[ ! -f "$STAGE1/config.json" ]]; then
  echo "Stage I HF export not found at $STAGE1; falling back to base VLM"
  STAGE1="$BASE/models/Qwen2.5-VL-7B-Instruct"
fi

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONPATH="$RECIPE:$VERL:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
mkdir -p "$BASE/ckpts/stage2" "$BASE/logs"
cd "$RECIPE"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  trainer.py \
  data.train_files="$BASE/data/parquet/stage2_train.parquet" \
  data.val_files="$BASE/data/parquet/stage2_val.parquet" \
  model.path="$STAGE1" \
  trainer.stage=stage2 \
  trainer.save_dir="$BASE/ckpts/stage2" \
  trainer.total_epochs=2 \
  optim.lr=5.0e-6 \
  "$@"
