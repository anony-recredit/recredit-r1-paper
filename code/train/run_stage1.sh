#!/usr/bin/env bash
# Stage I: geometric warm-start (weighted imitation by q_t) on multi-GPU.
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
CONDA_ROOT="$BASE/miniconda3"
RECIPE="$BASE/code/recredit_r1"
VERL="$BASE/code/verl"
NPROC="${NPROC:-8}"

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONPATH="$RECIPE:$VERL:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
mkdir -p "$BASE/ckpts/stage1" "$BASE/logs"
cd "$RECIPE"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  trainer.py \
  data.train_files="$BASE/data/parquet/stage1_train.parquet" \
  data.val_files="$BASE/data/parquet/stage1_val.parquet" \
  model.path="$BASE/models/Qwen2.5-VL-7B-Instruct" \
  trainer.stage=stage1 \
  trainer.save_dir="$BASE/ckpts/stage1" \
  trainer.total_epochs=3 \
  optim.lr=1.0e-5 \
  "$@"
