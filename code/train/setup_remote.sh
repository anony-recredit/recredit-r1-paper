#!/usr/bin/env bash
# Bootstrap Recredit-R1 + verl on the local multi-GPU workstation.
set -euo pipefail

BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
CONDA_ROOT="$BASE/miniconda3"
ENV_NAME="recredit"
PYTHON_VER="3.10"
VERL_TAG="v0.4.1"
VERL_DIR="$BASE/code/verl"
MODEL_DIR="$BASE/models/Qwen2.5-VL-7B-Instruct"
LOG="$BASE/logs/setup.log"

mkdir -p "$BASE"/{code,models,data/{raw,parquet},ckpts,logs,envs}
exec > >(tee -a "$LOG") 2>&1

echo "==== $(date -Is) setup start ===="
echo "BASE=$BASE"

export PATH="$CONDA_ROOT/bin:$PATH"
export HF_HOME="${HF_HOME:-$BASE/envs/hf_home}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
  echo "==== install miniconda ===="
  INST="$BASE/Miniconda3-py310.sh"
  if [[ ! -f "$INST" ]]; then
    wget -O "$INST" https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh
  fi
  bash "$INST" -b -p "$CONDA_ROOT"
fi

# conda defaults
"$CONDA_ROOT/bin/conda" config --set always_yes true
"$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
"$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

if ! "$CONDA_ROOT/bin/conda" env list | grep -q "^${ENV_NAME} "; then
  echo "==== create conda env ${ENV_NAME} ===="
  "$CONDA_ROOT/bin/conda" create -n "$ENV_NAME" "python=${PYTHON_VER}" -y
fi

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -V
pip install -U pip setuptools wheel

echo "==== clone verl ${VERL_TAG} ===="
if [[ ! -d "$VERL_DIR/.git" ]]; then
  git clone --branch "$VERL_TAG" --depth 1 https://github.com/verl-project/verl.git "$VERL_DIR"
else
  git -C "$VERL_DIR" fetch --depth 1 origin "refs/tags/${VERL_TAG}:refs/tags/${VERL_TAG}" || true
  git -C "$VERL_DIR" checkout "$VERL_TAG" || true
fi

echo "==== install torch cu124 ===="
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo "==== install verl python deps (FSDP SFT, no Megatron) ===="
pip install \
  "transformers==4.51.3" \
  "accelerate>=1.0.0" \
  "peft>=0.14.0" \
  "datasets" \
  "dill" \
  "hydra-core" \
  "omegaconf" \
  "codetiming" \
  "numpy" \
  "pandas" \
  "pyarrow>=15.0.0" \
  "pylatexenc" \
  "ray[default]" \
  "torchdata" \
  "wandb" \
  "orjson" \
  "tensordict" \
  "einops" \
  "qwen-vl-utils" \
  "sentencepiece" \
  "protobuf" \
  "packaging" \
  "ninja"

echo "==== install verl package (no-deps, keep our torch) ===="
pip install --no-deps -e "$VERL_DIR"

echo "==== try flash-attn wheel (optional) ===="
MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation || \
  echo "WARN: flash-attn install failed; trainer will use sdpa"

echo "==== download Qwen2.5-VL-7B-Instruct ===="
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  export MODEL_DIR
  python - <<'PY'
import os
d = os.environ["MODEL_DIR"]
print("downloading to", d)
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", local_dir=d)
print("done")
PY
fi

python - <<'PY'
import torch, transformers, verl
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("transformers", transformers.__version__)
print("verl", verl.__file__)
print("gpu count", torch.cuda.device_count())
PY

echo "==== $(date -Is) setup finished ===="
