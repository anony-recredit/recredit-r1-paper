#!/usr/bin/env bash
# Ablation v28 — v27 skeleton, SKIP Stage1, Stage2(v24-pos+closerep) → Attr-DPO(quotas)
#
# Disk layout:
#   code/data  : ${RECREDIT_ROOT}
#   ckpts      : ${CKPT_ROOT}   (largest free disk)
#   stage1 init: ${S1_INIT}
#
# Arms:
#   stage2_recredit/best
#   stage2_recredit_dpo/best
#   MANIFEST.json
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
RECIPE=$BASE/code/recredit_r1
PY=$BASE/miniconda3/envs/recredit/bin/python
LOGDIR=$BASE/logs/ablation_v28
CKPT_ROOT=${CKPT_ROOT:-${RECREDIT_ROOT}/ckpts/ablation_v28}
CE_PQ=$BASE/data/parquet_p123_shared/P3_recredit_ablation_v28_stage2
DPO_PQ_DIR=$BASE/data/parquet_p123_shared/P3_recredit_ablation_v28_dpo
S1_INIT="${S1_INIT:-${RECREDIT_ROOT}/ckpts/ablation_v27/stage1/best}"
NPROC="${NPROC:-8}"
SKIP_S2="${SKIP_S2:-0}"
SKIP_DPO="${SKIP_DPO:-0}"
PREPARE_CE="${PREPARE_CE:-1}"
DPO_LAMBDA="${DPO_LAMBDA:-0.3}"
DPO_BETA="${DPO_BETA:-0.1}"
S2_EPOCHS="${S2_EPOCHS:-2}"
DPO_EPOCHS="${DPO_EPOCHS:-1}"
STATUS=$BASE/results/ablation_v28/STATUS.json
mkdir -p "$LOGDIR" "$CKPT_ROOT" "$(dirname "$STATUS")" "$BASE/results/ablation_v28"
# optional discoverability symlink
mkdir -p "${RECREDIT_ROOT}/ckpts"
ln -sfn "$CKPT_ROOT" "${RECREDIT_ROOT}/ckpts/ablation_v28"
exec > >(tee -a "$LOGDIR/pipeline.log") 2>&1

echo "==== $(date -Is) ABLATION_V28 START ckpt_root=$CKPT_ROOT s1=$S1_INIT ===="
df -h "${RECREDIT_ROOT}" 2>/dev/null || true
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONPATH="$RECIPE:$BASE/code/recredit_data_engine_autofill:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export BAN_TOKEN_220="${BAN_TOKEN_220:-1}"

[[ -f "$S1_INIT/config.json" ]] || { echo "missing stage1 init $S1_INIT"; exit 1; }

write_status() { printf '%s' "$1" > "$STATUS"; }

pick_best_by_val() {
  local ckpt_root="$1"
  local best_link="$2"
  "$PY" - <<PY
import json
from pathlib import Path
root = Path("$ckpt_root")
link = Path("$best_link")
epochs = sorted(root.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
final = root / "final"
cands = []
for p in epochs + ([final] if final.exists() else []):
    m = p / "val_metrics.json"
    if m.exists():
        j = json.loads(m.read_text())
        cands.append((float(j.get("val_loss", 1e9)), p))
if cands:
    cands.sort()
    best = cands[0][1]
else:
    best = final if final.exists() else (epochs[-1] if epochs else None)
if best is None:
    raise SystemExit(f"no ckpt under {root}")
link.parent.mkdir(parents=True, exist_ok=True)
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to(best.resolve())
print("BEST", link, "->", best)
(root / "BEST.json").write_text(json.dumps({"best": str(best), "link": str(link)}, indent=2))
PY
}

if [[ "$PREPARE_CE" == "1" ]]; then
  bash "$RECIPE/prepare_v28_stage2_data.sh"
fi
[[ -f "$CE_PQ/stage2_train.parquet" ]] || { echo "missing $CE_PQ"; exit 1; }

# ---------- Stage1 pointer (frozen; not trained) ----------
S1_OUT="$CKPT_ROOT/stage1"
mkdir -p "$S1_OUT"
ln -sfn "$(readlink -f "$S1_INIT")" "$S1_OUT/best"
S1_BEST="$S1_OUT/best"
echo "FROZEN_S1 $S1_BEST -> $(readlink -f "$S1_BEST")"

# ---------- Stage 2 Recredit (NO DPO) ----------
S2_OUT="$CKPT_ROOT/stage2_recredit"
if [[ "$SKIP_S2" == "1" && -f "$S2_OUT/best/config.json" ]]; then
  echo "SKIP_S2 reuse $S2_OUT/best"
else
  write_status "{\"phase\":\"ablation_v28_stage2_recredit\",\"from\":\"$S1_BEST\"}"
  rm -rf "$S2_OUT"/epoch_* "$S2_OUT"/final 2>/dev/null || true
  mkdir -p "$S2_OUT"
  cd "$RECIPE"
  torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    trainer.py \
    data.train_files="$CE_PQ/stage2_train.parquet" \
    data.val_files="$CE_PQ/stage2_val.parquet" \
    data.train_batch_size=2 data.micro_batch_size_per_gpu=2 \
    data.max_length=8192 data.truncation=left data.max_history_images=3 \
    model.path="$S1_BEST" \
    trainer.stage=stage2 trainer.save_dir="$S2_OUT" trainer.total_epochs="$S2_EPOCHS" \
    optim.lr=5.0e-6 \
    dpo.enabled=false \
    2>&1 | tee -a "$LOGDIR/stage2_recredit.log"
  pick_best_by_val "$S2_OUT" "$S2_OUT/best"
fi
S2_BEST="$S2_OUT/best"
[[ -f "$S2_BEST/config.json" ]] || { echo "missing S2 best"; exit 1; }

# ---------- Attr-DPO ----------
DPO_OUT="$CKPT_ROOT/stage2_recredit_dpo"
DPO_PQ=$DPO_PQ_DIR/dpo_train_with_ref.parquet
if [[ "$SKIP_DPO" == "1" && -f "$DPO_OUT/best/config.json" ]]; then
  echo "SKIP_DPO reuse $DPO_OUT/best"
else
  export CKPT_REF="$S2_BEST"
  export CKPT_ROOT
  bash "$RECIPE/prepare_v28_dpo_data.sh"
  [[ -f "$DPO_PQ" ]] || { echo "missing $DPO_PQ"; exit 1; }

  write_status "{\"phase\":\"ablation_v28_stage2_attr_dpo\",\"from\":\"$S2_BEST\",\"lambda\":$DPO_LAMBDA}"
  rm -rf "$DPO_OUT"/epoch_* "$DPO_OUT"/final 2>/dev/null || true
  mkdir -p "$DPO_OUT"
  cd "$RECIPE"
  torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    trainer.py \
    data.train_files="$CE_PQ/stage2_train.parquet" \
    data.val_files="$CE_PQ/stage2_val.parquet" \
    data.train_batch_size=2 data.micro_batch_size_per_gpu=2 \
    data.max_length=8192 data.truncation=left data.max_history_images=3 \
    model.path="$S2_BEST" \
    trainer.stage=stage2 trainer.save_dir="$DPO_OUT" trainer.total_epochs="$DPO_EPOCHS" \
    optim.lr=5.0e-6 \
    dpo.enabled=true \
    dpo.files="$DPO_PQ" \
    dpo.beta="$DPO_BETA" \
    dpo.lambda_="$DPO_LAMBDA" \
    dpo.batch_size=1 \
    dpo.smoke_every=0 \
    2>&1 | tee -a "$LOGDIR/stage2_attr_dpo.log"
  pick_best_by_val "$DPO_OUT" "$DPO_OUT/best"
fi
DPO_BEST="$DPO_OUT/best"

"$PY" - <<PY
import json, time
from pathlib import Path
root = Path("$CKPT_ROOT")
manifest = {
  "name": "ablation_v28",
  "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "skeleton": "v27 (Stage2 recredit attribution -> Attr-DPO)",
  "changes_vs_v27": [
    "Stage2 CE = v24 pos skills + closerep upsample/window",
    "DPO quotas: C_skill~38% + closerep_seeing~28% + seeing_wrong<=8%",
    "Stage1 frozen from ablation_v27/stage1/best (not retrained)",
    "ckpts under $CKPT_ROOT",
  ],
  "test_set_used_in_training": False,
  "ckpt_selection": "validation loss only",
  "arms": {
    "B_stage1_frozen": {"path": str(Path("$S1_BEST").resolve()), "role": "frozen geometric warm-start"},
    "C_stage2_recredit_no_dpo": {"path": str(Path("$S2_BEST").resolve()), "role": "Recredit SFT with closerep aug"},
    "D_stage2_recredit_attr_dpo": {"path": str(Path("$DPO_BEST").resolve()) if Path("$DPO_BEST").exists() else None, "role": "quota Attr-DPO"},
  },
  "data": {
    "stage2_ce_parquet": "$CE_PQ",
    "attr_dpo_parquet": "$DPO_PQ",
  },
  "hyper": {"dpo_lambda": float("$DPO_LAMBDA"), "dpo_beta": float("$DPO_BETA"),
            "s2_epochs": int("$S2_EPOCHS"), "dpo_epochs": int("$DPO_EPOCHS")},
  "gates_n781": {"C78_ge": 12, "S_closerep_ge": 40, "M_closerep_ge": 18},
}
(root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY

write_status "{\"phase\":\"ablation_v28_train_done\"}"
echo "==== $(date -Is) ABLATION_V28 PIPELINE EXIT ===="
df -h "${RECREDIT_ROOT}" 2>/dev/null || true
