#!/usr/bin/env bash
# P0 Stage-II Full CE positive-only from matching-seed Stage-I (geo or u2).
set -euo pipefail
ARM="${ARM:?set ARM=geo|u2}"
SEED="${SEED:?set SEED=1|17|31}"
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
LIVE="$BASE/code/recredit_r1_qwen3"
PY="$BASE/miniconda3/envs/recredit/bin/python"
NPROC="${NPROC:-8}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5.0e-6}"

S1_BEST="${S1_BEST:-$BASE/ckpts/icra_p0/s1_${ARM}_s${SEED}/best}"
# Same Full CE pack as paper Qwen3 full arm
DATA_ROOT="${DATA_ROOT:-$BASE/data/parquet_p123_shared/P3_recredit_wt_v5_filt/full}"
OUT="$BASE/ckpts/icra_p0/full_${ARM}_s${SEED}"
LOGDIR="$BASE/logs/icra_p0/full_${ARM}_s${SEED}"
mkdir -p "$OUT" "$LOGDIR"

[[ -f "$S1_BEST/config.json" ]] || { echo "missing S1 $S1_BEST"; exit 1; }
[[ -f "$DATA_ROOT/stage2_train.parquet" ]] || { echo "missing $DATA_ROOT"; exit 1; }
[[ "$SEED" == "1" || "$SEED" == "17" || "$SEED" == "31" ]] || { echo "bad SEED"; exit 2; }

ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
FREEZE_S2="${FREEZE_S2:-$ABL/freeze/STAGE2_FULL_CE.json}"

source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONPATH="$LIVE:$BASE/code/recredit_data_engine_autofill:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export BAN_TOKEN_220="${BAN_TOKEN_220:-1}"
# signed_ce env is inherited by trainer; positive-only preflight MUST pass first
# so the negative branch is unreachable on this pack.
export NEG_CREDIT_MODE=signed_ce

echo "==== positive-only + train-code freeze preflight ====" | tee -a "$LOGDIR/train.log"
"$PY" "$ABL/code/ablations/preflight_stage2_positive_only.py" \
  --data-root "$DATA_ROOT" \
  --freeze "$FREEZE_S2" | tee -a "$LOGDIR/train.log"
"$PY" "$ABL/code/ablations/preflight_train_code_freeze.py" \
  --freeze "${FREEZE_CODE:-$ABL/freeze/TRAIN_CODE_FREEZE.json}" | tee -a "$LOGDIR/train.log"

cd "$LIVE"
ER_DATA="${ER_DATA:-/data1/dataset/embodied_reasoner/data}"
[[ -d "$ER_DATA/images" ]] || { echo "missing ER images at $ER_DATA/images"; exit 1; }
ln -sfn "$ER_DATA/images" "$LIVE/images"
[[ -e "$LIVE/images/pickup_and_put" ]] || { echo "images symlink broken: $LIVE/images"; exit 1; }

echo "==== $(date -Is) P0 Full CE arm=$ARM seed=$SEED s1=$S1_BEST out=$OUT ====" | tee -a "$LOGDIR/train.log"
rm -rf "$OUT"/epoch_* "$OUT"/final 2>/dev/null || true
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  trainer.py \
  data.train_files="$DATA_ROOT/stage2_train.parquet" \
  data.val_files="$DATA_ROOT/stage2_val.parquet" \
  data.train_batch_size=2 data.micro_batch_size_per_gpu=2 \
  data.max_length=8192 data.truncation=left data.max_history_images=3 \
  model.path="$S1_BEST" \
  trainer.stage=stage2 trainer.seed="$SEED" \
  trainer.save_dir="$OUT" trainer.total_epochs="$EPOCHS" \
  optim.lr="$LR" \
  dpo.enabled=false \
  2>&1 | tee -a "$LOGDIR/train.log"

"$PY" - <<PY
import json
from pathlib import Path
root = Path("$OUT")
link = root / "best"
cands = []
for p in sorted(root.glob("epoch_*"), key=lambda q: int(q.name.split("_")[1])) + [root / "final"]:
    m = p / "val_metrics.json"
    if p.exists() and m.exists():
        cands.append((float(json.loads(m.read_text()).get("val_loss", 1e9)), p))
if not cands:
    raise SystemExit(f"no checkpoint with val_metrics under {root}")
cands.sort(key=lambda c: c[0])
best_loss, best = cands[0]
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to(best.resolve())
meta = {
    "arm": "full_$ARM",
    "seed": int("$SEED"),
    "s1": "$S1_BEST",
    "data": "$DATA_ROOT",
    "best": str(best.resolve()),
    "best_val_loss": best_loss,
    "selection": "val_loss",
    "recipe": "Full CE positive-only (paper Qwen3 full)",
}
(root / "BEST.json").write_text(json.dumps(meta, indent=2))
print("BEST", link, "->", best)
PY

# Disk policy: keep only Full best (~50G); drop sibling epochs and Stage-I for this arm/seed.
KEEP_S1="${KEEP_S1:-0}"
S1_DIR="$BASE/ckpts/icra_p0/s1_${ARM}_s${SEED}"
echo "==== prune Full extras + Stage-I (KEEP_S1=$KEEP_S1) ====" | tee -a "$LOGDIR/train.log"
BEST_REAL="$(readlink -f "$OUT/best")"
[[ -d "$BEST_REAL" ]] || { echo "FATAL best missing: $OUT/best -> $BEST_REAL"; exit 1; }
[[ -f "$BEST_REAL/config.json" ]] || { echo "FATAL best incomplete: $BEST_REAL"; exit 1; }
shopt -s nullglob
for d in "$OUT"/epoch_* "$OUT"/final; do
  [[ -e "$d" ]] || continue
  if [[ "$(readlink -f "$d")" == "$BEST_REAL" ]]; then
    continue
  fi
  echo "rm -rf $d" | tee -a "$LOGDIR/train.log"
  rm -rf "$d"
done
# ensure best symlink still valid
if [[ ! -e "$OUT/best" ]]; then
  ln -sfn "$BEST_REAL" "$OUT/best"
fi
if [[ "$KEEP_S1" != "1" ]]; then
  echo "rm -rf $S1_DIR" | tee -a "$LOGDIR/train.log"
  rm -rf "$S1_DIR"
fi
du -sh "$OUT" "$OUT/best" 2>/dev/null | tee -a "$LOGDIR/train.log" || true
df -h /data2 | tee -a "$LOGDIR/train.log" || true

echo "==== done Full $ARM seed=$SEED best=$OUT/best ===="
