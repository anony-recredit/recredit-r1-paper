#!/usr/bin/env bash
set -euo pipefail

ARM="${ARM:?set ARM=attr|type_agnostic}"
SEED="${SEED:?set SEED=1|17|31}"
[[ "$ARM" == "attr" || "$ARM" == "type_agnostic" ]] || { echo "bad ARM=$ARM"; exit 2; }
[[ "$SEED" == "1" || "$SEED" == "17" || "$SEED" == "31" ]] || { echo "bad SEED=$SEED"; exit 2; }

BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
P1_ROOT="${P1_ROOT:?set P1_ROOT}"
LIVE="$BASE/code/recredit_r1_qwen3"
PY="$BASE/miniconda3/envs/recredit/bin/python"
PAIR_MANIFEST="$P1_ROOT/data/pairs/P1_PAIR_MANIFEST.json"
REF_DIR="$P1_ROOT/data/ref/${ARM}_s${SEED}"
REF_PQ="$REF_DIR/dpo_train_with_ref.parquet"
REF_META="$REF_DIR/dpo_train_with_ref.ref_meta.json"
CE_PQ="$BASE/data/parquet_p123_shared/P3_recredit_wt_v5_filt/full"
OUT="$P1_ROOT/ckpts/p1_${ARM}_s${SEED}"
LOGDIR="$P1_ROOT/logs/train_${ARM}_s${SEED}"
NPROC="${NPROC:-8}"

case "$SEED" in
  1) FULL_BEST="$BASE/ckpts/wt_ablation_v5_qwen3/full/best" ;;
  17|31) FULL_BEST="$BASE/ckpts/icra_p0/full_geo_s${SEED}/best" ;;
esac

mkdir -p "$OUT" "$LOGDIR"

"$PY" "$ABL/code/ablations/preflight_stage2_positive_only.py" \
  --data-root "$CE_PQ" --freeze "$ABL/freeze/STAGE2_FULL_CE.json"
"$PY" "$ABL/code/ablations/preflight_p1.py" \
  --arm "$ARM" --seed "$SEED" \
  --training-freeze "$ABL/freeze/P1_TRAINING.json" \
  --pair-manifest "$PAIR_MANIFEST" \
  --ref-parquet "$REF_PQ" --ref-meta "$REF_META" \
  --full-checkpoint "$FULL_BEST"

if [[ -f "$OUT/BEST.json" && -f "$OUT/best/config.json" ]]; then
  echo "SKIP verified inputs; completed checkpoint already exists: $OUT/best"
  exit 0
fi
if find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .; then
  echo "partial checkpoint directory requires manual inspection: $OUT"
  exit 3
fi

avail=$(df -BG --output=avail /data3 | tail -1 | tr -dc 0-9)
[[ "$avail" -ge 250 ]] || { echo "FATAL /data3 only ${avail}G free"; exit 1; }
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN OK arm=$ARM seed=$SEED full=$FULL_BEST ref=$REF_PQ out=$OUT"
  exit 0
fi

GO_FILE="$ABL/freeze/P1_GO.json"
"$PY" - "$GO_FILE" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing P1 GO gate: {path}")
payload = json.loads(path.read_text())
if payload.get("status") != "GO":
    raise SystemExit(f"P1 training blocked: gate status={payload.get('status')!r}")
print("P1 GO gate accepted")
PY

source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONPATH="$LIVE:$BASE/code/recredit_data_engine_autofill:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export BAN_TOKEN_220=1
export NEG_CREDIT_MODE=signed_ce

cd "$LIVE"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  trainer.py \
  data.train_files="$CE_PQ/stage2_train.parquet" \
  data.val_files="$CE_PQ/stage2_val.parquet" \
  data.train_batch_size=2 data.micro_batch_size_per_gpu=2 \
  data.max_length=8192 data.truncation=left data.max_history_images=3 \
  model.path="$FULL_BEST" \
  trainer.stage=stage2 trainer.seed="$SEED" \
  trainer.save_dir="$OUT" trainer.total_epochs=1 \
  optim.lr=5.0e-6 \
  dpo.enabled=true dpo.files="$REF_PQ" \
  dpo.beta=0.1 dpo.lambda_=0.3 dpo.batch_size=1 dpo.smoke_every=0 \
  2>&1 | tee -a "$LOGDIR/train.log"

"$PY" - "$OUT" "$ARM" "$SEED" "$FULL_BEST" "$REF_PQ" "$PAIR_MANIFEST" "$P1_ROOT" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

root, arm, seed, full, ref, manifest, p1_root = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6]), Path(sys.argv[7])
allowed = (p1_root / "ckpts").resolve()
if allowed not in root.resolve().parents:
    raise SystemExit(f"unsafe checkpoint root: {root}")
candidates=[]
for path in list(root.glob("epoch_*")) + [root / "final"]:
    metrics=path/"val_metrics.json"
    if path.is_dir() and metrics.is_file() and (path/"config.json").is_file():
        candidates.append((float(json.loads(metrics.read_text())["val_loss"]),path))
if not candidates: raise SystemExit(f"no complete checkpoints under {root}")
candidates.sort(key=lambda item:item[0]); loss,best=candidates[0]
link=root/"best"
if link.exists() or link.is_symlink(): link.unlink()
link.symlink_to(best.resolve())
for _,path in candidates[1:]:
    if path.resolve()!=best.resolve(): shutil.rmtree(path)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
payload={"arm":arm,"seed":seed,"from":str(full.resolve()),"best":str(best.resolve()),"best_val_loss":loss,"pair_weight":1.0,"ref_parquet":str(ref),"ref_parquet_sha256":sha(ref),"pair_manifest_sha256":sha(manifest),"dpo_beta":0.1,"dpo_lambda":0.3}
(root/"BEST.json").write_text(json.dumps(payload,indent=2,sort_keys=True))
print(json.dumps(payload,indent=2,sort_keys=True))
PY
