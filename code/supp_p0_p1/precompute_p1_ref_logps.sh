#!/usr/bin/env bash
set -euo pipefail

ARM="${ARM:?set ARM=attr|type_agnostic}"
SEED="${SEED:?set SEED=1|17|31}"
[[ "$ARM" == "attr" || "$ARM" == "type_agnostic" ]] || { echo "bad ARM=$ARM"; exit 2; }
[[ "$SEED" == "1" || "$SEED" == "17" || "$SEED" == "31" ]] || { echo "bad SEED=$SEED"; exit 2; }

BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
P1_ROOT="${P1_ROOT:?set P1_ROOT}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
PRECOMP="$BASE/code/recredit_r1_qwen3/precompute_dpo_ref_logps.py"
PAIR_DIR="$P1_ROOT/data/pairs"
PAIR_FILE="$PAIR_DIR/${ARM}_unweighted.parquet"
PAIR_MANIFEST="$PAIR_DIR/P1_PAIR_MANIFEST.json"
OUT="$P1_ROOT/data/ref/${ARM}_s${SEED}"
SHARDS="$OUT/shards"
LOGDIR="$P1_ROOT/logs/ref_${ARM}_s${SEED}"
DEST="$OUT/dpo_train_with_ref.parquet"
META="$OUT/dpo_train_with_ref.ref_meta.json"
NPROC="${NPROC:-8}"

case "$SEED" in
  1) FULL_BEST="$BASE/ckpts/wt_ablation_v5_qwen3/full/best" ;;
  17|31) FULL_BEST="$BASE/ckpts/icra_p0/full_geo_s${SEED}/best" ;;
esac

[[ -f "$PAIR_FILE" ]] || { echo "missing $PAIR_FILE"; exit 1; }
[[ -f "$FULL_BEST/config.json" ]] || { echo "missing $FULL_BEST"; exit 1; }
[[ -f "$PRECOMP" ]] || { echo "missing $PRECOMP"; exit 1; }
"$PY" "$ABL/code/ablations/preflight_p1_ref.py" \
  --arm "$ARM" --seed "$SEED" \
  --training-freeze "$ABL/freeze/P1_TRAINING.json" \
  --pair-manifest "$PAIR_MANIFEST" --full-checkpoint "$FULL_BEST"
if [[ -f "$DEST" && -f "$META" ]]; then
  "$PY" "$ABL/code/ablations/preflight_p1.py" \
    --arm "$ARM" --seed "$SEED" \
    --training-freeze "$ABL/freeze/P1_TRAINING.json" \
    --pair-manifest "$PAIR_MANIFEST" \
    --ref-parquet "$DEST" --ref-meta "$META" \
    --full-checkpoint "$FULL_BEST"
  echo "SKIP verified existing ref artifact: $OUT"
  exit 0
fi
if [[ -e "$DEST" || -e "$META" ]]; then
  echo "partial ref artifact requires manual inspection: $OUT"
  exit 3
fi
mkdir -p "$SHARDS" "$LOGDIR"

source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONPATH="$BASE/code/recredit_r1_qwen3:$BASE/code/recredit_data_engine_autofill:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

"$PY" - "$PAIR_FILE" "$SHARDS" "$NPROC" <<'PY'
import sys
from pathlib import Path
import pandas as pd

source, out, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
frame = pd.read_parquet(source)
for index in range(count):
    frame.iloc[index::count].reset_index(drop=True).to_parquet(out / f"pairs_shard{index}.parquet", index=False)
print({"n": len(frame), "n_shards": count})
PY

pids=()
for index in $(seq 0 $((NPROC - 1))); do
  CUDA_VISIBLE_DEVICES="$index" "$PY" "$PRECOMP" \
    --pairs "$SHARDS/pairs_shard${index}.parquet" \
    --model "$FULL_BEST" \
    --out "$SHARDS/ref_shard${index}.parquet" \
    --device 0 --batch-size 1 \
    >"$LOGDIR/ref_shard${index}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || { tail -40 "$LOGDIR"/ref_shard*.log; exit 4; }

"$PY" - "$PAIR_FILE" "$SHARDS" "$DEST" "$META" "$FULL_BEST" "$NPROC" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

source, shards, dest, meta, ref, count = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), int(sys.argv[6])
master = pd.read_parquet(source)
parts = [pd.read_parquet(shards / f"ref_shard{i}.parquet") for i in range(count)]
merged = pd.concat(parts, ignore_index=True).set_index("pair_id").loc[master["pair_id"].tolist()].reset_index()
for column in master.columns:
    if column not in merged.columns:
        merged[column] = master[column].values
merged.to_parquet(dest, index=False)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
payload={
    "n":len(merged),
    "ref_ckpt":str(ref.resolve()),
    "source_pairs":str(source),
    "source_pairs_sha256":sha(source),
    "output":str(dest),
    "output_sha256":sha(dest),
}
meta.write_text(json.dumps(payload,indent=2,sort_keys=True))
print(json.dumps(payload,indent=2,sort_keys=True))
PY
