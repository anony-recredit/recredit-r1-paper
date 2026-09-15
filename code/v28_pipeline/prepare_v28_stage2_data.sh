#!/usr/bin/env bash
# v28 Stage2 data: closerep-augmented pack → pos-only recredit parquet
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
RECIPE=$BASE/code/recredit_r1
PY=$BASE/miniconda3/envs/recredit/bin/python
LOGDIR=$BASE/logs/ablation_v28
SHARED=$BASE/data/wave23_shared_p123_v28
OUT_PQ=$BASE/data/parquet_p123_shared/P3_recredit_ablation_v28_stage2
mkdir -p "$LOGDIR" "$OUT_PQ"
exec > >(tee -a "$LOGDIR/prepare_stage2_data.log") 2>&1

echo "==== $(date -Is) v28 STAGE2 DATA PREPARE START ===="
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONPATH="$RECIPE:$BASE/code/recredit_data_engine_autofill:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "==== A) build v24-pos + closerep pack ===="
"$PY" "$RECIPE/build_v28_stage2_closerep_pack.py"

echo "==== B) convert parquet (recredit, neg_scale=0) ===="
rm -f "$OUT_PQ"/stage2_*.parquet "$OUT_PQ"/meta.json 2>/dev/null || true
"$PY" "$RECIPE/convert_wave3_presplit.py" \
  --train-json "$SHARED/shared_train.json" \
  --val-json "$SHARED/shared_val.json" \
  --data-root ${EMBODIED_REASONER_ROOT} \
  --out-dir "$OUT_PQ" \
  --stage stage2 \
  --clip-ordered 16 --clip-ultra 22 \
  --credit-mode recredit \
  --neg-scale 0.0 --neg-cap 0.0 \
  --neg-clip 0.0

"$PY" - <<'PY'
import json
from pathlib import Path
import pandas as pd
PQ=Path(__import__("os").environ["RECREDIT_ROOT"])/"data/parquet_p123_shared/P3_recredit_ablation_v28_stage2"
for name in ["stage2_train.parquet","stage2_val.parquet"]:
    df=pd.read_parquet(PQ/name, columns=["perc_weight","reas_weight"])
    pw=df["perc_weight"].astype(float); rw=df["reas_weight"].astype(float)
    rep={"file":name,"n":len(df),"perc_neg":int((pw<0).sum()),"reas_neg":int((rw<0).sum()),
         "perc_min":float(pw.min()),"reas_min":float(rw.min()),"perc_mean":float(pw.mean())}
    print(rep)
    assert rep["perc_neg"]==0 and rep["reas_neg"]==0, rep
print("PARQUET_POS_ONLY_OK")
PY

echo "==== C) pack audit (pre-DPO) ===="
"$PY" "$RECIPE/audit_v28_quotas.py" || true

echo "==== $(date -Is) v28 STAGE2 DATA PREPARE DONE ===="
