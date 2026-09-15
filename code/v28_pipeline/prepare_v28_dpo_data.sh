#!/usr/bin/env bash
# v28 Attr-DPO data: quota pairs + ref logps from Stage2 (no-DPO) best.
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
RECIPE=$BASE/code/recredit_r1
PY=$BASE/miniconda3/envs/recredit/bin/python
LOGDIR=$BASE/logs/ablation_v28
OUT=$BASE/data/parquet_p123_shared/P3_recredit_ablation_v28_dpo
CKPT_ROOT=${CKPT_ROOT:-${RECREDIT_ROOT}/ckpts/ablation_v28}
CKPT_REF="${CKPT_REF:-$CKPT_ROOT/stage2_recredit/best}"
mkdir -p "$LOGDIR" "$OUT/shards"
exec > >(tee -a "$LOGDIR/prepare_dpo_data.log") 2>&1

echo "==== $(date -Is) v28 ATTR-DPO DATA PREPARE START ref=$CKPT_REF ===="
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate recredit
export PYTHONPATH="$RECIPE:$BASE/code/recredit_data_engine_autofill:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

[[ -f "$CKPT_REF/config.json" ]] || { echo "missing ref ckpt $CKPT_REF"; exit 1; }

echo "==== A) build quota attribution pairs ===="
"$PY" "$RECIPE/build_attr_dpo_pairs_v28.py"

echo "==== B) quota audit gate ===="
"$PY" "$RECIPE/audit_v28_quotas.py"

echo "==== C) shard + precompute ref logps (8gpu) ===="
"$PY" - <<PY
import pandas as pd
from pathlib import Path
src=Path("$OUT/dpo_train.parquet")
df=pd.read_parquet(src)
out=Path("$OUT/shards"); out.mkdir(parents=True, exist_ok=True)
for i in range(8):
    part=df.iloc[i::8].reset_index(drop=True)
    part.to_parquet(out/f"pairs_shard{i}.parquet", index=False)
    print(i, len(part))
PY

pids=()
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i "$PY" "$RECIPE/precompute_dpo_ref_logps.py" \
    --pairs "$OUT/shards/pairs_shard${i}.parquet" \
    --model "$CKPT_REF" \
    --out "$OUT/shards/ref_shard${i}.parquet" \
    --device 0 \
    --batch-size 1 \
    >"$LOGDIR/ref_shard${i}.log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
[[ $fail -eq 0 ]] || { echo "ref shard failed"; tail -40 "$LOGDIR"/ref_shard*.log; exit 1; }

"$PY" - <<PY
import json, pandas as pd
from pathlib import Path
out=Path("$OUT")
dfs=[pd.read_parquet(out/f"shards/ref_shard{i}.parquet") for i in range(8)]
df=pd.concat(dfs, ignore_index=True)
master=pd.read_parquet(out/"dpo_train.parquet")
df=df.set_index("pair_id").loc[master["pair_id"].tolist()].reset_index()
for c in ("q_t","w_t","e_t","pair_weight","pair_kind"):
    if c in master.columns:
        df[c]=master[c].values
dest=out/"dpo_train_with_ref.parquet"
df.to_parquet(dest, index=False)
meta={"n":len(df),"mean_margin":float((df["ref_logp_chosen"]-df["ref_logp_rejected"]).mean()),"out":str(dest),"ref_ckpt":"$CKPT_REF"}
(out/"dpo_train_with_ref.ref_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
PY

"$PY" "$RECIPE/audit_v28_quotas.py"
echo "==== $(date -Is) v28 ATTR-DPO DATA PREPARE DONE ===="
