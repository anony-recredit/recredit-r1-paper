#!/usr/bin/env bash
# Prepare + train + n781-eval type_neg_invert (seed=1, eta*=0.25).
# Paper-symmetric Invert on failure type routing; mass-matched to type_neg.
# Uses free GPUs 2-7 by default (0-1 often occupied).
set -euo pipefail

BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
V3="$BASE/code/failctrl_v3_gr_mass_matched"
PY="${RECREDIT_PYTHON:-$BASE/miniconda3/envs/recredit/bin/python}"
EVAL_SH="$BASE/code/recredit_r1/run_eval_qwen2vl_n781.sh"
FILL_SH="$BASE/code/recredit_r1/fill_v27_n781_missing.sh"
RESULTS="$BASE/results/failctrl_v3_gr/n781"
STATUS="$BASE/results/failctrl_v3_gr/STATUS.json"
LOG="$BASE/logs/failctrl_v3_gr/type_neg_invert_s1_pipeline.log"
SCALE="16.23397888580789"
ETA=0.25
SEED=1
ARM=type_neg_invert
NAME="${ARM}_s${SEED}_formal"
GPU_LIST="${GPU_LIST:-2,3,4,5,6,7}"
NPROC="${NPROC:-6}"

mkdir -p "$RESULTS" "$RESULTS/logs" "$(dirname "$LOG")" "$(dirname "$STATUS")"
exec > >(tee -a "$LOG") 2>&1

stamp() { date -Is; }
set_status() {
  "$PY" - <<PY
import json, time
from pathlib import Path
Path("$STATUS").write_text(json.dumps({
    "phase": "$1",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "note": """$2""",
}, indent=2))
print("STATUS", "$1")
PY
}

score_one() {
  local name="$1"
  "$PY" - <<PY
import json
from pathlib import Path
name = "$name"
root = Path("$RESULTS") / name
uni = json.loads(Path("$BASE/data/eval/test_n781_SMT_C78.json").read_text())
tasks = uni if isinstance(uni, list) else (uni.get("examples") or uni.get("data") or [])
want = {str(e.get("identity")) for e in tasks}
have = {}
for p in root.rglob("result.json"):
    try:
        d = json.loads(p.read_text())
        i = str(d.get("identity"))
        m = d.get("metrics") or {}
        if "success" in m:
            have[i] = bool(m["success"])
    except Exception:
        pass
miss = sorted(want - set(have), key=lambda x: int(x) if x.isdigit() else x)
succ = sum(1 for v in have.values() if v)
out = {
    "name": name,
    "n": 781,
    "succ": succ,
    "sr_pct": round(100.0 * succ / 781, 2),
    "have": len(have),
    "missing": len(miss),
    "missing_ids": miss,
    "root": str(root),
    "eta_star": float("$ETA"),
    "arm": "type_neg_invert",
    "seed": int("$SEED"),
}
Path("$RESULTS/score_{}.json".format(name)).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY
}

cleanup_eval() {
  echo "$(stamp) cleanup orphan THOR / local_deploy"
  PIDS=$(ps -eo pid,ppid,cmd --no-headers | awk '$2==1' | grep thor-CloudRendering | awk '{print $1}' || true)
  [[ -n "${PIDS:-}" ]] && kill -9 $PIDS 2>/dev/null || true
  pkill -f "local_deploy.py" 2>/dev/null || true
  sleep 8
}

# ---- 1) prepare invert parquet (also refreshes uni/type; mass-checked) ----
set_status prepare_invert "eta=$ETA type_neg_invert mass-matched"
SUC_TR="$BASE/data/parquet_p123_shared/P3_recredit_wt_v5_filt/full/stage2_train.parquet"
SUC_VA="$BASE/data/parquet_p123_shared/P3_recredit_wt_v5_filt/full/stage2_val.parquet"
FAIL_SRC="$BASE/data/parquet_p123_shared/P3_recredit_failctrl_v2/eta025/type_neg/stage2_train.parquet"
OUT="$BASE/data/parquet_p123_shared/P3_recredit_failctrl_v3"
N781="$BASE/data/eval/test_n781_SMT_C78.json"
C105_ARG=()
[[ -f "$BASE/data/eval/test_C_105.json" ]] && C105_ARG=(--c105-json "$BASE/data/eval/test_C_105.json")
"$PY" "$V3/prepare_v3_parquets.py" \
  --success-train "$SUC_TR" \
  --success-val "$SUC_VA" \
  --failure-src "$FAIL_SRC" \
  --out-root "$OUT" \
  --eta "$ETA" \
  --weight-scale "$SCALE" \
  --n781-json "$N781" \
  "${C105_ARG[@]}"

# Sanity: perc/reas swapped vs type_neg, same m_t
"$PY" - <<PY
import pandas as pd
from pathlib import Path
root = Path("$OUT/eta025")
a = pd.read_parquet(root / "type_neg/failure_train.parquet")
b = pd.read_parquet(root / "type_neg_invert/failure_train.parquet")
assert len(a) == len(b)
assert (a["m_t"] - b["m_t"]).abs().max() < 1e-12
# perc/reas should be swapped (up to float noise)
d1 = (a["perc_weight"].abs() - b["reas_weight"].abs()).abs().max()
d2 = (a["reas_weight"].abs() - b["perc_weight"].abs()).abs().max()
assert d1 < 1e-9 and d2 < 1e-9, (d1, d2)
print("OK invert is type_neg with perc/reas swapped; mass identical")
print("type_neg perc/reas", float(a["perc_weight"].abs().sum()), float(a["reas_weight"].abs().sum()))
print("invert  perc/reas", float(b["perc_weight"].abs().sum()), float(b["reas_weight"].abs().sum()))
PY

# ---- 2) train ----
set_status train_invert "CUDA_VISIBLE_DEVICES=$GPU_LIST NPROC=$NPROC $NAME"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export NPROC
ARM="$ARM" SEED="$SEED" ETA="$ETA" PHASE=formal bash "$V3/run_arm.sh"

# ---- 3) eval n781 ----
set_status eval_invert "n781 $NAME"
CKROOT="$BASE/ckpts/failctrl_v3_gr/eta025"
model=$("$PY" "$V3/resolve_best_ckpt.py" "$CKROOT/$NAME")
echo "$(stamp) EVAL $NAME model=$model"
[[ -f "$model/config.json" ]] || { echo "FATAL no model $model"; exit 1; }
echo "{\"name\":\"$NAME\",\"model\":\"$model\",\"arm\":\"type_neg_invert\",\"seed\":$SEED,\"eta\":$ETA}" \
  > "$RESULTS/SELECTED_${NAME}.json"
cleanup_eval
env EVAL_NAME="$NAME" EVAL_MODEL="$model" RESULTS="$RESULTS" BAN_TOKEN_220=1 \
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  bash "$EVAL_SH"

score_one "$NAME"
sc="$RESULTS/score_${NAME}.json"
miss=$("$PY" -c "import json; print(json.load(open('$sc'))['missing'])")
if [[ "$miss" != "0" ]]; then
  echo "$(stamp) fill missing=$miss"
  env EVAL_NAME="$NAME" EVAL_MODEL="$model" RESULTS="$RESULTS" BAN_TOKEN_220=1 \
    CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    bash "$FILL_SH" || true
  score_one "$NAME"
fi

set_status invert_s1_done "type_neg_invert_s1_formal scored; see $RESULTS/score_${NAME}.json"
echo "$(stamp) DONE"
cat "$RESULTS/score_${NAME}.json"
