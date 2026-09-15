#!/usr/bin/env bash
# Formal TYPE-NEG n781 on the eval host. Same evaluator as current Q2 mech evals.
# Does not select eta or checkpoints from these scores.
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
V3="$BASE/code/failctrl_v3_gr_mass_matched"
EVAL_SH="$BASE/code/recredit_r1/run_eval_qwen2vl_n781.sh"
FILL_SH="$BASE/code/recredit_r1/fill_v27_n781_missing.sh"
PY="$BASE/miniconda3/envs/recredit/bin/python"
RESULTS="$BASE/results/failctrl_v3_gr/n781"
CKROOT="$BASE/ckpts/failctrl_v3_gr"
ETA_JSON="$CKROOT/ETA_SELECTION.json"
mkdir -p "$RESULTS" "$RESULTS/logs"

eta=$("$PY" -c "import json; print(json.load(open('$ETA_JSON'))['eta_star'])")
tag=$("$PY" -c "print(f'eta{int(round(float(\"$eta\")*100)):03d}')")
echo "formal n781 eta_star=$eta tag=$tag"

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
    "eta_star": float("$eta"),
}
Path("$RESULTS/score_{}.json".format(name)).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY
}

cleanup() {
  echo "$(date -Is) cleanup orphan THOR / local_deploy (keep 7860)"
  PIDS=$(ps -eo pid,ppid,cmd --no-headers | awk '$2==1' | grep thor-CloudRendering | awk '{print $1}' || true)
  [[ -n "${PIDS:-}" ]] && kill -9 $PIDS 2>/dev/null || true
  pkill -f "local_deploy.py" 2>/dev/null || true
  sleep 8
}

NAMES=()
for seed in 1 17 31; do
  for arm in neg_uniform type_neg; do
    NAMES+=("${arm}_s${seed}_formal")
  done
done

for name in "${NAMES[@]}"; do
  run_root="$CKROOT/$tag/$name"
  model=$("$PY" "$V3/resolve_best_ckpt.py" "$run_root")
  echo "==== $(date -Is) EVAL $name model=$model ===="
  [[ -f "$model/config.json" ]] || { echo "FATAL no $model"; exit 1; }
  echo "{\"name\":\"$name\",\"model\":\"$model\"}" > "$RESULTS/SELECTED_${name}.json"
  cleanup
  env EVAL_NAME="$name" EVAL_MODEL="$model" RESULTS="$RESULTS" BAN_TOKEN_220=1 \
    bash "$EVAL_SH"
  score_one "$name"
  miss=$("$PY" -c "import json; print(json.load(open('$RESULTS/score_${name}.json'))['missing'])")
  if [[ "$miss" -gt 0 ]]; then
    echo "==== $(date -Is) FILL $name missing=$miss ===="
    cleanup
    env EVAL_NAME="$name" EVAL_MODEL="$model" RESULTS="$RESULTS" BAN_TOKEN_220=1 MAX_SHARDS=8 \
      bash "$FILL_SH" || echo "WARN fill nonzero $name"
    score_one "$name"
  fi
done

cleanup
"$PY" - <<PY
import json
from pathlib import Path
root = Path("$RESULTS")
rows = []
for seed in (1, 17, 31):
    uni = json.loads((root / f"score_neg_uniform_s{seed}_formal.json").read_text())
    typ = json.loads((root / f"score_type_neg_s{seed}_formal.json").read_text())
    rows.append({
        "seed": seed,
        "neg_uniform": uni["sr_pct"],
        "type_neg": typ["sr_pct"],
        "diff_pp": round(typ["sr_pct"] - uni["sr_pct"], 2),
        "uni_succ": uni["succ"],
        "type_succ": typ["succ"],
        "uni_missing": uni["missing"],
        "type_missing": typ["missing"],
    })
out = {"eta_star": float("$eta"), "rows": rows, "host": "anonymous", "protocol": "run_eval_qwen2vl_n781.sh BAN_TOKEN_220=1"}
(root / "SR_TABLE_formal.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY
echo "==== $(date -Is) formal n781 done ===="
