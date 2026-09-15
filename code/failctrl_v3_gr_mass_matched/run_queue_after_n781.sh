#!/usr/bin/env bash
# Wait for the current n781 eval_after_train job, then run failctrl_v3.
# Never starts failctrl_v2. Never trains on n781/C105/RH20T.
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
V3="$BASE/code/failctrl_v3_gr_mass_matched"
STATUS="$BASE/results/failctrl_v3_gr/STATUS.json"
LOG="$BASE/logs/failctrl_v3_gr/queue_after_n781.log"
mkdir -p "$(dirname "$STATUS")" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

stamp() { date -Is; }
set_status() {
  python3 - <<PY
import json, time
from pathlib import Path
p=Path("$STATUS")
d={"phase": "$1", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": """$2"""}
p.write_text(json.dumps(d, indent=2))
print("STATUS", d)
PY
}

wait_n781_idle() {
  set_status wait_n781_eval "do not steal GPUs from eval_after_train / run_eval_qwen2vl_n781"
  while true; do
    if pgrep -f "eval_after_train_q2.sh" >/dev/null 2>&1 \
      || pgrep -f "run_eval_qwen2vl_n781.sh" >/dev/null 2>&1 \
      || pgrep -f "fill_v27_n781_missing.sh" >/dev/null 2>&1; then
      echo "$(stamp) n781 eval still running; sleep 120"
      sleep 120
      continue
    fi
    # also wait until no leftover trainer from mech Q2
    if pgrep -f "wt_ablation_v5_mech" >/dev/null 2>&1; then
      echo "$(stamp) mech trainer still up; sleep 120"
      sleep 120
      continue
    fi
    echo "$(stamp) n781/mech idle"
    break
  done
}

install_and_patch() {
  set_status install_patch "copy live trainer/dataset then apply v3 GR patches"
  mkdir -p "$V3/live_src"
  cp -a "$BASE/code/recredit_r1/"*.py "$V3/live_src/"
  cp -a "$BASE/code/recredit_r1/config.yaml" "$V3/live_src/config.yaml"
  # Frozen Stage-II hydra config from published Full run
  if [[ -f "$BASE/ckpts/wt_ablation_v5/full/epoch_1/config.yaml" ]]; then
    cp -a "$BASE/ckpts/wt_ablation_v5/full/epoch_1/config.yaml" "$V3/live_src/config.yaml"
  else
    cp -a "$BASE/code/recredit_r1/config.yaml" "$V3/live_src/config.yaml"
  fi
  python3 "$V3/patch_dataset_gr.py" "$V3/live_src/dataset.py"
  python3 "$V3/patch_trainer_v3.py" "$V3/live_src/trainer.py"
  python3 "$V3/test_mass.py"
}

prepare_data() {
  set_status prepare_data "cap-before-route parquets; leak check n781/C105/RH20T"
  local SUC_TR="$BASE/data/parquet_p123_shared/P3_recredit_wt_v5_filt/full/stage2_train.parquet"
  local SUC_VA="$BASE/data/parquet_p123_shared/P3_recredit_wt_v5_filt/full/stage2_val.parquet"
  local FAIL_SRC=""
  for cand in \
      "$BASE/data/parquet_p123_shared/P3_recredit_failctrl_v2/eta025/type_neg/stage2_train.parquet" \
      "$BASE/data/parquet_p123_shared/P3_recredit_failctrl_v2/eta025/neg_uniform/stage2_train.parquet"; do
    if [[ -f "$cand" ]]; then FAIL_SRC="$cand"; break; fi
  done
  [[ -n "$FAIL_SRC" ]] || { echo "missing failctrl_v2 failure source"; exit 1; }
  [[ -f "$SUC_TR" ]] || { echo "missing Full success train"; exit 1; }
  local OUT="$BASE/data/parquet_p123_shared/P3_recredit_failctrl_v3"
  local N781="$BASE/data/eval/test_n781_SMT_C78.json"
  local C105="$BASE/data/eval/test_C_105.json"
  local RH=""
  if [[ -f "$BASE/data/rh20t_recredit_cfg3_pc_n100/all.json" ]]; then
    RH="$BASE/data/rh20t_recredit_cfg3_pc_n100/all.json"
  fi
  for eta in 0.25 0.5; do
    python3 "$V3/prepare_v3_parquets.py" \
      --success-train "$SUC_TR" \
      --success-val "$SUC_VA" \
      --failure-src "$FAIL_SRC" \
      --out-root "$OUT" \
      --eta "$eta" \
      --weight-scale 1.0 \
      --n781-json "$N781" \
      ${C105:+--c105-json "$C105"} \
      ${RH:+--rh20t-json "$RH"}
  done
}

run_pilots() {
  set_status pilots "4 short seed=23 pilots; none count as formal"
  cd "$V3"
  for eta in 0.25 0.5; do
    for arm in neg_uniform type_neg; do
      echo "$(stamp) pilot $arm eta=$eta"
      ARM="$arm" SEED=23 ETA="$eta" PHASE=pilot bash "$V3/run_arm.sh"
    done
  done
  python3 "$V3/select_eta.py" --log-root "$BASE/ckpts/failctrl_v3_gr"
}

run_formal() {
  local eta
  eta=$(python3 - <<'PY'
import json
from pathlib import Path
p=Path(__import__("os").environ["RECREDIT_ROOT"])/"ckpts/failctrl_v3_gr/ETA_SELECTION.json"
print(json.loads(p.read_text())["eta_star"])
PY
)
  set_status formal "six formal trains eta_star=$eta seeds 1/17/31"
  for seed in 1 17 31; do
    for arm in neg_uniform type_neg; do
      echo "$(stamp) formal $arm seed=$seed eta=$eta"
      ARM="$arm" SEED="$seed" ETA="$eta" PHASE=formal bash "$V3/run_arm.sh"
    done
  done
  set_status formal_done "six checkpoints written; next is n781 on the eval host"
}

eval_formal_n781() {
  set_status eval_n781 "same run_eval_qwen2vl_n781.sh + BAN_TOKEN_220=1 as current Q2; scores do not reselect eta/ckpt"
  bash "$V3/eval_formal_n781.sh"
  set_status eval_n781_done "six formal n781 scores on the eval host"
}

wait_n781_idle
install_and_patch
prepare_data
run_pilots
run_formal
eval_formal_n781
echo "$(stamp) failctrl_v3 queue finished"
