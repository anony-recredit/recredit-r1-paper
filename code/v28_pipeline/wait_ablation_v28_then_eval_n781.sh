#!/usr/bin/env bash
# Wait for ablation_v28 train to finish, then auto-run n781 eval ONLY
# (Stage2 no-DPO + Attr-DPO). NEVER runs n806.
set -euo pipefail

BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
RECIPE=$BASE/code/recredit_r1
CKPT_ROOT=${CKPT_ROOT:-${RECREDIT_ROOT}/ckpts/ablation_v28}
STATUS=$BASE/results/ablation_v28/STATUS.json
LOGDIR=$BASE/logs/ablation_v28
RESULTS=$BASE/results/ablation_v28
EVAL_SH=$RECIPE/run_eval_qwen2vl_n781.sh
INPUT=$BASE/data/eval/test_n781_SMT_C78.json
PY=$BASE/miniconda3/envs/recredit/bin/python
POLL_SEC="${POLL_SEC:-60}"

mkdir -p "$LOGDIR" "$RESULTS"
exec >>"$LOGDIR/wait_then_eval_n781.log" 2>&1

echo "==== $(date -Is) waiter start: v28 train_done → n781 eval ONLY ===="
echo "CKPT_ROOT=$CKPT_ROOT INPUT=$INPUT"
[[ -f "$INPUT" ]] || { echo "FATAL missing n781 pack $INPUT"; exit 1; }
[[ -f "$EVAL_SH" ]] || { echo "FATAL missing $EVAL_SH"; exit 1; }

# Hard refuse accidental n806 wiring
case "$INPUT" in
  *n806*|*/test_809.json)
    echo "FATAL waiter misconfigured with n806 input: $INPUT"; exit 2 ;;
esac
"$PY" - <<PY
import json
from pathlib import Path
n=len(json.loads(Path("$INPUT").read_text()))
assert n==781, n
print("n781 pack OK", n)
PY

wait_train_done() {
  while true; do
    phase=""
    if [[ -f "$STATUS" ]]; then
      phase=$("$PY" -c "import json;print(json.load(open('$STATUS')).get('phase',''))" 2>/dev/null || true)
    fi
    s2_ok=0; dpo_ok=0
    [[ -f "$CKPT_ROOT/stage2_recredit/best/config.json" ]] && s2_ok=1
    [[ -f "$CKPT_ROOT/stage2_recredit_dpo/best/config.json" ]] && dpo_ok=1
    busy=0
    if pgrep -f "trainer.py.*ablation_v28|trainer.py.*stage2_recredit_dpo|trainer.py.*P3_recredit_ablation_v28" >/dev/null 2>&1; then
      busy=1
    fi
    # also treat pipeline bash still in DPO train as busy
    if pgrep -f "run_ablation_v28_pipeline.sh" >/dev/null 2>&1 && [[ "$phase" != "ablation_v28_train_done" ]]; then
      # pipeline may still be writing MANIFEST after trainer exits; keep waiting unless phase says done
      :
    fi
    echo "$(date -Is) phase=$phase s2=$s2_ok dpo=$dpo_ok trainer_busy=$busy"
    if [[ "$phase" == "ablation_v28_train_done" && "$s2_ok" == "1" && "$dpo_ok" == "1" && "$busy" == "0" ]]; then
      return 0
    fi
    if [[ -f "$CKPT_ROOT/MANIFEST.json" && "$s2_ok" == "1" && "$dpo_ok" == "1" && "$busy" == "0" ]]; then
      echo "MANIFEST present; treating as train done"
      return 0
    fi
    sleep "$POLL_SEC"
  done
}

resolve_ckpt() {
  local arm="$1"
  if [[ -f "$CKPT_ROOT/$arm/best/config.json" ]]; then
    echo "$CKPT_ROOT/$arm/best"
  elif [[ -f "$CKPT_ROOT/$arm/final/config.json" ]]; then
    echo "$CKPT_ROOT/$arm/final"
  else
    local last
    last=$(ls -d "$CKPT_ROOT/$arm"/epoch_* 2>/dev/null | sort -V | tail -1 || true)
    [[ -n "$last" && -f "$last/config.json" ]] && echo "$last" && return 0
    return 1
  fi
}

wait_train_done
sleep 20

printf '%s' '{"phase":"ablation_v28_eval_n781_running","universe":"n781"}' > "$STATUS"

for ARM in stage2_recredit stage2_recredit_dpo; do
  CKPT=$(resolve_ckpt "$ARM") || { echo "FATAL missing ckpt for $ARM"; exit 1; }
  NAME="ablation_v28_${ARM}_n781"
  echo "==== $(date -Is) EVAL n781 $NAME from $CKPT ===="
  # Explicit env — never point at n806 scripts
  RESULTS="$RESULTS" \
  EVAL_NAME="$NAME" \
  EVAL_MODEL="$CKPT" \
  INPUT_PATH="$INPUT" \
  BAN_TOKEN_220=1 \
  bash "$EVAL_SH" || echo "WARN eval failed for $ARM"
done

echo "==== $(date -Is) score n781 pack (v28 + baselines; no n806 eval) ===="
SCORE_ARGS=(--report-dir "$BASE/results/eval_pack_n781")
# prior baselines already have result trees; scoring subsets to n781 identities
[[ -d "$BASE/results/p3_test809/p3_v26_dpo_test809_legacy" ]] && \
  SCORE_ARGS+=(--run v26_DPO "$BASE/results/p3_test809/p3_v26_dpo_test809_legacy")
[[ -d "$BASE/results/ablation_v27/ablation_v27_stage2_recredit_n806" ]] && \
  SCORE_ARGS+=(--run v27_noDPO "$BASE/results/ablation_v27/ablation_v27_stage2_recredit_n806")
[[ -d "$BASE/results/ablation_v27/ablation_v27_stage2_recredit_dpo_n806" ]] && \
  SCORE_ARGS+=(--run v27_DPO "$BASE/results/ablation_v27/ablation_v27_stage2_recredit_dpo_n806")
SCORE_ARGS+=(--run v28_noDPO "$RESULTS/ablation_v28_stage2_recredit_n781")
SCORE_ARGS+=(--run v28_DPO "$RESULTS/ablation_v28_stage2_recredit_dpo_n781")
"$PY" "$RECIPE/build_and_score_eval_pack_n781.py" "${SCORE_ARGS[@]}" \
  2>&1 || echo "WARN build_and_score_eval_pack_n781 failed"

# Always write local scoreboard too
"$PY" - <<PY
import json
from pathlib import Path
root = Path("$RESULTS")
rows = []
for name in ["ablation_v28_stage2_recredit_n781", "ablation_v28_stage2_recredit_dpo_n781"]:
    p = root / name / "N781_SR.json"
    if p.exists():
        rows.append(json.loads(p.read_text()))
out = {
    "universe": "test_n781_SMT_C78",
    "n": 781,
    "arms": rows,
    "note": "n806 NOT run by wait_ablation_v28_then_eval_n781.sh",
}
(root / "SCOREBOARD_n781_ablation_v28.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
PY

printf '%s' '{"phase":"ablation_v28_eval_n781_done","universe":"n781"}' > "$STATUS"
echo "==== $(date -Is) ALL n781 evals done (n806 skipped) ===="
