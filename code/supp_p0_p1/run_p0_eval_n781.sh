#!/usr/bin/env bash
# Run frozen Qwen3 n781 eval then STRICT score (metrics.success + manifest ID equality).
set -euo pipefail
ARM="${ARM:?geo|u2}"
SEED="${SEED:?1|17|31}"
# Optional replicate index. Unset keeps the in-flight chain name p0_full_${ARM}_s${SEED}.
REP="${REP:-}"
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
if [[ -n "$REP" ]]; then
  [[ "$REP" =~ ^[0-9]+$ ]] || { echo "REP must be a positive integer, got $REP"; exit 2; }
  NAME="p0_full_${ARM}_s${SEED}_r${REP}"
else
  NAME="p0_full_${ARM}_s${SEED}"
fi
MODEL="${MODEL:-$BASE/ckpts/icra_p0/full_${ARM}_s${SEED}/best}"
RESULTS="${RESULTS:-$BASE/results/icra_p0}"
EVAL_SH="$BASE/code/recredit_r1_wt_ablation_v5_qwen3/run_eval_qwen3vl_n781.sh"
MANIFEST="${MANIFEST:-$ABL/freeze/TASK_MANIFEST.json}"

mkdir -p "$RESULTS"
export RECREDIT_ROOT="$BASE"
export EVAL_NAME="$NAME"
export EVAL_MODEL="$MODEL"
export RESULTS
bash "$EVAL_SH"

# Overwrite any broken N781_SR.json from eval script with strict scorer
"$PY" "$ABL/code/ablations/score_n781_strict.py" \
  --results-dir "$RESULTS/$NAME" \
  --manifest "$MANIFEST" \
  --name "$NAME" \
  --model "$MODEL" \
  --out-json "$RESULTS/$NAME/N781_SR.json" \
  --out-task-csv "$RESULTS/$NAME/task_success.csv"

echo "==== scored $NAME ===="
cat "$RESULTS/$NAME/N781_SR.json"
