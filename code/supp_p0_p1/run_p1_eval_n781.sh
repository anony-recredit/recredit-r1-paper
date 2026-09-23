#!/usr/bin/env bash
set -euo pipefail

ARM="${ARM:?set ARM=attr|type_agnostic}"
SEED="${SEED:?set SEED=1|17|31}"
REP="${REP:-1}"
[[ "$ARM" == "attr" || "$ARM" == "type_agnostic" ]] || { echo "bad ARM=$ARM"; exit 2; }
[[ "$SEED" == "1" || "$SEED" == "17" || "$SEED" == "31" ]] || { echo "bad SEED=$SEED"; exit 2; }
[[ "$REP" =~ ^[1-9][0-9]*$ ]] || { echo "bad REP=$REP"; exit 2; }

BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
P1_ROOT="${P1_ROOT:?set P1_ROOT}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
MODEL="$P1_ROOT/ckpts/p1_${ARM}_s${SEED}/best"
NAME="p1_${ARM}_s${SEED}_r${REP}"
RESULTS="${RESULTS:-$BASE/results/icra_p1}"
EVAL_SH="$BASE/code/recredit_r1_wt_ablation_v5_qwen3/run_eval_qwen3vl_n781.sh"

[[ -f "$MODEL/config.json" ]] || { echo "missing $MODEL"; exit 1; }
export RECREDIT_ROOT="$BASE" EVAL_NAME="$NAME" EVAL_MODEL="$MODEL" RESULTS
bash "$EVAL_SH"
"$PY" "$ABL/code/ablations/score_n781_strict.py" \
  --results-dir "$RESULTS/$NAME" \
  --manifest "$ABL/freeze/TASK_MANIFEST.json" \
  --name "$NAME" --model "$MODEL" \
  --out-json "$RESULTS/$NAME/N781_SR.json" \
  --out-task-csv "$RESULTS/$NAME/task_success.csv"
