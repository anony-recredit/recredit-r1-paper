#!/usr/bin/env bash
set -euo pipefail

BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
RESULTS="${RESULTS:-$BASE/results/icra_p1}"
REPS="${REPS:-1 2 3}"

for rep in $REPS; do
  pairs=()
  for seed in 1 17 31; do
    attr="$RESULTS/p1_attr_s${seed}_r${rep}/task_success.csv"
    control="$RESULTS/p1_type_agnostic_s${seed}_r${rep}/task_success.csv"
    [[ -f "$attr" && -f "$control" ]] || { echo "missing rep=$rep seed=$seed results"; exit 1; }
    pairs+=(--pair "${seed}:${attr}:${control}")
  done
  "$PY" "$ABL/code/ablations/stats_scene_cluster_signflip.py" \
    --scene-map "$ABL/freeze/SCENE_MAP.json" \
    --manifest "$ABL/freeze/TASK_MANIFEST.json" \
    "${pairs[@]}" --arm-a-name attr --arm-b-name type_agnostic \
    --out "$RESULTS/P1_ATTR_VS_TYPE_AGNOSTIC_R${rep}_STATS.json"
done
