#!/usr/bin/env bash
# After all P0 evals: build paired Geo vs U2 stats for seeds 1/17/31.
set -euo pipefail
BASE="${RECREDIT_ROOT:?set RECREDIT_ROOT}"
ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
PY="$BASE/miniconda3/envs/recredit/bin/python"
RESULTS="${RESULTS:-$BASE/results/icra_p0}"
OUT="${OUT:-$RESULTS/P0_GEO_VS_U2_STATS.json}"
MANIFEST="$ABL/freeze/TASK_MANIFEST.json"
SCENE="$ABL/freeze/SCENE_MAP.json"

PAIRS=()
for SEED in 1 17 31; do
  A="$RESULTS/p0_full_geo_s${SEED}/task_success.csv"
  B="$RESULTS/p0_full_u2_s${SEED}/task_success.csv"
  [[ -f "$A" && -f "$B" ]] || { echo "missing $A or $B"; exit 1; }
  PAIRS+=(--pair "${SEED}:${A}:${B}")
done

"$PY" "$ABL/code/ablations/stats_scene_cluster_signflip.py" \
  --scene-map "$SCENE" \
  --manifest "$MANIFEST" \
  "${PAIRS[@]}" \
  --out "$OUT"
echo "wrote $OUT"
