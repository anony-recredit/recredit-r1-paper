#!/usr/bin/env bash
set -euo pipefail

ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
REPS="${REPS:-1 2 3}"
for rep in $REPS; do
  for seed in 1 17 31; do
    for arm in attr type_agnostic; do
      ARM="$arm" SEED="$seed" REP="$rep" bash "$ABL/code/ablations/run_p1_eval_n781.sh"
    done
  done
done
