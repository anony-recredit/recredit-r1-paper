#!/usr/bin/env bash
set -euo pipefail

ABL="${RECREDIT_ABLATIONS:?set RECREDIT_ABLATIONS}"
if pgrep -f 'run_p0.*eval|evaluate.py|local_deploy.py|torchrun.*trainer.py' >/dev/null 2>&1; then
  echo "GPU workload detected; refusing to overlap P1 with training/evaluation"
  exit 2
fi

for seed in 1 17 31; do
  for arm in attr type_agnostic; do
    ARM="$arm" SEED="$seed" bash "$ABL/code/ablations/run_p1_dpo.sh"
  done
done
