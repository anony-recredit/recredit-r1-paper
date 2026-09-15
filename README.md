<div align="center">
<h1>
Seeing Wrong or Thinking Wrong?<br/>
Type-Aware Geometric Recrediting for Embodied Manipulation
</h1>
<p><b>Recredit</b> — auditable geometric credit assignment for embodied OTA traces</p>

<p>
<a href="https://huggingface.co/datasets/anony-recredit/recredit-r1-annotations"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" alt="Hugging Face Dataset"/></a>
<a href="https://huggingface.co/anony-recredit/recredit-r1-ckpts"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-green" alt="Hugging Face Model"/></a>
</p>

<p align="center">
  <img src="./assets/method_overview.jpg" width="95%" alt="Recredit method overview"/>
</p>
</div>

## Introduction

Vision-language models for embodied manipulation generate interleaved
observation–thought–action (OTA) trajectories. Common post-training recipes
still broadcast a delayed episode outcome uniformly across steps and channels,
which can **false-praise** off-target actions on successes and **false-blame**
geometrically correct contacts on failures—and cannot tell *seeing wrong* from
*thinking wrong*.

**Recredit** redistributes delayed outcomes along both **time** (which step)
and **type** (grounding vs reasoning) using:

1. key-action coverage outcome \(R_{L2}\)
2. AABB geometric prior \(q_t\) → outcome-conditioned weights \(w_t\)
3. optional type-gated span coefficients (auxiliary)
4. Stage I \(q_t\)-weighted CE → Stage II \(w_t\)-weighted Full CE (+ optional
   attribution DPO inside Stage II)

No human seeing/thinking failure labels are required.

<p align="center">
  <img src="./assets/diagnosis.jpg" width="92%" alt="Delayed-credit diagnosis"/>
</p>

## Annotations & Checkpoints

This GitHub repository ships **code only**. Annotated packs and trained weights
are hosted on Hugging Face (anonymous account):

| Asset | URL |
|-------|-----|
| **Annotations** (recredit.v1 credit fields) | https://huggingface.co/datasets/anony-recredit/recredit-r1-annotations |
| **Checkpoints** (Stage I / Full CE / CE+DPO) | https://huggingface.co/anony-recredit/recredit-r1-ckpts |

```bash
# Dataset (annotations)
huggingface-cli download anony-recredit/recredit-r1-annotations \
  --repo-type dataset --local-dir ./data/recredit-r1-annotations

# Model checkpoints
huggingface-cli download anony-recredit/recredit-r1-ckpts \
  --local-dir ./ckpts/recredit-r1-ckpts
```

Point `${RECREDIT_ROOT}` at a local experiment root that contains (or symlinks)
these downloads before running training / eval scripts.

## Repository Contents

```text
assets/                # README images
code/
  data_engine/         # R_L2 / q_t / w_t / A_t / e_t annotator
  train/               # Stage I / Stage II trainer recipe (verl-style)
  v28_pipeline/        # Stage-II pack / Attr-DPO pair build / n781 eval driver
  failctrl_v3_gr_mass_matched/   # failure-control GR training patches
  rh20t_eval/          # RH20T offline action-matching helpers
evaluation/rh20t/      # RH20T metrics / audits / six-variant runner
```

## Code overview

```bash
export RECREDIT_ROOT=/path/to/your/experiment/root
export EMBODIED_REASONER_ROOT=/path/to/embodied_reasoner_assets   # for data_engine
export RECREDIT_PYTHON=python
```

| Component | Role |
|-----------|------|
| `code/data_engine/` | Offline autofill of auditable credit fields from OTA + AABB metadata |
| `code/train/` | Parquet conversion + Stage I / Stage II CE training entrypoints |
| `code/v28_pipeline/` | Stage-II closerep packs, attribution DPO pair mining, n781 eval wrapper |
| `code/failctrl_v3_gr_mass_matched/` | Optional failure-control GR mass-matched training patches |
| `code/rh20t_eval/` + `evaluation/rh20t/` | Offline RH20T action-matching diagnostic tooling |

## Citation

```bibtex
@inproceedings{recredit2026,
  title     = {Seeing Wrong or Thinking Wrong? Type-Aware Geometric Recrediting for Embodied Manipulation},
  author    = {Anonymous Authors},
  booktitle = {Under double-blind review},
  year      = {2026}
}
```

## Acknowledgment

We thank the Embodied-Reasoner authors for releasing open OTA trajectories.
Generative AI tools were used solely for figure refinement and basic formatting
checks; the authors take full responsibility for all content.
