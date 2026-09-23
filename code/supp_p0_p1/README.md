# Supplementary P0 / P1 controls (isolated)

Train/eval drivers for two **page-budget supplementary** controls that
strengthen claims already made in the main paper (geometry-informed
credit; type-aware preference direction). Kept separate from
`code/v28_pipeline/` and `code/train/` so the shipping recipe and these
controls do not mix.

Requires environment variables (no hardcoded host paths):

```bash
export RECREDIT_ROOT=/path/to/experiment/root
export RECREDIT_ABLATIONS=/path/to/this/or/sibling/ablation/root   # optional freeze trees
export P1_ROOT=/path/to/p1/experiment/root                        # P1 data/ckpts
export RECREDIT_PYTHON=python
```

## Scripts

| Area | Scripts |
|------|---------|
| P0 Stage I / Full | `run_p0_stage1.sh`, `run_p0_stage2_full.sh`, `run_p0_all_seeds.sh` |
| P0 mass-match U2 | `build_mass_matched_uniform_stage1_parquet.py`, `rebuild_u2_parquet.sh` |
| P0 n781 eval | `run_p0_eval_n781.sh`, `run_p0_eval_x3.sh`, `run_p0_eval_all.sh`, `score_n781_strict.py` |
| P1 pairs / DPO | `p1_pairs.py`, `build_p1_pairs.py`, `run_p1_dpo.sh`, `prepare_p1_all.sh` |
| P1 n781 eval | `run_p1_eval_n781.sh`, `run_p1_eval_all.sh`, `run_p1_resume_eval.sh` |
| Stats | `run_p0_aggregate_stats.sh`, `run_p1_aggregate_stats.sh`, `stats_scene_cluster_signflip.py` |

Numbers and motivation: repo-root README § *Supplementary P0 / P1*;
figures under `assets/supp_p0_p1/`.
