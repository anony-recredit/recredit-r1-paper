# Ablation v28 — Stage-II closerep + attribution DPO

## Narrative
- Keep Stage-II Recredit fields \(q_t,w_t,e_t\); Attr-DPO only optimizes attributed preferences.
- Freeze Stage-I (`ablation_v27/stage1/best`).
- Stage-II data: long-horizon positives + closerep upsample / window clips.
- DPO: attribution gates + hard quotas (thinking / closerep seeing); cap generic seeing.

## Paths
| Use | Path |
|-----|------|
| code / data / logs | `${RECREDIT_ROOT}` |
| checkpoints | `${CKPT_ROOT}` |
| Stage-I init | `${S1_INIT}` |

## DPO quotas (targets)
| Bucket | Fraction |
|--------|----------|
| holding+bridge+seg2 (long-horizon thinking) | ~38% |
| closerep_seeing | ~28% |
| thinking_wrong | ~18% |
| anti_loop | ~8% |
| seeing_wrong (generic) | ≤8% (hard cap) |

Audit gates: `C_skill≥30%`, `closerep≥20%`, `seeing≤15%`.

## Launch
```bash
# Build Stage-II pack + parquet + audit (no training)
bash code/v28_pipeline/prepare_v28_stage2_data.sh

# Full pipeline: pack → Stage-II CE → quota Attr-DPO
bash code/v28_pipeline/run_ablation_v28_pipeline.sh

# Skip CE rebuild when parquet already exists
PREPARE_CE=0 bash code/v28_pipeline/run_ablation_v28_pipeline.sh
```

## Post-train gate (n781)
- Prefer improving overall / short / mid over Stage-I
- Avoid large regressions on open short/mid slices
