# Recredit-R1 data engine (autofill)

Convert Embodied-Reasoner multiturn OTA JSON into `recredit.v1` records with
rule-derived \(R_{L2}\), \(q_t\), \(w_t\), and type proxies (no human
seeing/thinking labels).

## Paper-faithful coefficients (Eqs. 7–8 → Eq. 10)

| Symbol | Paper | Code |
|--------|-------|------|
| \(w_t\) | outcome-conditioned spatial weight | `spatial_weights` |
| \(s_t^k\) | independently normalized type scores (Eq. 7) | `type_scores` |
| \(A_t^k\) | \(((1-\lambda)w_t+\lambda s_t^k)\,A_t\) (Eq. 8) | `gated_advantages` |
| \(\mathcal{L}^{\mathrm{CE}}_{II}\) | \(-\sum_t(A_t^{\mathrm{perc}}\log\pi_P+\hat A_t^{\mathrm{reas}}\log\pi_R)\) (Eq. 10) | Stage-II parquet uses `A_t_*` as token weights; `trainer.py` does signed weighted CE |

Canonical helper: `paper_eq8_coefficients` in `reward.py`.  
Tests: `python test_eq8_eq10.py`.

> **Note:** `code/failctrl_v3_gr_mass_matched/` is an *optional* failure-aware
> unlikelihood protocol (η / mass-matched). It is **not** Eq. (10).

```bash
export EMBODIED_REASONER_ROOT=/path/to/embodied_reasoner_assets
export ER_ROOT="$EMBODIED_REASONER_ROOT"
MODE=local ./scripts/run_autofill.sh
```

Acceptance: validation report `ok: true` and `n_errors: 0`.
