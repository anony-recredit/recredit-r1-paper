# Recredit-R1 data engine (autofill)

Convert Embodied-Reasoner multiturn OTA JSON into `recredit.v1` records with
rule-derived \(R_{L2}\), \(q_t\), \(w_t\), and type proxies (no human
seeing/thinking labels).

```bash
export EMBODIED_REASONER_ROOT=/path/to/embodied_reasoner_assets
export ER_ROOT="$EMBODIED_REASONER_ROOT"
MODE=local ./scripts/run_autofill.sh
```

Acceptance: validation report `ok: true` and `n_errors: 0`.
