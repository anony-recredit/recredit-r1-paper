#!/usr/bin/env python3
"""Select eta_star from the four seed-23 pilots using mean V_pos."""
from __future__ import annotations

import json
from pathlib import Path


def best_vpos(ckpt_root: Path) -> float | None:
    vals = []
    for p in ckpt_root.rglob("val_metrics.json"):
        d = json.loads(p.read_text())
        if d.get("epoch") in (None, "final"):
            continue
        if "val_loss" in d:
            vals.append((int(d["epoch"]), float(d["val_loss"])))
    if not vals:
        return None
    vals.sort()
    # min V_pos, tie -> earlier epoch
    best = min(vals, key=lambda x: (x[1], x[0]))
    return best[1]


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--log-root", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for eta in (0.25, 0.5):
        tag = f"eta{int(round(eta * 100)):03d}"
        vs = []
        for arm in ("neg_uniform", "type_neg"):
            root = args.log_root / tag / f"{arm}_s23_pilot"
            v = best_vpos(root)
            vs.append(v)
            rows.append({"eta": eta, "arm": arm, "v_pos": v, "path": str(root)})
        if None in vs:
            raise SystemExit(f"missing V_pos for eta={eta}: {vs}")
        j = 0.5 * (vs[0] + vs[1])
        rows.append({"eta": eta, "J": j})
    j025 = next(r["J"] for r in rows if r.get("eta") == 0.25 and "J" in r)
    j050 = next(r["J"] for r in rows if r.get("eta") == 0.5 and "J" in r)
    if abs(j025 - j050) <= 1e-4:
        eta_star = 0.25
        reason = "tie_or_within_1e-4_prefer_smaller"
    else:
        eta_star = 0.25 if j025 < j050 else 0.5
        reason = "min_mean_V_pos"
    out = {"rows": rows, "eta_star": eta_star, "reason": reason, "J_025": j025, "J_050": j050}
    dest = args.log_root / "ETA_SELECTION.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
