#!/usr/bin/env python3
"""Horizon SR for failctrl_v3 type_neg vs neg_uniform (paper SMT bins).

Canonical (same as Table I / PREREG):
  Short: n_ka < 4
  Mid:   4 <= n_ka <= 6
  Longer: n_ka > 6
Universe: test_n781_SMT_C78.json (396 / 271 / 114)
n_ka from result.json key_actions (pack itself has no key_actions).
CPU-only rescore — no re-eval.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

BASE = Path(os.environ["RECREDIT_ROOT"])
UNIVERSE = BASE / "data/eval/test_n781_SMT_C78.json"
ROOT = BASE / "results/failctrl_v3_gr/n781"
OUT = ROOT / "HORIZON_SMT_type_neg_vs_neg_uniform.json"

ARMS = {
    "type_neg": ["type_neg_s1_formal", "type_neg_s17_formal", "type_neg_s31_formal"],
    "neg_uniform": [
        "neg_uniform_s1_formal",
        "neg_uniform_s17_formal",
        "neg_uniform_s31_formal",
    ],
}
HORIZONS = ("Short", "Mid", "Longer")


def horizon_of(n_ka: int) -> str:
    if n_ka < 4:
        return "Short"
    if n_ka <= 6:
        return "Mid"
    return "Longer"


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "sd": None, "vals": []}
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return {"mean": round(m, 2), "sd": None, "vals": [round(v, 2) for v in vals]}
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return {
        "mean": round(m, 2),
        "sd": round(math.sqrt(var), 2),
        "vals": [round(v, 2) for v in vals],
    }


def load_universe_ids():
    data = json.loads(UNIVERSE.read_text())
    tasks = data if isinstance(data, list) else (data.get("examples") or data.get("data") or [])
    return {str(t.get("identity")) for t in tasks}


def build_n_ka_map(uni_ids):
    """Canonical n_ka from a complete scored run (key_actions length)."""
    ref = ROOT / "type_neg_s1_formal"
    out = {}
    for p in ref.rglob("result.json"):
        j = json.loads(p.read_text())
        iid = str(j.get("identity"))
        if iid not in uni_ids:
            continue
        ka = j.get("key_actions") or []
        out[iid] = len(ka)
    assert len(out) == len(uni_ids), (len(out), len(uni_ids))
    return out


def load_run(name, uni_nka):
    rdir = ROOT / name
    out = {}
    for p in rdir.rglob("result.json"):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        iid = str(j.get("identity"))
        if iid not in uni_nka:
            continue
        m = j.get("metrics") or {}
        succ = 1 if m.get("success") in (1, True, "1", "true", "True") else 0
        n_ka = uni_nka[iid]
        out[iid] = {"success": succ, "n_ka": n_ka, "horizon": horizon_of(n_ka)}
    return out


def summarize(name, items, uni_nka):
    uni_h = {h: 0 for h in HORIZONS}
    for n_ka in uni_nka.values():
        uni_h[horizon_of(n_ka)] += 1
    have = set(items)
    miss = sorted(set(uni_nka) - have, key=lambda x: int(x) if x.isdigit() else x)
    by_h = {h: {"succ": 0, "n_scored": 0} for h in HORIZONS}
    for v in items.values():
        h = v["horizon"]
        by_h[h]["n_scored"] += 1
        by_h[h]["succ"] += v["success"]
    complete = len(have) == len(uni_nka)
    for h in HORIZONS:
        bn = uni_h[h] if complete else by_h[h]["n_scored"]
        bs = by_h[h]["succ"]
        by_h[h]["n"] = bn
        by_h[h]["sr_pct"] = round(100.0 * bs / bn, 2) if bn else None
    succ = sum(v["success"] for v in items.values())
    n = len(uni_nka) if complete else len(have)
    return {
        "name": name,
        "n": n,
        "have": len(have),
        "succ": succ,
        "sr_pct": round(100.0 * succ / n, 2) if n else None,
        "missing": len(miss),
        "by_horizon": by_h,
        "universe_horizon_n": uni_h,
    }


def main():
    uni_ids = load_universe_ids()
    uni_nka = build_n_ka_map(uni_ids)
    uni_h = {h: 0 for h in HORIZONS}
    for n_ka in uni_nka.values():
        uni_h[horizon_of(n_ka)] += 1
    assert uni_h == {"Short": 396, "Mid": 271, "Longer": 114}, uni_h

    per_run = {}
    for arm, names in ARMS.items():
        for name in names:
            per_run[name] = summarize(name, load_run(name, uni_nka), uni_nka)

    means = {}
    for arm, names in ARMS.items():
        overall = [per_run[n]["sr_pct"] for n in names]
        by_h = {
            h: mean_sd([per_run[n]["by_horizon"][h]["sr_pct"] for n in names])
            for h in HORIZONS
        }
        means[arm] = {"overall": mean_sd(overall), "by_horizon": by_h}

    deltas = {"overall": [], "by_horizon": {h: [] for h in HORIZONS}}
    for i in range(3):
        tn, nu = ARMS["type_neg"][i], ARMS["neg_uniform"][i]
        deltas["overall"].append(round(per_run[tn]["sr_pct"] - per_run[nu]["sr_pct"], 2))
        for h in HORIZONS:
            deltas["by_horizon"][h].append(
                round(
                    per_run[tn]["by_horizon"][h]["sr_pct"]
                    - per_run[nu]["by_horizon"][h]["sr_pct"],
                    2,
                )
            )
    delta_summary = {
        "overall": mean_sd(deltas["overall"]),
        "by_horizon": {h: mean_sd(deltas["by_horizon"][h]) for h in HORIZONS},
        "per_seed_pp": deltas,
    }

    report = {
        "protocol": "failctrl_v3_gr_mass_matched horizon rescore",
        "definition": {
            "Short": "n_ka<4",
            "Mid": "4<=n_ka<=6",
            "Longer": "n_ka>6",
            "source": "same as paper Table I / SCOREBOARD_n781_horizon_SMT",
            "n_ka_source": "len(result.key_actions) on type_neg_s1_formal",
        },
        "universe": str(UNIVERSE),
        "n_universe": len(uni_nka),
        "bucket_sizes": uni_h,
        "per_run": per_run,
        "three_seed_mean": means,
        "delta_type_neg_minus_neg_uniform": delta_summary,
    }
    OUT.write_text(json.dumps(report, indent=2))

    print("=== PER-SEED ===")
    hdr = f"{'run':<28} {'SR':>7} {'Short':>7} {'Mid':>7} {'Longer':>7}"
    print(hdr)
    for arm, names in ARMS.items():
        for name in names:
            r = per_run[name]
            print(
                f"{name:<28} {r['sr_pct']:7.2f} "
                f"{r['by_horizon']['Short']['sr_pct']:7.2f} "
                f"{r['by_horizon']['Mid']['sr_pct']:7.2f} "
                f"{r['by_horizon']['Longer']['sr_pct']:7.2f}"
            )
    print("\n=== 3-SEED MEAN ===")
    print(f"{'Arm':<14} {'SR':>7} {'Short':>7} {'Mid':>7} {'Longer':>7}")
    for arm in ("type_neg", "neg_uniform"):
        m = means[arm]
        print(
            f"{arm:<14} {m['overall']['mean']:7.2f} "
            f"{m['by_horizon']['Short']['mean']:7.2f} "
            f"{m['by_horizon']['Mid']['mean']:7.2f} "
            f"{m['by_horizon']['Longer']['mean']:7.2f}"
        )
    d = delta_summary
    print(
        f"{'delta(t-u)':<14} {d['overall']['mean']:+7.2f} "
        f"{d['by_horizon']['Short']['mean']:+7.2f} "
        f"{d['by_horizon']['Mid']['mean']:+7.2f} "
        f"{d['by_horizon']['Longer']['mean']:+7.2f}"
    )
    print("\nper-seed delta overall pp:", deltas["overall"])
    print("per-seed delta Short:", deltas["by_horizon"]["Short"])
    print("per-seed delta Mid:", deltas["by_horizon"]["Mid"])
    print("per-seed delta Longer:", deltas["by_horizon"]["Longer"])
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
