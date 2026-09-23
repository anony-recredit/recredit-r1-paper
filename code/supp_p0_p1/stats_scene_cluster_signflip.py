#!/usr/bin/env python3
"""Scene-cluster bootstrap CI + scene-level sign-flip p (STATS_PROTOCOL v4.1).

Also emits pre-registered report fields:
  - per-seed SR_A / SR_B / Δ_s
  - arm mean±std SR across seeds
  - per-seed McNemar discordant counts + exact/mid-p auxiliary p
  - strict CSV identity-set equality vs freeze manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


def load_task_success(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            tid = str(row["task_id"])
            if tid in out:
                raise SystemExit(f"duplicate task_id in {path}: {tid}")
            out[tid] = int(row["success"])
    return out


def load_scene_map(path: Path) -> dict[str, list[str]]:
    blob = json.loads(path.read_text())
    sm = blob["scene_to_tasks"]
    return {str(k): [str(t) for t in v] for k, v in sm.items()}


def assert_id_set_equal(label: str, got: set[str], expected: set[str]) -> None:
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise SystemExit(
            f"{label} identity set mismatch: missing={len(missing)} extra={len(extra)} "
            f"missing_sample={missing[:5]} extra_sample={extra[:5]}"
        )


def sr_pct(y: dict[str, int], task_ids: list[str]) -> float:
    return 100.0 * sum(y[t] for t in task_ids) / len(task_ids)


def delta_pp(ya: dict[str, int], yb: dict[str, int], task_ids: list[str]) -> float:
    return sr_pct(ya, task_ids) - sr_pct(yb, task_ids)


def mcnemar_counts(ya: dict[str, int], yb: dict[str, int], task_ids: list[str]) -> dict[str, Any]:
    n10 = n01 = n11 = n00 = 0
    for t in task_ids:
        a, b = ya[t], yb[t]
        if a == 1 and b == 0:
            n10 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 1:
            n11 += 1
        else:
            n00 += 1
    # exact two-sided McNemar (binomial mid-p on discordants)
    n_disc = n10 + n01
    if n_disc == 0:
        p_exact = 1.0
        p_mid = 1.0
    else:
        # P(|X - n/2| >= |n10 - n/2|) under X~Bin(n,0.5)
        # exact: 2 * sum_{k=0}^{min(n10,n01)} Binom(n,k) / 2^n   (standard exact McNemar)
        from math import comb

        k = min(n10, n01)
        tail = sum(comb(n_disc, i) for i in range(0, k + 1))
        p_exact = min(1.0, 2.0 * tail / (2**n_disc))
        # mid-p: subtract half the point mass at observed min
        p_mid = min(1.0, 2.0 * (tail - 0.5 * comb(n_disc, k)) / (2**n_disc))
    return {
        "n10": n10,
        "n01": n01,
        "n11": n11,
        "n00": n00,
        "n_discordant": n_disc,
        "p_mcnemar_exact": p_exact,
        "p_mcnemar_midp": p_mid,
        "note": "auxiliary only; not in Holm family",
    }


def mean_std(xs: list[float]) -> dict[str, float]:
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return {"mean": m, "std": 0.0}
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return {"mean": m, "std": math.sqrt(var)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-map", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--pair",
        action="append",
        required=True,
        help="seed:pathA:pathB  (task_id,success CSVs); A=Geo B=U2",
    )
    ap.add_argument("--arm-a-name", type=str, default="geo")
    ap.add_argument("--arm-b-name", type=str, default="u2")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--R", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    task_ids = [str(x) for x in manifest["task_ids_sorted"]]
    expected = set(task_ids)
    if len(expected) != 781 or len(task_ids) != 781:
        raise SystemExit("manifest must list 781 unique task ids")

    scenes = load_scene_map(args.scene_map)
    scene_ids = sorted(scenes.keys())
    M = len(scene_ids)

    pairs: dict[int, tuple[dict[str, int], dict[str, int]]] = {}
    per_seed: dict[str, Any] = {}
    for spec in args.pair:
        seed_s, pa, pb = spec.split(":", 2)
        s = int(seed_s)
        ya = load_task_success(Path(pa))
        yb = load_task_success(Path(pb))
        assert_id_set_equal(f"seed{s}:{pa}", set(ya), expected)
        assert_id_set_equal(f"seed{s}:{pb}", set(yb), expected)
        pairs[s] = (ya, yb)
        sr_a = sr_pct(ya, task_ids)
        sr_b = sr_pct(yb, task_ids)
        per_seed[str(s)] = {
            f"SR_{args.arm_a_name}": sr_a,
            f"SR_{args.arm_b_name}": sr_b,
            "delta_pp": sr_a - sr_b,
            "mcnemar": mcnemar_counts(ya, yb, task_ids),
        }

    seeds = sorted(pairs.keys())
    deltas = {s: per_seed[str(s)]["delta_pp"] for s in seeds}
    delta_hat = sum(deltas.values()) / len(seeds)
    s_delta = mean_std(list(deltas.values()))["std"]

    sr_a_list = [per_seed[str(s)][f"SR_{args.arm_a_name}"] for s in seeds]
    sr_b_list = [per_seed[str(s)][f"SR_{args.arm_b_name}"] for s in seeds]

    rng = random.Random(args.seed)

    boot: list[float] = []
    for _ in range(args.B):
        drawn = [scene_ids[rng.randrange(M)] for _ in range(M)]
        tasks_star: list[str] = []
        for sc in drawn:
            tasks_star.extend(scenes[sc])
        dstar = [delta_pp(pairs[s][0], pairs[s][1], tasks_star) for s in seeds]
        boot.append(sum(dstar) / len(dstar))
    boot.sort()
    lo = boot[int(0.025 * args.B)]
    hi = boot[int(0.975 * args.B)]

    def scene_delta(c: str) -> float:
        tc = scenes[c]
        return sum(delta_pp(pairs[s][0], pairs[s][1], tc) for s in seeds) / len(seeds)

    delta_c = {c: scene_delta(c) for c in scene_ids}
    weights = {c: float(len(scenes[c])) for c in scene_ids}
    wsum = sum(weights.values())

    def signed_delta(signs: dict[str, int]) -> float:
        return sum(weights[c] * signs[c] * delta_c[c] for c in scene_ids) / wsum

    enumerate_all = (1 << M) <= 1_000_000 and M <= 20
    n_ext = 0
    if enumerate_all:
        R = 1 << M
        for mask in range(R):
            signs = {scene_ids[i]: (1 if (mask >> i) & 1 else -1) for i in range(M)}
            if abs(signed_delta(signs)) >= abs(delta_hat):
                n_ext += 1
        p = n_ext / R
        p_mode = f"enumerate_2^{M}"
    else:
        R = args.R
        for _ in range(R):
            signs = {c: rng.choice((-1, 1)) for c in scene_ids}
            if abs(signed_delta(signs)) >= abs(delta_hat):
                n_ext += 1
        p = (1 + n_ext) / (R + 1)
        p_mode = f"monte_carlo_R={R}"

    report: dict[str, Any] = {
        "comparison": f"{args.arm_a_name}_vs_{args.arm_b_name}",
        "seeds": seeds,
        "per_seed": per_seed,
        f"SR_{args.arm_a_name}_mean_std": mean_std(sr_a_list),
        f"SR_{args.arm_b_name}_mean_std": mean_std(sr_b_list),
        "delta_s_pp": deltas,
        "delta_hat_pp": delta_hat,
        "s_delta": s_delta,
        "ci95_scene_cluster": [lo, hi],
        "B": args.B,
        "p_two_sided": p,
        "p_mode": p_mode,
        "n_extreme": n_ext,
        "M_scenes": M,
        "identity_set_checks": "all CSV task_id sets == manifest (781)",
        "assumptions": (
            "sign-flip: scene paired diffs exchangeable/approx symmetric under H0; "
            "estimand conditional on fixed seeds; McNemar auxiliary only"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
