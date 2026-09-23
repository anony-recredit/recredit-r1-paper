#!/usr/bin/env python3
"""Preflight: Stage-II Full CE must be positive-only + SHA-frozen.

Fails if any row has R_L2!=1 or perc/reas < 0, or file SHA ≠ freeze manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_split(path: Path, expected_sha: str) -> dict:
    got = sha256_file(path)
    if got != expected_sha:
        raise SystemExit(f"SHA mismatch {path}: got {got} expected {expected_sha}")
    df = pd.read_parquet(path, columns=["R_L2", "perc_weight", "reas_weight"])
    n = len(df)
    n_bad_r = int((df["R_L2"].astype(int) != 1).sum())
    n_neg_p = int((df["perc_weight"].astype(float) < 0).sum())
    n_neg_r = int((df["reas_weight"].astype(float) < 0).sum())
    if n_bad_r or n_neg_p or n_neg_r:
        raise SystemExit(
            f"positive-only gate FAIL {path}: n={n} R_L2!=1:{n_bad_r} "
            f"perc<0:{n_neg_p} reas<0:{n_neg_r}"
        )
    return {
        "path": str(path),
        "n": n,
        "sha256": got,
        "R_L2_all_1": True,
        "perc_weight_min": float(df["perc_weight"].min()),
        "reas_weight_min": float(df["reas_weight"].min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--freeze", type=Path, required=True)
    args = ap.parse_args()
    freeze = json.loads(args.freeze.read_text())
    report = {"ok": True, "splits": {}}
    for name, sha in freeze["files"].items():
        report["splits"][name] = check_split(args.data_root / name, sha)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
