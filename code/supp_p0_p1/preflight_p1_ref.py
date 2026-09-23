#!/usr/bin/env python3
"""Validate frozen pairs, code, and Full CE start before ref-logp work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from p1_pairs import FORBIDDEN_ATTR_COLUMNS, sha256_file


def verify(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"SHA mismatch {path}: expected={expected} actual={actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("attr", "type_agnostic"), required=True)
    parser.add_argument("--seed", choices=(1, 17, 31), type=int, required=True)
    parser.add_argument("--training-freeze", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.training_freeze.read_text())
    for key in ("trainer", "dataset", "precompute_ref"):
        verify(Path(freeze[key]["path"]), freeze[key]["sha256"])
    checkpoint = freeze["full_checkpoints"][str(args.seed)]
    expected_full = Path(checkpoint["path"]).resolve()
    if args.full_checkpoint.resolve() != expected_full:
        raise SystemExit(f"wrong Full CE start: {args.full_checkpoint} != {expected_full}")
    verify(Path(checkpoint["best_json"]), checkpoint["best_json_sha256"])
    if not (expected_full / "config.json").is_file() or not list(expected_full.glob("*.safetensors")):
        raise SystemExit(f"incomplete Full CE checkpoint: {expected_full}")

    manifest = json.loads(args.pair_manifest.read_text())
    key = "attr" if args.arm == "attr" else "type_agnostic"
    pair = Path(manifest["files"][key]["path"])
    verify(pair, manifest["files"][key]["sha256"])
    frame = pd.read_parquet(pair)
    if len(frame) != manifest["n_per_arm"] or frame["pair_id"].duplicated().any():
        raise SystemExit("pair row count or IDs invalid")
    forbidden = (FORBIDDEN_ATTR_COLUMNS - {"pair_kind", "pair_weight"}) & set(frame.columns)
    if forbidden:
        raise SystemExit(f"forbidden attribution fields: {sorted(forbidden)}")
    if not (frame["pair_weight"].astype(float) == 1.0).all():
        raise SystemExit("pair_weight is not identically 1.0")
    print(json.dumps({"ok": True, "arm": args.arm, "seed": args.seed, "n": len(frame), "full": str(expected_full)}, indent=2))


if __name__ == "__main__":
    main()
