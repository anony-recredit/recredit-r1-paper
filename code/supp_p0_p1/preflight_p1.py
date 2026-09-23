#!/usr/bin/env python3
"""Fail-closed preflight for one strict P1 DPO run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from p1_pairs import FORBIDDEN_ATTR_COLUMNS, assert_finite_ref_columns, sha256_file


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
    parser.add_argument("--ref-parquet", type=Path, required=True)
    parser.add_argument("--ref-meta", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=4628)
    args = parser.parse_args()

    freeze = json.loads(args.training_freeze.read_text())
    for key in ("trainer", "dataset", "precompute_ref"):
        entry = freeze[key]
        verify(Path(entry["path"]), entry["sha256"])
    checkpoint = freeze["full_checkpoints"][str(args.seed)]
    expected_full = Path(checkpoint["path"]).resolve()
    if args.full_checkpoint.resolve() != expected_full:
        raise SystemExit(f"wrong Full CE start for seed {args.seed}: {args.full_checkpoint} != {expected_full}")
    verify(Path(checkpoint["best_json"]), checkpoint["best_json_sha256"])
    if not (expected_full / "config.json").is_file():
        raise SystemExit(f"incomplete Full CE checkpoint: {expected_full}")
    if not list(expected_full.glob("*.safetensors")):
        raise SystemExit(f"Full CE checkpoint has no safetensors: {expected_full}")

    pair_manifest = json.loads(args.pair_manifest.read_text())
    arm_key = "attr" if args.arm == "attr" else "type_agnostic"
    pair_entry = pair_manifest["files"][arm_key]
    verify(Path(pair_entry["path"]), pair_entry["sha256"])
    if pair_manifest["n_per_arm"] != args.expected_n or pair_manifest["pair_weight"] != 1.0:
        raise SystemExit("pair manifest violates strict P1 count/weight")

    meta = json.loads(args.ref_meta.read_text())
    if Path(meta["ref_ckpt"]).resolve() != expected_full:
        raise SystemExit(f"ref-logp checkpoint mismatch: {meta['ref_ckpt']} != {expected_full}")
    if meta["source_pairs_sha256"] != pair_entry["sha256"]:
        raise SystemExit("ref-logp source pair SHA mismatch")
    verify(args.ref_parquet, meta["output_sha256"])

    frame = pd.read_parquet(args.ref_parquet)
    if len(frame) != args.expected_n or frame["pair_id"].duplicated().any():
        raise SystemExit(f"bad ref parquet rows/ids: n={len(frame)}")
    forbidden = (FORBIDDEN_ATTR_COLUMNS - {"pair_kind", "pair_weight"}) & set(frame.columns)
    if forbidden:
        raise SystemExit(f"forbidden attribution fields in ref parquet: {sorted(forbidden)}")
    if not (frame["pair_weight"].astype(float) == 1.0).all():
        raise SystemExit("ref parquet pair_weight is not identically 1.0")
    assert_finite_ref_columns(frame)
    print(
        json.dumps(
            {
                "ok": True,
                "arm": args.arm,
                "seed": args.seed,
                "n": len(frame),
                "full_checkpoint": str(expected_full),
                "ref_parquet_sha256": meta["output_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
