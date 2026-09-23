#!/usr/bin/env python3
"""Audit strict P1 arms and emit the immutable pair manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from p1_pairs import (
    CONTEXT_COLUMNS,
    FORBIDDEN_ATTR_COLUMNS,
    action_only_response,
    load_forbidden_ids,
    normalize_action,
    sha256_file,
)


def scalar(value: Any) -> Any:
    return None if pd.isna(value) else value


def assert_arm(frame: pd.DataFrame, label: str) -> None:
    if frame["pair_id"].duplicated().any():
        raise SystemExit(f"{label}: duplicate pair_id")
    forbidden = (FORBIDDEN_ATTR_COLUMNS - {"pair_kind", "pair_weight"}) & set(frame.columns)
    if forbidden:
        raise SystemExit(f"{label}: forbidden attribution columns present: {sorted(forbidden)}")
    weights = frame["pair_weight"].astype(float)
    if not (weights == 1.0).all() or weights.nunique() != 1:
        raise SystemExit(f"{label}: pair_weight must be exactly 1.0")
    for row in frame.to_dict("records"):
        chosen = normalize_action(row["chosen_a"])
        rejected = normalize_action(row["rejected_a"])
        if not chosen or not rejected or chosen == rejected:
            raise SystemExit(f"{label}: invalid actions for {row['pair_id']}")
        if row["chosen_response"] != action_only_response(chosen):
            raise SystemExit(f"{label}: chosen response mismatch for {row['pair_id']}")
        if row["rejected_response"] != action_only_response(rejected):
            raise SystemExit(f"{label}: rejected response mismatch for {row['pair_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attr", type=Path, required=True)
    parser.add_argument("--type-agnostic", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--forbidden-eval", type=Path, action="append", default=[])
    parser.add_argument("--expected-n", type=int, default=4628)
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    attr = pd.read_parquet(args.attr)
    type_agnostic = pd.read_parquet(args.type_agnostic)
    candidates = pd.read_parquet(args.candidate_pool)
    assert_arm(attr, "attr")
    assert_arm(type_agnostic, "type_agnostic")

    if len(attr) != args.expected_n or len(type_agnostic) != args.expected_n:
        raise SystemExit(f"expected {args.expected_n} rows per arm, got {len(attr)} and {len(type_agnostic)}")
    attr = attr.set_index("pair_id", drop=False).sort_index()
    type_agnostic = type_agnostic.set_index("pair_id", drop=False).sort_index()
    candidates = candidates.set_index("pair_id", drop=False).sort_index()
    if list(attr.index) != list(type_agnostic.index) or list(attr.index) != list(candidates.index):
        raise SystemExit("pair_id sets/order differ between P1 artifacts")

    forbidden_ids = load_forbidden_ids(args.forbidden_eval)
    same_direction = 0
    for pair_id in attr.index:
        a = attr.loc[pair_id]
        b = type_agnostic.loc[pair_id]
        c = candidates.loc[pair_id]
        for column in CONTEXT_COLUMNS:
            if scalar(a[column]) != scalar(b[column]) or scalar(a[column]) != scalar(c[column]):
                raise SystemExit(f"context mismatch pair={pair_id} column={column}")
        unordered_a = sorted((normalize_action(a["chosen_a"]), normalize_action(a["rejected_a"])))
        unordered_b = sorted((normalize_action(b["chosen_a"]), normalize_action(b["rejected_a"])))
        unordered_c = sorted((normalize_action(c["action_a"]), normalize_action(c["action_b"])))
        if unordered_a != unordered_b or unordered_a != unordered_c:
            raise SystemExit(f"unordered action mismatch for {pair_id}")
        same_direction += int(normalize_action(a["chosen_a"]) == normalize_action(b["chosen_a"]))
        identity = scalar(a.get("identity"))
        if identity is not None and str(identity) in forbidden_ids:
            raise SystemExit(f"eval identity leaked into P1: {identity}")
        if args.require_images and not Path(str(a["image"])).is_file():
            raise SystemExit(f"missing image for {pair_id}: {a['image']}")

    manifest = {
        "protocol": "P1 strict pair-direction control v1",
        "n_per_arm": len(attr),
        "same_direction_n": same_direction,
        "reversed_direction_n": len(attr) - same_direction,
        "pair_weight": 1.0,
        "context_and_unordered_pairs_identical": True,
        "forbidden_eval_id_count": len(forbidden_ids),
        "files": {
            "attr": {"path": str(args.attr), "sha256": sha256_file(args.attr)},
            "type_agnostic": {"path": str(args.type_agnostic), "sha256": sha256_file(args.type_agnostic)},
            "candidate_pool": {"path": str(args.candidate_pool), "sha256": sha256_file(args.candidate_pool)},
            "source_freeze": {"path": str(args.source_freeze.resolve()), "sha256": sha256_file(args.source_freeze)},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
