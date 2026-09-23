#!/usr/bin/env python3
"""Build strict unweighted Attr and type-agnostic DPO pairs for P1.

The two arms share identical contexts and unordered action pairs. The Attr arm
retains the frozen v27 preference direction. The type-agnostic arm receives a
sanitized candidate table and orients each action pair only from outcome and
action-validity statistics computed over frozen training-domain sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from p1_pairs import (
    CONTEXT_COLUMNS,
    FORBIDDEN_ATTR_COLUMNS,
    OutcomeValidityStats,
    action_only_response,
    horizon_bin,
    identity_of,
    load_examples,
    load_forbidden_ids,
    normalize_action,
    sha256_file,
)


def verify_frozen_file(path: Path, expected_sha: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise SystemExit(f"SHA mismatch: {path}: expected={expected_sha} actual={actual}")


def load_source_freeze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    for entry in payload["files"].values():
        verify_frozen_file(Path(entry["path"]), entry["sha256"])
    return payload


def build_candidates(attr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [column for column in CONTEXT_COLUMNS + ["chosen_a", "rejected_a"] if column not in attr]
    if missing:
        raise SystemExit(f"attribution parquet missing columns: {missing}")
    if attr["pair_id"].duplicated().any():
        raise SystemExit("duplicate pair_id in attribution parquet")

    candidates: list[dict[str, Any]] = []
    orientations: list[dict[str, str]] = []
    for row in attr.to_dict("records"):
        chosen = normalize_action(row["chosen_a"])
        rejected = normalize_action(row["rejected_a"])
        if not chosen or not rejected or chosen == rejected:
            raise SystemExit(f"invalid action pair: {row['pair_id']}")
        action_a, action_b = sorted((chosen, rejected))
        candidate = {column: row.get(column) for column in CONTEXT_COLUMNS}
        candidate.update(
            {
                "candidate_schema": "recredit_r1.p1_candidate.v1",
                "action_a": action_a,
                "action_b": action_b,
                "horizon_bin": horizon_bin(int(row["t"])),
            }
        )
        candidates.append(candidate)
        orientations.append(
            {
                "pair_id": str(row["pair_id"]),
                "attr_chosen": chosen,
                "attr_rejected": rejected,
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    leaked = FORBIDDEN_ATTR_COLUMNS & set(candidate_frame.columns)
    if leaked:
        raise SystemExit(f"sanitized candidate table leaked attribution columns: {sorted(leaked)}")
    return candidate_frame, pd.DataFrame(orientations)


def training_row(candidate: dict[str, Any], chosen: str, rejected: str, arm: str) -> dict[str, Any]:
    row = {column: candidate.get(column) for column in CONTEXT_COLUMNS}
    row.update(
        {
            "pair_kind": arm,
            "schema_version": "recredit_r1.p1_strict.v1",
            "chosen_a": chosen,
            "rejected_a": rejected,
            "chosen_response": action_only_response(chosen),
            "rejected_response": action_only_response(rejected),
            "pair_weight": 1.0,
            "horizon_bin": candidate["horizon_bin"],
        }
    )
    return row


def orient_type_agnostic(
    candidates: pd.DataFrame,
    stats: OutcomeValidityStats,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    leaked = FORBIDDEN_ATTR_COLUMNS & set(candidates.columns)
    if leaked:
        raise SystemExit(f"type-agnostic miner received forbidden columns: {sorted(leaked)}")

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        tasktype = str(candidate.get("tasktype") or "")
        bucket = str(candidate["horizon_bin"])
        score_a = stats.score(tasktype, bucket, candidate["action_a"])
        score_b = stats.score(tasktype, bucket, candidate["action_b"])
        record = {
            "pair_id": str(candidate["pair_id"]),
            "action_a": candidate["action_a"],
            "action_b": candidate["action_b"],
            "score_a": None if score_a is None else list(score_a),
            "score_b": None if score_b is None else list(score_b),
        }
        if score_a is None or score_b is None or score_a == score_b:
            record["reason"] = "missing_support" if score_a is None or score_b is None else "exact_tie"
            dropped.append(record)
            continue
        chosen, rejected = (
            (candidate["action_a"], candidate["action_b"])
            if score_a > score_b
            else (candidate["action_b"], candidate["action_a"])
        )
        rows.append(training_row(candidate, chosen, rejected, "p1_type_agnostic"))
        record.update({"type_chosen": chosen, "type_rejected": rejected})
        audit.append(record)
    return pd.DataFrame(rows), pd.DataFrame(audit), pd.DataFrame(dropped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--expected-attr-n", type=int, default=4705)
    parser.add_argument("--expected-resolved-n", type=int, default=4628)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    freeze = load_source_freeze(args.source_freeze)
    attr_path = Path(freeze["files"][freeze["roles"]["attr_pairs"]]["path"])
    outcome_paths = [Path(freeze["files"][name]["path"]) for name in freeze["roles"]["outcome_sources"]]
    forbidden_paths = [Path(freeze["files"][name]["path"]) for name in freeze["roles"]["forbidden_eval_sources"]]

    outputs = {
        "candidates": args.out_dir / "candidates_sanitized.parquet",
        "attr": args.out_dir / "attr_unweighted.parquet",
        "type": args.out_dir / "type_agnostic_unweighted.parquet",
        "orientation": args.out_dir / "orientation_audit.parquet",
        "dropped": args.out_dir / "dropped_unresolved.parquet",
        "meta": args.out_dir / "BUILD_META.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"refusing to overwrite existing P1 outputs: {existing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    attr_source = pd.read_parquet(attr_path)
    if len(attr_source) != args.expected_attr_n:
        raise SystemExit(f"expected {args.expected_attr_n} attr pairs, got {len(attr_source)}")
    candidates, orientations = build_candidates(attr_source)

    forbidden_ids = load_forbidden_ids(forbidden_paths)
    stats = OutcomeValidityStats(min_support=args.min_support)
    source_counts: dict[str, int] = {}
    for path in outcome_paths:
        examples = load_examples(path)
        source_counts[str(path)] = len(examples)
        for example in examples:
            identity = identity_of(example)
            if identity is not None and identity in forbidden_ids:
                continue
            stats.add_example(example)

    type_frame, score_audit, dropped = orient_type_agnostic(candidates, stats)
    if len(type_frame) != args.expected_resolved_n:
        raise SystemExit(
            f"expected {args.expected_resolved_n} resolved pairs, got {len(type_frame)}; "
            f"dropped={len(dropped)}"
        )

    resolved = set(type_frame["pair_id"].astype(str))
    orientation_map = orientations.set_index("pair_id").to_dict("index")
    candidate_map = candidates.set_index("pair_id").to_dict("index")
    attr_rows: list[dict[str, Any]] = []
    joined_audit: list[dict[str, Any]] = []
    for type_record in type_frame.to_dict("records"):
        pair_id = str(type_record["pair_id"])
        orientation = orientation_map[pair_id]
        attr_rows.append(
            training_row(
                {"pair_id": pair_id, **candidate_map[pair_id]},
                orientation["attr_chosen"],
                orientation["attr_rejected"],
                "p1_attr_unweighted",
            )
        )
    attr_frame = pd.DataFrame(attr_rows)

    score_map = score_audit.set_index("pair_id").to_dict("index")
    same_direction = 0
    for pair_id in sorted(resolved):
        orientation = orientation_map[pair_id]
        score_record = score_map[pair_id]
        same = orientation["attr_chosen"] == score_record["type_chosen"]
        same_direction += int(same)
        joined_audit.append({**orientation, **score_record, "same_direction": same})

    candidates[candidates["pair_id"].astype(str).isin(resolved)].to_parquet(outputs["candidates"], index=False)
    attr_frame.to_parquet(outputs["attr"], index=False)
    type_frame.to_parquet(outputs["type"], index=False)
    pd.DataFrame(joined_audit).to_parquet(outputs["orientation"], index=False)
    dropped.to_parquet(outputs["dropped"], index=False)

    meta = {
        "protocol": "P1 strict pair-direction control v1",
        "attr_source_n": len(attr_source),
        "resolved_n": len(type_frame),
        "dropped_n": len(dropped),
        "same_direction_n": same_direction,
        "reversed_direction_n": len(type_frame) - same_direction,
        "pair_weight": 1.0,
        "source_counts": source_counts,
        "stats": stats.summary(),
        "source_freeze_sha256": sha256_file(args.source_freeze),
    }
    outputs["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
