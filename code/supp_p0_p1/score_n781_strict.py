#!/usr/bin/env python3
"""Strict n781 scorer: metrics.success + task-ID set equality with freeze manifest.

Fails hard on missing / duplicate / extra identities. Never reads top-level
success/task_success/is_success (those fields are absent on Omni result.json
and previously produced false 0% SR for full_r2 / qwen3_base_r1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _as_id(x: Any) -> str:
    return str(x)


def load_manifest_ids(manifest_path: Path) -> list[str]:
    blob = json.loads(manifest_path.read_text())
    ids = blob.get("task_ids_sorted")
    if not isinstance(ids, list) or len(ids) != 781:
        raise SystemExit(f"manifest must have task_ids_sorted len=781, got {type(ids)} {getattr(ids,'__len__',lambda:None)()}")
    return [_as_id(x) for x in ids]


def collect_results(results_dir: Path) -> dict[str, Path]:
    """Map identity → result.json path; error on duplicates."""
    found: dict[str, Path] = {}
    dups: list[str] = []
    for p in sorted(results_dir.rglob("result.json")):
        try:
            j = json.loads(p.read_text())
        except Exception as e:
            raise SystemExit(f"unreadable {p}: {e}") from e
        if "identity" not in j:
            raise SystemExit(f"missing identity in {p}")
        tid = _as_id(j["identity"])
        if tid in found:
            dups.append(tid)
        found[tid] = p
    if dups:
        raise SystemExit(f"duplicate identities ({len(dups)}): {dups[:10]}")
    return found


def read_success(result_path: Path) -> int:
    j = json.loads(result_path.read_text())
    metrics = j.get("metrics")
    if not isinstance(metrics, dict) or "success" not in metrics:
        raise SystemExit(f"missing metrics.success in {result_path}")
    s = metrics["success"]
    if s is True or s == 1 or s == "1" or s == "true":
        return 1
    if s is False or s == 0 or s == "0" or s == "false":
        return 0
    raise SystemExit(f"non-binary metrics.success={s!r} in {result_path}")


def score(
    results_dir: Path,
    manifest_path: Path,
    *,
    name: str,
    model: str | None = None,
) -> dict[str, Any]:
    expected = load_manifest_ids(manifest_path)
    expected_set = set(expected)
    if len(expected_set) != 781:
        raise SystemExit("manifest has duplicate task ids")

    found = collect_results(results_dir)
    found_ids = set(found)
    missing = sorted(expected_set - found_ids)
    extra = sorted(found_ids - expected_set)
    if missing or extra:
        raise SystemExit(
            f"identity set mismatch: missing={len(missing)} extra={len(extra)} "
            f"missing_sample={missing[:5]} extra_sample={extra[:5]}"
        )

    by_task: dict[str, int] = {}
    n_succ = 0
    for tid in expected:
        y = read_success(found[tid])
        by_task[tid] = y
        n_succ += y

    meta = {
        "name": name,
        "universe": "n781_SMT_C78",
        "n_completed": 781,
        "n_success": n_succ,
        "sr_pct": round(100.0 * n_succ / 781, 4),
        "policy": "strict_manifest_metrics_success",
        "success_field": "metrics.success",
        "manifest": str(manifest_path),
        "results_dir": str(results_dir),
        "model": model,
        "n_missing": 0,
        "n_extra": 0,
        "n_duplicate": 0,
    }
    return meta, by_task


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--out-json", type=Path, default=None, help="N781_SR.json path")
    ap.add_argument("--out-task-csv", type=Path, default=None, help="task_id,success CSV")
    args = ap.parse_args()

    meta, by_task = score(args.results_dir, args.manifest, name=args.name, model=args.model)
    out = args.out_json or (args.results_dir / "N781_SR.json")
    out.write_text(json.dumps(meta, indent=2) + "\n")
    if args.out_task_csv:
        # preserve manifest order for pairing
        lines = ["task_id,success\n"] + [
            f"{tid},{by_task[tid]}\n" for tid in load_manifest_ids(args.manifest)
        ]
        args.out_task_csv.write_text("".join(lines))
    print(json.dumps(meta, indent=2))
    if meta["n_completed"] != 781:
        sys.exit(2)


if __name__ == "__main__":
    main()
