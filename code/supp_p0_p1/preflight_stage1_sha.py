#!/usr/bin/env python3
"""Preflight: Stage-I parquet SHA must match STAGE1_P0.json for the selected arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["geo", "u2"], required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--freeze", type=Path, required=True)
    args = ap.parse_args()
    freeze = json.loads(args.freeze.read_text())
    arm = freeze["arms"][args.arm]
    report = {"ok": True, "arm": args.arm, "files": {}}
    for name, expected in arm["files"].items():
        path = args.data_root / name
        if not path.exists():
            raise SystemExit(f"missing {path}")
        got = sha256_file(path)
        if got != expected:
            raise SystemExit(f"SHA mismatch {path}: got {got} expected {expected}")
        report["files"][name] = {"path": str(path), "sha256": got}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
