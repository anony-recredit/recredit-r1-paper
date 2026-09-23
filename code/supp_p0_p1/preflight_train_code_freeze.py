#!/usr/bin/env python3
"""Assert TRAIN_CODE_FREEZE.json SHAs still match on disk."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", type=Path, required=True)
    args = ap.parse_args()
    freeze = json.loads(args.freeze.read_text())
    report = {"ok": True, "files": {}}
    for key, meta in freeze["files"].items():
        path = Path(meta["path"])
        got = sha256_file(path)
        if got != meta["sha256"]:
            raise SystemExit(f"code freeze mismatch {key}: got {got} expected {meta['sha256']}")
        report["files"][key] = {"path": str(path), "sha256": got}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
