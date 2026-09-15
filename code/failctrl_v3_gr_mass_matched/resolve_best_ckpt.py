#!/usr/bin/env python3
"""Pick the V_pos-selected checkpoint: min val_loss, tie -> earlier epoch."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def resolve(root: Path) -> Path:
    cands = []
    for p in root.rglob("val_metrics.json"):
        d = json.loads(p.read_text())
        epoch = d.get("epoch")
        if epoch in (None, "final"):
            continue
        if "val_loss" not in d:
            continue
        ck = p.parent
        if not (ck / "config.json").exists() and not (ck / "fsdp_full.pt").exists():
            continue
        cands.append((int(epoch), float(d["val_loss"]), ck))
    if cands:
        cands.sort(key=lambda x: (x[1], x[0]))
        return cands[0][2]
    for name in ("best", "epoch_1", "epoch_0", "final"):
        ck = root / name
        if (ck / "config.json").exists():
            return ck
    raise SystemExit(f"no selectable checkpoint under {root}")


if __name__ == "__main__":
    print(resolve(Path(sys.argv[1])))
