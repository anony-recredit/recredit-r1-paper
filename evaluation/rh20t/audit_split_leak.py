#!/usr/bin/env python3
"""Confirm the 100 RH20T episode IDs do not appear in train-domain manifests.

This is a local/remote identity intersection check. It never uses RH20T as train data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]
GT = PAPER / "data/rh20t_recredit_cfg3_pc_n100/all.json"
RH_RE = re.compile(r"RH20T_cfg3_task_\d+_user_\d+_scene_\d+_cfg_\d+")


def main() -> None:
    pack = json.loads(GT.read_text())
    ids = {ex["id"] for ex in pack["examples"]}
    assert len(ids) == 100, len(ids)
    hits = []
    # Local paper artifacts that must not treat RH20T as train.
    search_roots = [
        PAPER / "data",
        PAPER.parent / "recredit_r1_wt_v5_paper",
    ]
    skip_parts = {"rh20t_recredit_cfg3_pc_n100", "rh20t_cfg3_pc_n100", "rh20t_metrics", "rh20t_last_frame"}
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in skip_parts for part in p.parts):
                continue
            if p.suffix.lower() not in {".json", ".jsonl", ".txt", ".csv", ".md"}:
                continue
            try:
                text = p.read_text(errors="ignore")
            except Exception:
                continue
            found = set(RH_RE.findall(text)) & ids
            if found:
                hits.append({"path": str(p), "n": len(found), "sample": sorted(found)[:5]})
    out = {
        "n_rh20t_test_ids": len(ids),
        "local_non_rh20t_hits": hits,
        "note": (
            "RH20T 100 episodes are an external offline test set. "
            "Any hit outside rh20t_* trees must be inspected; hits inside result/figure copies are expected."
        ),
    }
    dest = PAPER / "results/rh20t_metrics/LEAK_AUDIT_local.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"n_ids": len(ids), "n_hit_files": len(hits), "out": str(dest)}, indent=2))


if __name__ == "__main__":
    main()
