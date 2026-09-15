#!/usr/bin/env python3
"""Build the Gate D audit sample after refill. Does not claim two human annotators."""
from __future__ import annotations

import os
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ["RECREDIT_ROOT"]) / "results" / "rh20t_metrics"
OUT = ROOT / "gate_d"
SEED = 20260913


def load_rows(name: str) -> list[dict]:
    p = ROOT / name / "per_step_normalized.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    pool = []
    for name in ("q2_stage1_ol", "q2_full_ol", "q2_dpo_ol", "q3_stage1_ol", "q3_full_ol", "q3_dpo_ol"):
        for r in load_rows(name):
            r = dict(r)
            r["variant"] = name
            pool.append(r)
    by = defaultdict(list)
    for r in pool:
        if r.get("request_error"):
            by["request_error"].append(r)
        elif r.get("format_valid") is False:
            by[f"fmt_{r.get('format_error') or 'invalid'}"].append(r)
        else:
            by[f"verb_{r.get('gt_verb')}"].append(r)
    sample = []
    for key, items in sorted(by.items()):
        take = min(8, len(items))
        if take:
            sample.extend(rng.sample(items, take))
    # unique, cap 100
    seen = set()
    out = []
    for r in sample:
        k = (r["variant"], r["episode_id"], r["step_id"])
        if k in seen:
            continue
        seen.add(k)
        out.append({
            "variant": r["variant"],
            "episode_id": r["episode_id"],
            "step_id": r["step_id"],
            "instruction": r.get("instruction"),
            "raw_gt": r.get("raw_gt"),
            "gt_verb": r.get("gt_verb"),
            "gt_object": r.get("gt_object"),
            "pred_verb": r.get("pred_verb"),
            "pred_object": r.get("pred_object"),
            "raw_prediction": r.get("raw_prediction"),
            "full_response_tail": (r.get("full_response") or "")[-400:],
            "request_error": r.get("request_error"),
            "format_error": r.get("format_error"),
            "auto_verb_correct": r.get("verb_correct"),
            "auto_object_correct": r.get("object_correct"),
            "auto_joint_correct": r.get("joint_correct"),
            "annotator_a_auto": True,
            "annotator_b_verb_ok": None,
            "annotator_b_object_ok": None,
            "annotator_b_format_ok": None,
            "annotator_b_auto_match_ok": None,
        })
        if len(out) >= 100:
            break
    dest = OUT / "audit_sample_100.jsonl"
    dest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    meta = {
        "n": len(out),
        "seed": SEED,
        "status": "awaiting_second_annotator",
        "note": "Annotator A is the frozen automatic parser. Gate D still needs an independent human pass B. This file is the worksheet, not completed dual annotation.",
        "strata": {k: min(8, len(v)) for k, v in sorted(by.items())},
    }
    (OUT / "STATUS.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
