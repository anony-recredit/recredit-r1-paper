#!/usr/bin/env python3
"""List Q3 Full steps that must be re-inferred. Do not drop them from the denominator."""
from __future__ import annotations

import json
from pathlib import Path

from recover_from_vlm_logs import classify_request_error

PAPER = Path(__file__).resolve().parents[2]
PRED = PAPER / "results/rh20t_cfg3_pc_n100/q3_full_ol"


def main() -> None:
    failed = []
    ok = 0
    for p in sorted(PRED.glob("*/result.json")):
        d = json.loads(p.read_text())
        for s in d.get("steps") or []:
            req = classify_request_error(s.get("error"))
            rec = {
                "episode_id": d["id"],
                "step_id": int(s["t"]),
                "gt": s.get("gt"),
                "is_key": bool(s.get("is_key")),
                "request_error": req,
            }
            if req:
                failed.append(rec)
            else:
                ok += 1
    out = {
        "variant": "q3_full_ol",
        "checkpoint": "${RECREDIT_ROOT}/ckpts/wt_ablation_v5_qwen3/full/best",
        "protocol": "openloop_multiturn_teacher_force_v2",
        "n_steps_total": ok + len(failed),
        "n_ok_cached": ok,
        "n_failed_must_refill": len(failed),
        "do_not_drop_from_denominator": True,
        "failed_steps": failed,
    }
    dest = PAPER / "results/rh20t_metrics/q3_full_ol_FAILED_STEPS.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "failed_steps"}, indent=2))


if __name__ == "__main__":
    main()
