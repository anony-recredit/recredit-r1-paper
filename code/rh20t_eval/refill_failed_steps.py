#!/usr/bin/env python3
"""Refill request-failed RH20T steps. Same offline eval protocol eval_rh20t_offline_action.py.

Does not drop steps from the 1131 denominator. Successful cached steps are kept.
Saves the full model reply in `full_response` (not only 240-char response_head).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_rh20t_offline_action import (  # noqa: E402
    SYSTEM,
    TASK_PREFIX,
    USER_IMAGE_PREFIX,
    call_chat,
    gt_verb,
    img_content,
    parse_pred,
)

sys.path.insert(0, str(HERE.parents[1] / "evaluation" / "rh20t"))
from recover_from_vlm_logs import classify_request_error  # noqa: E402


def refill_example(ex: dict, pack_root: Path, port: int, out_dir: Path, failed_ts: set[int]) -> dict:
    eid = str(ex["id"])
    dest = out_dir / eid
    dest.mkdir(parents=True, exist_ok=True)
    rj = dest / "result.json"
    old = json.loads(rj.read_text()) if rj.exists() else {}
    old_steps = {int(s["t"]): s for s in (old.get("steps") or [])}

    instruction = str(ex.get("instruction") or "")
    steps = ex.get("steps") or []
    messages = [{"role": "system", "content": SYSTEM}]
    step_rows = []
    n_ok = n_key = n_key_ok = 0
    prev_gt = "init"
    n_refilled = 0

    for i, step in enumerate(steps):
        verb = gt_verb(step)
        t = int(step.get("t"))
        img_rel = step.get("image") or ""
        img_path = pack_root / img_rel
        cached = old_steps.get(t, {})
        req = classify_request_error(cached.get("error"))
        need = (t in failed_ts) or bool(req)

        if i == 0:
            user_text = TASK_PREFIX.format(task_name=instruction)
        else:
            user_text = USER_IMAGE_PREFIX.format(action=prev_gt)
        if img_path.exists():
            messages.append({"role": "user", "content": img_content(img_path, user_text)})

        err = cached.get("error")
        pred = cached.get("pred") or ""
        resp = cached.get("full_response") or ""
        head = cached.get("response_head") or ""

        if need and img_path.exists():
            err = None
            pred = ""
            resp = ""
            try:
                resp = call_chat(port, messages)
                pred = parse_pred(resp)
                n_refilled += 1
            except Exception as e:
                err = str(e)
            head = (resp or "")[:240]

        ok = bool(pred) and pred == verb
        if ok:
            n_ok += 1
        is_key = verb in ("grasp", "place")
        if is_key:
            n_key += 1
            if ok:
                n_key_ok += 1

        step_rows.append({
            "t": t,
            "gt": verb,
            "pred": pred,
            "ok": ok,
            "is_key": is_key,
            "error": err,
            "response_head": head,
            "full_response": resp or None,
            "refilled": bool(need),
        })
        messages.append({"role": "assistant", "content": f"<DecisionMaking>{verb}</DecisionMaking>"})
        prev_gt = verb

    n = len(steps)
    success = bool(n > 0 and n_ok == n)
    key_success = bool(n_key > 0 and n_key_ok == n_key)
    result = {
        "id": eid,
        "identity": eid,
        "instruction": instruction,
        "R_L2": ex.get("R_L2"),
        "n_steps": n,
        "protocol": "openloop_multiturn_teacher_force_v2",
        "step_acc": (n_ok / n) if n else 0.0,
        "n_correct": n_ok,
        "key_action_acc": (n_key_ok / n_key) if n_key else None,
        "n_key": n_key,
        "n_key_ok": n_key_ok,
        "traj_exact": success,
        "key_traj_exact": key_success,
        "success": int(success),
        "metrics": {"success": int(success), "success_key": int(key_success), "step_acc": (n_ok / n) if n else 0.0},
        "n_refilled": n_refilled,
        "steps": step_rows,
    }
    rj.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fail-list", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18001)
    args = ap.parse_args()

    fail = json.loads(args.fail_list.read_text())
    by_ep: dict[str, set[int]] = {}
    for row in fail["failed_steps"]:
        by_ep.setdefault(row["episode_id"], set()).add(int(row["step_id"]))

    pack = json.loads((args.pack / "all.json").read_text())
    examples = {e["id"]: e for e in pack["examples"]}
    t0 = time.time()
    n_ref = 0
    for i, (eid, ts) in enumerate(sorted(by_ep.items()), 1):
        r = refill_example(examples[eid], args.pack, args.port, args.out, ts)
        n_ref += int(r.get("n_refilled") or 0)
        print(f"[{i}/{len(by_ep)}] {eid} refilled={r.get('n_refilled')} remain_err={sum(1 for s in r['steps'] if s.get('error'))}", flush=True)
    print(json.dumps({"episodes": len(by_ep), "steps_refilled": n_ref, "sec": round(time.time() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
