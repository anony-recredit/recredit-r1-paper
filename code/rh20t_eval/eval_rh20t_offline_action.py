#!/usr/bin/env python3
"""RH20T offline open-loop eval aligned to n781 reporting scale.

Cannot run closed-loop AI2-THOR on real RGB. Closest same-scale protocol:
  - Same message shape as embodied_reasoner/evaluate.py (system + TASK_PREFIX +
    multi-turn <image> turns, <DecisionMaking> action).
  - Teacher-force GT actions into history (open-loop).
  - Episode binary metrics.success (all steps match) → sr_pct like n781.

Usage:
  python eval_rh20t_offline_action.py --pack ... --out ... --ports 10001 --shard 0 --n-shards 8
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

DM_RE = re.compile(r"<DecisionMaking>\s*(.*?)\s*</DecisionMaking>", re.I | re.S)
VERB_SET = ("move", "place", "grasp", "end")

# Mirror evaluate/prompt.py EMBODIED_SYSTEM_PROMPT + TASK_PREFIX structure.
SYSTEM = (
    "You are a robot in given room. You need to complete the tasks according to "
    "human instructions. We provide an Available_Actions set and the corresponding "
    "explanations for each action. Each step, you should select one action from "
    "Available_Actions."
)

TASK_PREFIX = """This is an image from your frontal perspective. Please select an action from the Available_Actions and fill in the arguments.
Task: "{task_name}"
Available_Actions: {{
"move": Move / approach with the end-effector.
"grasp": Close the gripper to grasp the object.
"place": Open the gripper / place the object.
"end": If you think you have completed the task, please output "end".}}
Before making each decision, you can think, plan, and even reflect step by step, and then output your final action.
Your final action must strictly follow format: <DecisionMaking>Your Action</DecisionMaking>, for example, <DecisionMaking>move</DecisionMaking>."""

USER_IMAGE_PREFIX = """After executing your previous "{action}", you get this new image above.
To complete your task, you can think step by step at first and then output your new action from the Available_Actions.
Your action must strictly follow format: <DecisionMaking>Your Action</DecisionMaking>, for example, <DecisionMaking>move</DecisionMaking>."""


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def gt_verb(step: dict) -> str:
    a = step.get("action") or {}
    if isinstance(a, str):
        try:
            import ast

            a = ast.literal_eval(a)
        except Exception:
            return str(a).strip().lower()
    raw = str(a.get("raw") or a.get("verb") or "").strip().lower()
    return raw.split()[0] if raw else ""


def parse_pred(text: str) -> str:
    """Extract a verb in VERB_SET from model output (prefer DecisionMaking body)."""
    if not text:
        return ""
    m = DM_RE.search(text)
    body = (m.group(1) if m else text).strip().lower()
    # exact first token
    tok = re.sub(r"[^a-z_\s]", " ", body).strip().split()
    if tok:
        for v in VERB_SET:
            if tok[0] == v:
                return v
    # scan whole DM / text for a legal verb (models sometimes dump prose in DM)
    for v in VERB_SET:
        if re.search(rf"\b{v}\b", body):
            return v
    return tok[0] if tok else ""


def call_chat(port: int, messages: list[dict], timeout: int = 300) -> str:
    url = f"http://127.0.0.1:{port}/chat"
    payload = {"inputs": [{"messages": messages}]}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    out = r.json()["output_text"]
    if isinstance(out, list):
        out = out[0]
    return out if isinstance(out, str) else str(out)


def img_content(path: Path, text: str) -> list[dict]:
    b64 = encode_image(path)
    return [
        {"type": "image", "image": f"data:image/jpeg;base64,{b64}"},
        {"type": "text", "text": text},
    ]


def eval_example(
    ex: dict,
    pack_root: Path,
    port: int,
    out_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    eid = str(ex["id"])
    dest = out_dir / eid
    dest.mkdir(parents=True, exist_ok=True)
    rj = dest / "result.json"
    if rj.exists() and not overwrite:
        try:
            old = json.loads(rj.read_text())
            # require n781-aligned schema
            if "metrics" in old and "success" in old["metrics"] and old.get("protocol", "").startswith("openloop"):
                return old
        except Exception:
            pass

    instruction = str(ex.get("instruction") or "")
    steps = ex.get("steps") or []
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]
    step_rows: list[dict] = []
    n_ok = 0
    n_key = 0
    n_key_ok = 0
    prev_gt = "init"

    for i, step in enumerate(steps):
        verb = gt_verb(step)
        img_rel = step.get("image") or ""
        img_path = pack_root / img_rel
        if not img_path.exists():
            step_rows.append(
                {
                    "t": step.get("t"),
                    "gt": verb,
                    "pred": "",
                    "ok": False,
                    "error": f"missing_image:{img_rel}",
                }
            )
            # still teacher-force so later turns stay on rails
            messages.append(
                {"role": "assistant", "content": f"<DecisionMaking>{verb}</DecisionMaking>"}
            )
            prev_gt = verb
            continue

        if i == 0:
            user_text = TASK_PREFIX.format(task_name=instruction)
        else:
            user_text = USER_IMAGE_PREFIX.format(action=prev_gt)

        messages.append({"role": "user", "content": img_content(img_path, user_text)})

        err = None
        pred = ""
        resp = ""
        try:
            resp = call_chat(port, messages)
            pred = parse_pred(resp)
        except Exception as e:
            err = str(e)

        ok = bool(pred) and pred == verb
        if ok:
            n_ok += 1
        is_key = verb in ("grasp", "place")
        if is_key:
            n_key += 1
            if ok:
                n_key_ok += 1

        step_rows.append(
            {
                "t": step.get("t"),
                "gt": verb,
                "pred": pred,
                "ok": ok,
                "is_key": is_key,
                "error": err,
                "response_head": (resp or "")[:240],
            }
        )

        # Teacher-force GT into dialogue (open-loop), matching BC-style offline eval.
        messages.append(
            {"role": "assistant", "content": f"<DecisionMaking>{verb}</DecisionMaking>"}
        )
        prev_gt = verb

    n = len(steps)
    success = bool(n > 0 and n_ok == n)  # open-loop episode success ≡ n781 binary scale
    key_success = bool(n_key > 0 and n_key_ok == n_key)

    result = {
        "id": eid,
        "identity": eid,  # n781-compatible field name for scoring scripts
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
        "metrics": {
            "success": int(success),
            "success_key": int(key_success),
            "step_acc": (n_ok / n) if n else 0.0,
        },
        "steps": step_rows,
    }
    rj.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def load_examples(pack: Path, split: str) -> list[dict]:
    if split == "all":
        blob = json.loads((pack / "all.json").read_text())
    elif split == "test":
        blob = json.loads((pack / "test.json").read_text())
    elif split == "train":
        blob = json.loads((pack / "train.json").read_text())
    else:
        raise ValueError(split)
    return blob["examples"] if isinstance(blob, dict) else blob


def summarize(out_dir: Path, examples: list[dict], arm: str = "", model: str = "") -> dict:
    """n781-compatible primary fields: n, succ, sr_pct."""
    by_id = {str(e["id"]): e for e in examples}
    rows = []
    for eid in by_id:
        rj = out_dir / eid / "result.json"
        if not rj.exists():
            continue
        try:
            rows.append(json.loads(rj.read_text()))
        except Exception:
            continue
    if not rows:
        return {"name": arm, "n": 0, "succ": 0, "sr_pct": None, "n_missing": len(by_id)}

    succ = sum(1 for r in rows if int((r.get("metrics") or {}).get("success", r.get("success", 0))))
    succ_key = sum(
        1 for r in rows if int((r.get("metrics") or {}).get("success_key", 0))
    )
    step_accs = [float(r.get("step_acc") or (r.get("metrics") or {}).get("step_acc") or 0) for r in rows]
    n = len(rows)
    return {
        "name": arm or out_dir.name,
        "n": n,
        "succ": succ,
        "sr_pct": round(100.0 * succ / n, 2),
        "succ_key": succ_key,
        "sr_key_pct": round(100.0 * succ_key / n, 2),
        "mean_step_acc_pct": round(100.0 * sum(step_accs) / n, 2),
        "n_universe": len(by_id),
        "n_missing": len(by_id) - n,
        "missing_ids": sorted(set(by_id) - {r["id"] for r in rows}),
        "model": model,
        "root": str(out_dir),
        "protocol": "openloop_multiturn_teacher_force_v2",
        "note": (
            "sr_pct = open-loop episode success (all steps match GT). "
            "Same binary episode % scale as n781 metrics.success; not closed-loop env SR."
        ),
        "by_R_L2": {
            str(rval): {
                "n": sum(1 for x in rows if x.get("R_L2") == rval),
                "sr_pct": round(
                    100.0
                    * sum(
                        1
                        for x in rows
                        if x.get("R_L2") == rval
                        and int((x.get("metrics") or {}).get("success", 0))
                    )
                    / max(1, sum(1 for x in rows if x.get("R_L2") == rval)),
                    2,
                ),
            }
            for rval in (0, 1)
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ports", type=str, default="10001")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--split", choices=("all", "train", "test"), default="all")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ids", type=str, default="")
    ap.add_argument("--workers-per-port", type=int, default=1)
    args = ap.parse_args()

    ports = [int(x) for x in args.ports.split(",") if x.strip()]
    examples = load_examples(args.pack, args.split)
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        examples = [e for e in examples if str(e["id"]) in want]
    examples = [e for i, e in enumerate(examples) if i % args.n_shards == args.shard]
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"[shard {args.shard}/{args.n_shards}] n={len(examples)} ports={ports} out={args.out}",
        flush=True,
    )

    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, len(ports) * args.workers_per_port)) as pool:
        for i, ex in enumerate(examples):
            port = ports[i % len(ports)]
            jobs.append(
                pool.submit(eval_example, ex, args.pack, port, args.out, args.overwrite)
            )
        done = 0
        for fut in as_completed(jobs):
            done += 1
            try:
                r = fut.result()
                print(
                    f"[{done}/{len(jobs)}] {r['id']} success={r['metrics']['success']} "
                    f"step_acc={r['step_acc']:.3f}",
                    flush=True,
                )
            except Exception as e:
                print(f"[{done}/{len(jobs)}] ERROR {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
