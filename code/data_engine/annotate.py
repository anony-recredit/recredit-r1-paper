"""Build Recredit-R1.v1 examples from (instruction, key actions, optional images/thoughts)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .geometry import (
    aabb_center,
    compute_q_t,
    find_by_id,
    find_by_type,
    index_objects,
)
from .reward import (
    compute_R_L2,
    error_type,
    gated_advantages,
    spatial_weights,
    target_ids_from_actions,
    type_scores,
)

DM_RE = re.compile(r"<DecisionMaking>(.*?)</DecisionMaking>", re.S | re.I)


def load_scene_metadata(path: Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text())
    return data[0] if isinstance(data, list) else data


def split_assistant(text: str) -> Dict[str, Any]:
    m = DM_RE.search(text or "")
    if not m:
        return {
            "think_t": (text or "").strip(),
            "g_t": "",
            "a_t": "",
            "y_t": text or "",
            "char_spans": {
                "think_t": [0, len(text or "")],
                "g_t": [0, 0],
                "a_t": [0, 0],
                "P_t": [0, 0],
                "R_t_think": [0, len(text or "")],
                "R_t_action": [0, 0],
            },
            "decision_raw": "",
        }
    think = (text or "")[: m.start()].strip()
    dm = m.group(1).strip()
    # objectType ≈ g_t for ER-style DecisionMaking
    obj = ""
    mm = re.match(
        r"(navigate to|pickup|put in|put on|open|close|toggle)\s+(.+)$",
        dm,
        re.I,
    )
    if mm:
        obj = re.sub(r"^(the|a|an)\s+", "", mm.group(2).strip(), flags=re.I)
    g_span = [m.start(), m.start()]
    if obj:
        rel = dm.lower().rfind(obj.lower())
        if rel >= 0:
            abs0 = m.start(1) + rel
            g_span = [abs0, abs0 + len(obj)]
    return {
        "think_t": think,
        "g_t": obj,
        "a_t": dm,
        "y_t": text or "",
        "char_spans": {
            "think_t": [0, m.start()],
            "g_t": g_span,
            "a_t": [m.start(), m.end()],
            "P_t": g_span,
            "R_t_think": [0, m.start()],
            "R_t_action": [m.start(), m.end()],
        },
        "decision_raw": dm,
    }


def action_from_key(a: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verb": (a.get("action") or a.get("verb") or "").lower(),
        "objectType": a.get("objectType") or None,
        "objectId": a.get("objectId") or "",
        "raw": f"{a.get('action', a.get('verb', ''))}"
        + (f" {a['objectType']}" if a.get("objectType") else "").rstrip(),
        "relatedObject": list(a.get("relatedObject") or []),
        "reward": a.get("reward", 0),
    }


def default_thought(step_i: int, action: Dict[str, Any], instruction: str) -> str:
    """Placeholder CoT when VLM thought synthesis is not run yet."""
    return (
        f"<situation analysis> Step {step_i}: following key-action plan for: {instruction} "
        f"</situation analysis>\n"
        f"<Planning> Execute {action['raw']}. </Planning>\n"
        f"Hmm..., I've settled on a choice.<DecisionMaking>{action['raw']}</DecisionMaking>"
    )


def annotate_trajectory(
    *,
    scene: str,
    tasktype: str,
    instruction: str,
    key_actions: Sequence[Dict[str, Any]],
    scene_meta: Dict[str, Any],
    executed_actions: Optional[Sequence[Dict[str, Any]]] = None,
    images: Optional[Sequence[str]] = None,
    assistant_texts: Optional[Sequence[str]] = None,
    traj_id: Optional[str] = None,
    lambda_type: float = 1.0,
) -> Dict[str, Any]:
    """
    Core engine step after key actions (and optional RGB/thoughts) exist.

    executed_actions: what was actually run (defaults to key_actions = oracle success).
    images / assistant_texts: aligned per step (len ~= n executed).
    """
    executed = list(executed_actions or key_actions)
    by_type = index_objects(scene_meta)

    # targets = leaf relatedObject ids from metadata when possible
    tid_list = target_ids_from_actions(key_actions)
    target_objs = []
    for oid in tid_list:
        o = find_by_id(scene_meta, oid)
        if o:
            # prefer non-huge receptacles: keep all; filter later by pickupable if many
            target_objs.append(o)
    # heuristic: prefer pickupable / toggleable as true targets
    preferred = [
        o
        for o in target_objs
        if o.get("pickupable") or o.get("toggleable") or not o.get("receptacle")
    ]
    if preferred:
        target_objs = preferred

    R = compute_R_L2(executed=executed, key_actions=key_actions, tasktype=tasktype)
    steps: List[Dict[str, Any]] = []

    for t, act_raw in enumerate(executed):
        act = action_from_key(act_raw)
        obj = None
        if act.get("objectId"):
            obj = find_by_id(scene_meta, act["objectId"])
        if obj is None and act.get("objectType"):
            obj = find_by_type(by_type, act["objectType"])

        g_point = aabb_center(obj) if obj else None
        q = compute_q_t(
            action_verb=act["verb"],
            action_object=obj,
            target_objects=target_objs,
            grounding_point=g_point,
            success_end=(R == 1),
        )

        if assistant_texts and t < len(assistant_texts):
            y = assistant_texts[t]
        else:
            y = default_thought(t, act, instruction)
        spans = split_assistant(y)

        img = None
        if images and t < len(images):
            img = images[t]

        steps.append(
            {
                "t": t,
                "image": img,
                "action": act,
                "q_t": round(float(q), 4),
                "spans": {
                    "think_t": spans["think_t"],
                    "g_t": spans["g_t"],
                    "a_t": spans["a_t"],
                    "y_t": spans["y_t"],
                    "char_spans": spans["char_spans"],
                    "P_t_equals": "g_t",
                    "R_t_equals": "(think_t, a_t)",
                },
                "g_t_structured": {
                    "objectType": act.get("objectType"),
                    "objectId": (obj or {}).get("objectId") or act.get("objectId"),
                    "world_point": g_point,
                    "aabb": (obj or {}).get("axisAlignedBoundingBox"),
                },
            }
        )

    qs = [s["q_t"] for s in steps]
    ws = spatial_weights(qs, R)
    sp, sr = type_scores(qs, R)
    A_traj = float(2 * int(R) - 1)  # paper: A_t = 2 R_L2 - 1
    ap, ar = gated_advantages(ws, sp, sr, A_traj, lam=lambda_type)
    mean_q = sum(qs) / max(len(qs), 1)

    for s, w, spp, srr, app, arr in zip(steps, ws, sp, sr, ap, ar):
        s.update(
            {
                "w_t": round(w, 6),
                "s_t_perc": round(spp, 6),
                "s_t_reas": round(srr, 6),
                "A_t_perc": round(app, 6),
                "A_t_reas": round(arr, 6),
                "A_t_spatial": round(w * A_traj, 6),
                "error_type": error_type(s["q_t"], R),
                "R_L2": R,
            }
        )

    n = len(steps)
    role = (
        "Stage I short"
        if n <= 3
        else ("Stage I→II" if n <= 5 else ("Stage II long" if n <= 12 else "Stage II extreme"))
    )
    traj = traj_id or f"{scene}_{tasktype}_gen"

    return {
        "id": traj,
        "scene": scene,
        "traj": traj,
        "tasktype": tasktype,
        "instruction": instruction,
        "role_horizon": role,
        "paper_stage_hint": "Stage I" if n <= 4 else "Stage II",
        "is_failure_traj": R == 0,
        "target_objects": [o.get("objectId") for o in target_objs],
        "target_object_types": list({o.get("objectType") for o in target_objs}),
        "task_metadata": {
            "taskname": instruction,
            "tasktype": tasktype,
            "actions": list(key_actions),
            "totalreward": sum(float(a.get("reward") or 0) for a in key_actions),
        },
        "R_L2": R,
        "A_traj_placeholder": A_traj,
        "mean_q_t": round(mean_q, 4),
        "cliff_ratio_flag": bool(R == 0 and mean_q >= 0.85),
        "lambda_type_routing": lambda_type,
        "n_steps": n,
        "images": [s["image"] for s in steps],
        "steps": steps,
        "schema_version": "recredit_r1.v1",
        "pipeline": "instruction→key_actions→(optional RGB/thought)→AABB q_t→env R_L2",
    }
