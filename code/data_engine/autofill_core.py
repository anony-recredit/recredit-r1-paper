"""Shared ER multiturn → recredit_r1.v1 autofill logic."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from recredit_data_engine.annotate import annotate_trajectory, load_scene_metadata
from recredit_data_engine.reward import (
    compute_R_L2,
    error_type,
    gated_advantages,
    spatial_weights,
    type_scores,
)

DM_RE = re.compile(r"<DecisionMaking>(.*?)</DecisionMaking>", re.S | re.I)
VERB_RE = re.compile(
    r"^(navigate to|pickup|pick up|put in|put on|open|close|toggle|observe|end|move forward)\s*(.*)$",
    re.I,
)


def human_inst(item: dict) -> str:
    for m in item.get("messages") or []:
        if m.get("role") != "user":
            continue
        c = m.get("content") or ""
        m2 = re.search(r'Task:\s*"([^"]+)"', c)
        if m2:
            return m2.group(1).strip()
        m3 = re.search(r"Task:\s*(.+)$", c, re.M)
        if m3:
            return m3.group(1).strip().strip('"')
    return ""


def scene_from_traj(traj: str) -> str:
    m = re.match(r"(FloorPlan\d+)", traj)
    return m.group(1) if m else "FloorPlan1"


def parse_dm(text: str) -> dict:
    m = DM_RE.search(text or "")
    raw = (m.group(1).strip() if m else "").strip()
    if not raw:
        return {"action": "observe", "objectType": "", "objectId": "", "reward": 0}
    mm = VERB_RE.match(raw)
    if not mm:
        return {"action": raw.lower(), "objectType": "", "objectId": "", "reward": 0}
    verb = mm.group(1).lower().replace("pick up", "pickup")
    obj = re.sub(r"^(the|a|an)\s+", "", (mm.group(2) or "").strip(), flags=re.I)
    return {
        "action": verb,
        "objectType": obj,
        "objectId": "",
        "reward": 0 if verb in ("end", "observe", "move forward") else 1,
    }


def success_language(text: str) -> bool:
    t = (text or "").lower()
    pos = any(
        w in t
        for w in [
            "successfully",
            "task is complete",
            "completed effectively",
            "has been found",
            "successfully achieved",
            "task objective",
            "task appears completed",
            "appears to be complete",
            "completion of the task",
            "no further required",
            "no further action is required",
            "successful interaction",
            "now complete",
            "i conclude that the task",
            "holding the",
            "retrieved the target",
            "completing the task",
        ]
    )
    neg = any(
        w in t
        for w in [
            "unable to",
            "cannot find",
            "could not complete",
            "failed to",
            "still missing",
            "not able to",
        ]
    )
    return pos and not neg


def infer_R_L2(tasktype: str, assistants: list[str], actions: list[dict]) -> int:
    last = assistants[-1] if assistants else ""
    if success_language(last):
        return 1
    keys = [a for a in actions if (a.get("action") or "").lower() != "end"]
    verbs = [(a.get("action") or "").lower() for a in actions]
    if "fault" in (tasktype or "") and verbs[-1:] != ["end"]:
        return 0
    if verbs[-1:] == ["end"] and keys:
        r = compute_R_L2(executed=actions, key_actions=actions, tasktype=tasktype)
        return 1 if success_language(last) else (0 if "fault" in (tasktype or "") else r)
    if "fault" in (tasktype or ""):
        return 0
    return int(verbs[-1:] == ["end"])


def index_train(train: list) -> dict[tuple[str, str], int]:
    by: dict[tuple[str, str], int] = {}
    for i, item in enumerate(train):
        key = traj_key_from_item(item)
        if key:
            by[key] = i
    return by


def traj_key_from_item(item: dict) -> tuple[str, str] | None:
    imgs = item.get("images") or []
    if not imgs:
        return None
    parts = Path(str(imgs[0]).replace("\\", "/")).parts
    if "images" not in parts:
        return None
    j = parts.index("images")
    if len(parts) > j + 2:
        return parts[j + 1], parts[j + 2]
    return None


def rel_img(root: Path, p: Path | str) -> str:
    path = Path(str(p).replace("\\", "/"))
    if path.is_absolute():
        return str(path.resolve().relative_to(root.resolve()))
    return str(path).lstrip("./")


def resolve_image_paths(root: Path, item: dict, require_exists: bool) -> tuple[str, str, list[str]] | None:
    key = traj_key_from_item(item)
    if not key:
        return None
    tt, traj = key
    rels: list[str] = []
    for raw in item.get("images") or []:
        rel = rel_img(root, raw)
        if require_exists and not (root / rel).exists():
            return None
        rels.append(rel)
    return tt, traj, rels


def local_trajs(img_root: Path) -> list[tuple[str, str, list[Path]]]:
    out: list[tuple[str, str, list[Path]]] = []
    if not img_root.exists():
        return out
    for tt_dir in sorted(img_root.iterdir()):
        if not tt_dir.is_dir():
            continue
        for traj_dir in sorted(tt_dir.iterdir()):
            if not traj_dir.is_dir():
                continue
            pngs = sorted(traj_dir.glob("*.png"))
            if pngs:
                out.append((tt_dir.name, traj_dir.name, pngs))
    return out


def build_one(
    root: Path,
    meta_dir: Path,
    tt: str,
    traj: str,
    image_paths: list[str],
    item: dict,
) -> dict | None:
    scene = scene_from_traj(traj)
    meta_path = meta_dir / f"{scene}.json"
    if not meta_path.exists():
        return None
    scene_meta = load_scene_metadata(meta_path)

    assistants = [m["content"] for m in item.get("messages") or [] if m.get("role") == "assistant"]
    if not assistants:
        return None

    n = min(len(image_paths), len(assistants))
    if n == 0:
        return None
    imgs_n = image_paths[:n]
    asst = assistants[:n]
    executed = [parse_dm(t) for t in asst]
    all_actions = [parse_dm(t) for t in assistants]
    if not any((a.get("action") or "").lower() == "end" for a in all_actions):
        all_actions = all_actions + [{"action": "end", "objectType": "", "objectId": "", "reward": 0}]

    instruction = human_inst(item) or f"Task in {scene}"
    R_hint = infer_R_L2(tt, assistants, all_actions)

    ex = annotate_trajectory(
        scene=scene,
        tasktype=tt,
        instruction=instruction,
        key_actions=all_actions,
        scene_meta=scene_meta,
        executed_actions=executed,
        images=imgs_n,
        assistant_texts=asst,
        traj_id=f"{scene}_{traj}",
    )

    R = R_hint
    qs = [s["q_t"] for s in ex["steps"]]
    ws = spatial_weights(qs, R)
    sp, sr = type_scores(qs, R)
    A_traj = float(2 * int(R) - 1)  # paper: A_t = 2 R_L2 - 1
    ap, ar = gated_advantages(ws, sp, sr, A_traj, lam=1.0)
    for s, w, spp, srr, app, arr in zip(ex["steps"], ws, sp, sr, ap, ar):
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
    ex["R_L2"] = R
    ex["is_failure_traj"] = R == 0
    ex["A_traj_placeholder"] = A_traj
    ex["mean_q_t"] = round(sum(qs) / max(len(qs), 1), 4)
    ex["cliff_ratio_flag"] = bool(R == 0 and ex["mean_q_t"] >= 0.85)
    ex["messages"] = item.get("messages")
    ex["image_source"] = "embodied_reasoner_original"
    ex["source_traj"] = traj
    ex["failure_source"] = "env_fault_or_language" if R == 0 else None
    for s in ex["steps"]:
        s["image_source"] = "embodied_reasoner_original"
    return ex


def to_view(blob: dict) -> dict:
    return {
        "paper": blob["paper"],
        "n": blob["n"],
        "n_success": blob["n_success"],
        "n_failure": blob["n_failure"],
        "image_source": blob.get("image_source", "embodied_reasoner_original"),
        "examples": [
            {
                "id": e["id"],
                "scene": e["scene"],
                "tasktype": e["tasktype"],
                "instruction": e["instruction"],
                "R_L2": e["R_L2"],
                "is_failure_traj": e["is_failure_traj"],
                "n_steps": e["n_steps"],
                "mean_q_t": e["mean_q_t"],
                "images": e["images"],
                "steps": [
                    {
                        "t": s["t"],
                        "image": s["image"],
                        "action": s["action"],
                        "q_t": s["q_t"],
                        "error_type": s["error_type"],
                        "w_t": s["w_t"],
                        "A_t_perc": s["A_t_perc"],
                        "A_t_reas": s["A_t_reas"],
                        "spans": s["spans"],
                        "g_t_structured": {
                            "objectId": (s.get("g_t_structured") or {}).get("objectId"),
                            "objectType": (s.get("g_t_structured") or {}).get("objectType"),
                            "world_point": (s.get("g_t_structured") or {}).get("world_point"),
                        },
                        "s_t_perc": s["s_t_perc"],
                        "s_t_reas": s["s_t_reas"],
                    }
                    for s in e["steps"]
                ],
            }
            for e in blob["examples"]
        ],
    }


def load_train(path: Path) -> list[dict]:
    return json.loads(path.read_text())
