"""Rule-verifiable geometric quality q_t from AI2-THOR metadata AABB."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


def aabb_center(obj: Dict[str, Any]) -> Dict[str, float]:
    aabb = obj.get("axisAlignedBoundingBox") or {}
    c = aabb.get("center") or obj["position"]
    return {k: float(c[k]) for k in ("x", "y", "z")}


def aabb_half(obj: Dict[str, Any]) -> Dict[str, float]:
    aabb = obj.get("axisAlignedBoundingBox") or {}
    s = aabb.get("size") or {"x": 0.2, "y": 0.2, "z": 0.2}
    return {k: float(s[k]) / 2.0 for k in ("x", "y", "z")}


def point_to_aabb_dist(point: Dict[str, float], obj: Dict[str, Any]) -> float:
    c, h = aabb_center(obj), aabb_half(obj)
    dx = max(0.0, abs(point["x"] - c["x"]) - h["x"])
    dy = max(0.0, abs(point["y"] - c["y"]) - h["y"])
    dz = max(0.0, abs(point["z"] - c["z"]) - h["z"])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def q_from_dist(dist: float, scale: float = 1.25) -> float:
    """Map 3D point–AABB distance to q_t ∈ (0, 1]."""
    return float(1.0 / (1.0 + dist / scale))


def index_objects(scene_meta: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for o in scene_meta.get("objects") or []:
        by_type.setdefault(o["objectType"], []).append(o)
    return by_type


def find_by_id(scene_meta: Dict[str, Any], object_id: str) -> Optional[Dict[str, Any]]:
    for o in scene_meta.get("objects") or []:
        if o.get("objectId") == object_id:
            return o
    return None


def find_by_type(by_type: Dict[str, List[Dict[str, Any]]], object_type: str) -> Optional[Dict[str, Any]]:
    if not object_type:
        return None
    for k, vs in by_type.items():
        if k.lower() == object_type.lower():
            vis = [o for o in vs if o.get("visible")]
            return (vis or vs)[0]
    return None


def receptacle_contains(action_obj: Dict[str, Any], targets: Sequence[Dict[str, Any]]) -> bool:
    ids = set(action_obj.get("receptacleObjectIds") or [])
    aid = action_obj.get("objectId")
    for t in targets:
        if t.get("objectId") in ids:
            return True
        if aid in (t.get("parentReceptacles") or []):
            return True
    return False


def compute_q_t(
    *,
    action_verb: str,
    action_object: Optional[Dict[str, Any]],
    target_objects: Sequence[Dict[str, Any]],
    grounding_point: Optional[Dict[str, float]] = None,
    success_end: bool = False,
) -> float:
    """
    Minimal Recredit-R1 Stage-I style verifier:
    q_t from point–AABB / object–target alignment (no learned PRM).
    """
    verb = (action_verb or "").lower().strip()
    if verb == "end":
        return 1.0 if success_end else 0.25
    if verb == "observe":
        return 0.55

    targets = list(target_objects)
    if action_object and targets:
        # exact target
        if any(action_object.get("objectId") == t.get("objectId") for t in targets):
            return 1.0
        if any(action_object.get("objectType") == t.get("objectType") for t in targets):
            return 1.0
        if receptacle_contains(action_object, targets):
            return 0.92

    point = grounding_point or (aabb_center(action_object) if action_object else None)
    if point and targets:
        d = min(point_to_aabb_dist(point, t) for t in targets)
        return max(0.02, min(1.0, q_from_dist(d)))

    if verb in ("navigate to", "open", "close") and action_object:
        return 0.45
    return 0.2
