"""Environment-grounded R_L2 from key actions / related objects (not model self-report)."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence


def target_ids_from_actions(actions: Sequence[Dict[str, Any]]) -> List[str]:
    """Prefer leaf objects listed in relatedObject of the final/end action."""
    ids: List[str] = []
    for a in actions:
        for oid in a.get("relatedObject") or []:
            if oid and oid not in ids:
                ids.append(oid)
    # drop pure receptacles if both present: keep all; downstream uses types too
    return ids


def key_action_coverage(actions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Which rewarded key actions were present in the executed trajectory."""
    rewarded = [a for a in actions if float(a.get("reward") or 0) > 0]
    return {
        "n_key": len(rewarded),
        "totalreward_meta": sum(float(a.get("reward") or 0) for a in actions),
        "verbs": [a.get("action") for a in rewarded],
    }


def compute_R_L2(
    *,
    executed: Sequence[Dict[str, Any]],
    key_actions: Sequence[Dict[str, Any]],
    tasktype: str = "",
) -> int:
    """
    Delayed task success.
    Default: executed trajectory covers all non-end key actions (objectId match when present).
    """
    keys = [a for a in key_actions if (a.get("action") or "").lower() != "end"]
    if not keys:
        # fall back: executed ends and has any navigate/toggle/pickup
        verbs = [(a.get("action") or a.get("verb") or "").lower() for a in executed]
        return int(verbs[-1:] == ["end"] and any(v in verbs for v in ("navigate to", "pickup", "toggle", "put in", "put on")))

    def norm(a: Dict[str, Any]) -> tuple:
        verb = (a.get("action") or a.get("verb") or "").lower()
        oid = a.get("objectId") or ""
        otype = (a.get("objectType") or "").lower()
        return verb, oid, otype

    exec_set = {norm(a) for a in executed}
    exec_verbs_types = {(v, t) for v, _, t in exec_set}
    exec_verbs_ids = {(v, i) for v, i, _ in exec_set if i}

    for k in keys:
        v, oid, otype = norm(k)
        if oid and (v, oid) in exec_verbs_ids:
            continue
        if otype and (v, otype) in exec_verbs_types:
            continue
        # soft: same verb+type with empty id on either side
        if any(ev == v and (et == otype or not otype or not et) for ev, ei, et in exec_set):
            continue
        return 0

    # must terminate
    last = (executed[-1].get("action") or executed[-1].get("verb") or "").lower() if executed else ""
    if last != "end":
        return 0
    return 1


def spatial_weights(qs: Sequence[float], R_L2: int, eps: float = 1e-8) -> List[float]:
    if R_L2 == 0:
        e = [1.0 - q for q in qs]
        s = sum(e) + eps
        return [x / s for x in e]
    s = sum(qs) + eps
    return [q / s for q in qs]


def type_scores(qs: Sequence[float], R_L2: int, eps: float = 1e-8):
    """Paper Eq. (7): independently normalize γ^k over the trajectory.

    γ^{perc}_t = q_t if R_L2=1 else (1-q_t);  γ^{reas}_t = q_t.
    s^k_t = γ^k_t / sum_i γ^k_i.
    On success this yields s^{perc}=s^{reas}=w_t.
    """
    if R_L2 == 0:
        s_perc = [1.0 - q for q in qs]
        s_reas = list(qs)
    else:
        s_perc = list(qs)
        s_reas = list(qs)
    sp, sr = sum(s_perc) + eps, sum(s_reas) + eps
    return [x / sp for x in s_perc], [x / sr for x in s_reas]


def gated_advantages(
    ws: Sequence[float],
    s_perc: Sequence[float],
    s_reas: Sequence[float],
    A_traj: float,
    lam: float = 1.0,
):
    """Paper Eq. (8): A^k_t = ((1-λ) w_t + λ s^k_t) A_t.

    Uses the independently normalized type scores s^k from ``type_scores``.
    Do **not** multiply w_t by raw q_t / (1-q_t) (that produced q_t^2-like
    coefficients and diverged from the paper).
    """
    if not (len(ws) == len(s_perc) == len(s_reas)):
        raise ValueError("ws, s_perc, s_reas must have equal length")
    ap, ar = [], []
    for w, sp, sr in zip(ws, s_perc, s_reas):
        ap.append(((1.0 - lam) * float(w) + lam * float(sp)) * A_traj)
        ar.append(((1.0 - lam) * float(w) + lam * float(sr)) * A_traj)
    return ap, ar


def paper_eq8_coefficients(
    qs: Sequence[float],
    R_L2: int,
    lam: float = 1.0,
    eps: float = 1e-8,
):
    """Convenience: w_t, s^k_t, A^k_t exactly as Eqs. (5)/(7)/(8)."""
    ws = spatial_weights(qs, R_L2, eps=eps)
    sp, sr = type_scores(qs, R_L2, eps=eps)
    A_traj = float(2 * int(R_L2) - 1)  # At = 2 R_L2 - 1 ∈ {-1,+1}
    ap, ar = gated_advantages(ws, sp, sr, A_traj, lam=lam)
    return {
        "w_t": ws,
        "s_t_perc": sp,
        "s_t_reas": sr,
        "A_traj": A_traj,
        "A_t_perc": ap,
        "A_t_reas": ar,
    }


def error_type(q: float, R_L2: int) -> str:
    if R_L2 == 1:
        return "success_step" if q >= 0.5 else "lucky_success_low_q"
    if q < 0.4:
        return "seeing_wrong"
    if q >= 0.6:
        return "thinking_wrong"
    return "mixed"
