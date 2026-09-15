#!/usr/bin/env python3
"""Build attribution-gated DPO pairs for ablation v28 with hard quotas.

v27 skeleton kept: pairs defined by (q_t, w_t, e_t); DPO only optimizes action spans.

v28 change vs v27:
  - Hard quotas — kill the seeing sea.
  - Two attribution preference families:
      * long-horizon thinking (holding/bridge/seg2/thinking_wrong)
      * closerep seeing (open/search/pickup-from-closerep gated seeing_wrong)
  - Cap generic seeing_wrong.

Quotas (fractions of final set, soft-filled then capped):
  C_skill (holding+bridge+seg2) : 0.38
  closerep_seeing               : 0.28
  thinking_wrong                : 0.18
  anti_loop                     : 0.08
  seeing_wrong (generic)        : 0.08   (hard cap)
"""
from __future__ import annotations

import os

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from action_format import normalize_put_raw
from convert_recredit_to_parquet import _resolve_image, _step_action_raw, _step_response
from er_prompts import first_user_text, followup_user_text

BASE = Path(os.environ["RECREDIT_ROOT"])
PACK = BASE / "data/wave23_skill_pack"
SHARED_V28 = BASE / "data/wave23_shared_p123_v28/shared_train.json"
SHARED_V24 = BASE / "data/wave23_shared_p123_skill22/shared_train.json"
ANTI = BASE / "data/wave23_shared_p123_v28/closerep_anti_illegal.json"
SEG2_PACK = BASE / "data/wave23_seg2_pack_fixed/recredit_r1_wave23_train_seg2pack_fixed.json"
TEST_C = BASE / "data/eval/test_C_105.json"
TEST_809 = Path(os.environ.get("EMBODIED_REASONER_ROOT", "/path/to/embodied_reasoner")) / "test_809.json"
N806 = BASE / "data/eval/test_809_n806_no_emulator3.json"
OUT_DIR = BASE / "data/parquet_p123_shared/P3_recredit_ablation_v28_dpo"
DATA_ROOT = Path(os.environ.get("EMBODIED_REASONER_ROOT", "/path/to/embodied_reasoner"))

RECEPTACLES = [
    "GarbageCan", "SinkBasin", "CounterTop", "Fridge", "Cabinet",
    "Microwave", "Pan", "Plate", "Sofa", "Chair", "SideTable", "Drawer",
]
Q_LOW = 0.35
Q_HIGH = 0.65
SEED = 28

# Absolute caps before quota sample (mine more, then downsample)
MINE_CAP = {
    "seeing_wrong": 4000,
    "closerep_seeing": 4000,
    "thinking_wrong": 4000,
    "anti_loop": 3000,
}
TARGET_TOTAL = 3200
QUOTAS = {
    "holding": 0.14,
    "bridge": 0.12,
    "seg2": 0.12,          # C_skill sum = 0.38
    "closerep_seeing": 0.28,
    "thinking_wrong": 0.18,
    "anti_loop": 0.08,
    "seeing_wrong": 0.08,  # hard cap family
}


def _load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    x = json.loads(path.read_text())
    items = x if isinstance(x, list) else (x.get("examples") or x.get("data") or [])
    return {str(e.get("identity")) for e in items if e.get("identity") is not None}


def forbidden_identities() -> set[str]:
    return _load_ids(TEST_C) | _load_ids(TEST_809) | _load_ids(N806)


def load_examples(name: str) -> list[dict]:
    p = PACK / name
    if not p.exists():
        print(f"[warn] missing skill file {p}")
        return []
    return json.loads(p.read_text())["examples"]


def base_id(ex: dict) -> str:
    return (ex.get("id") or "").split("__skill_")[0].split("__v28_")[0]


def identity_of(ex: dict) -> str | None:
    if ex.get("identity") is not None:
        return str(ex["identity"])
    m = re.match(r"^(\d+)_", ex.get("id") or "")
    return m.group(1) if m else None


def is_forbidden(ex: dict, bad: set[str]) -> bool:
    ident = identity_of(ex)
    if ident and ident in bad:
        return True
    m = re.match(r"^(\d+)_", ex.get("id") or "")
    return bool(m and m.group(1) in bad)


def raw_of(step: dict) -> str:
    a = step.get("action")
    if isinstance(a, dict):
        return normalize_put_raw((a.get("raw") or "").strip())
    return normalize_put_raw(str(a or "").strip())


def q_of(step: dict) -> float:
    try:
        return float(step.get("q_t") if step.get("q_t") is not None else step.get("q") or 0.0)
    except Exception:
        return 0.0


def w_of(step: dict) -> float:
    for k in ("w_t", "spatial_weight", "credit_w"):
        if step.get(k) is not None:
            try:
                return float(step[k])
            except Exception:
                pass
    return q_of(step)


def e_of(step: dict) -> str:
    e = step.get("e_t") or step.get("error_type") or step.get("problem_type") or step.get("type") or ""
    return str(e).strip().lower()


def action_key(raw: str) -> tuple[str, str]:
    toks = normalize_put_raw(raw).split()
    if not toks:
        return ("", "")
    if len(toks) == 1:
        return (toks[0], "")
    return (toks[0], toks[-1])


def action_only_response(a_t: str) -> str:
    return f"<DecisionMaking>{normalize_put_raw(a_t)}</DecisionMaking>"


def is_closerep_ex(ex: dict) -> bool:
    tt = str(ex.get("tasktype") or "").lower()
    if "closerep" in tt:
        return True
    tags = ex.get("skill_tags") or {}
    if str(tags.get("v28_from") or "").startswith("closerep") or "anti_illegal" in str(tags.get("v28_from") or ""):
        return True
    acts = [raw_of(s) for s in (ex.get("steps") or [])]
    return any(a.lower().startswith(("open ", "close ")) for a in acts) and any(
        a.lower().startswith("pickup ") for a in acts
    )


def history_fields(ex: dict, step_idx: int, data_root: Path) -> dict:
    steps = list(ex.get("steps") or [])
    history_images, history_responses, history_actions = [], [], []
    for j in range(step_idx):
        prev = steps[j]
        history_images.append(_resolve_image(prev.get("image") or "", data_root))
        history_responses.append(_step_response(prev))
        history_actions.append(normalize_put_raw(_step_action_raw(prev)))
    instruction = ex.get("instruction") or ""
    user_text = (
        first_user_text(instruction)
        if step_idx == 0
        else followup_user_text(history_actions[-1] if history_actions else "observe")
    )
    img = _resolve_image(steps[step_idx].get("image") or "", data_root)
    return {
        "instruction": instruction,
        "user_text": user_text,
        "history_images": json.dumps(history_images, ensure_ascii=False),
        "history_responses": json.dumps(history_responses, ensure_ascii=False),
        "history_actions": json.dumps(history_actions, ensure_ascii=False),
        "image": img,
        "scene": ex.get("scene"),
        "tasktype": ex.get("tasktype") or "",
        "traj_id": ex.get("id"),
        "t": int(steps[step_idx].get("t") if steps[step_idx].get("t") is not None else step_idx),
        "identity": identity_of(ex),
    }


def make_pair_row(
    *,
    kind: str,
    ctx: dict,
    chosen_a: str,
    rejected_a: str,
    pair_id: str,
    q_t: float,
    w_t: float,
    e_t: str,
    pair_weight: float = 1.0,
) -> dict | None:
    chosen_a = normalize_put_raw(chosen_a)
    rejected_a = normalize_put_raw(rejected_a)
    if not chosen_a or not rejected_a or chosen_a == rejected_a:
        return None
    if not ctx.get("image"):
        return None
    if "seeing" in kind or e_t in {"seeing-wrong", "seeing_wrong", "see"}:
        perc_w, reas_w = 1.0, 0.0
    else:
        perc_w, reas_w = 0.0, 1.0
    return {
        "pair_id": pair_id,
        "pair_kind": kind,
        "multiturn": True,
        "prompt_style": "er_available_actions",
        "schema_version": "recredit_r1.v28_attr_dpo",
        "chosen_a": chosen_a,
        "rejected_a": rejected_a,
        "chosen_response": action_only_response(chosen_a),
        "rejected_response": action_only_response(rejected_a),
        "think_t": "",
        "g_t": "",
        "q_t": float(q_t),
        "w_t": float(w_t),
        "e_t": e_t or "",
        "R_L2": 1,
        "perc_weight": perc_w,
        "reas_weight": reas_w,
        "pair_weight": float(pair_weight),
        **ctx,
    }


def near_open_close(steps: list[dict], i: int, radius: int = 3) -> bool:
    lo, hi = max(0, i - radius), min(len(steps), i + radius + 1)
    for j in range(lo, hi):
        a = raw_of(steps[j]).lower()
        if a.startswith("open ") or a.startswith("close "):
            return True
    return False


def build_attr_pools(examples: list[dict], bad: set[str]) -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {
        "seeing_wrong": [],
        "closerep_seeing": [],
        "thinking_wrong": [],
        "anti_loop": [],
    }
    for e in examples:
        if is_forbidden(e, bad):
            continue
        steps = e.get("steps") or []
        if len(steps) < 3:
            continue
        closerep = is_closerep_ex(e)
        for i in range(1, len(steps) - 1):
            cur, nxt = steps[i], steps[i + 1]
            a_cur, a_nxt = raw_of(cur), raw_of(nxt)
            if not a_cur or not a_nxt or a_cur in ("end", "init") or a_nxt in ("end", "init"):
                continue
            q_c, q_n = q_of(cur), q_of(nxt)
            w_c, w_n = w_of(cur), w_of(nxt)
            e_c = e_of(cur)
            ctx = history_fields(e, i + 1, DATA_ROOT)

            # seeing family
            if q_c <= Q_LOW:
                nav_like = action_key(a_cur)[0] in {"move", "turn", "look", "observe", "navigate"} or "see" in e_c
                # also treat redundant open/close / observe after open as seeing-wrong on closerep
                open_idle = (
                    closerep
                    and near_open_close(steps, i)
                    and action_key(a_cur)[0] in {"move", "turn", "look", "observe", "navigate", "open", "close"}
                )
                progressed = (w_n >= w_c + 0.02) or (
                    action_key(a_nxt)[0] in {"pickup", "put", "open", "close", "toggle"}
                )
                if (nav_like or open_idle) and progressed and a_nxt != a_cur:
                    kind = "closerep_seeing" if (closerep and near_open_close(steps, i)) else "seeing_wrong"
                    if len(pools[kind]) < MINE_CAP[kind]:
                        row = make_pair_row(
                            kind=kind,
                            ctx=ctx,
                            chosen_a=a_nxt,
                            rejected_a=a_cur,
                            pair_id=f"{kind}::{base_id(e)}::t{i}",
                            q_t=q_n,
                            w_t=w_n,
                            e_t=e_c or "seeing-wrong",
                            pair_weight=max(0.5, w_n + 0.5),
                        )
                        if row:
                            pools[kind].append(row)

            # thinking-wrong
            if len(pools["thinking_wrong"]) < MINE_CAP["thinking_wrong"] and q_c >= Q_HIGH:
                if action_key(a_cur) == action_key(raw_of(steps[i - 1])) and a_nxt != a_cur:
                    put_later = any(raw_of(steps[j]).startswith("put") for j in range(i + 1, min(len(steps), i + 4)))
                    nav_now = action_key(a_cur)[0] in {"move", "turn", "look", "observe"}
                    pw = max(0.5, w_n) + (0.25 if put_later and nav_now else 0.0)
                    # boost long-horizon / ordered / closerep put chains
                    if "ordered" in str(e.get("tasktype") or "") or closerep:
                        pw += 0.15
                    row = make_pair_row(
                        kind="thinking_wrong",
                        ctx=ctx,
                        chosen_a=a_nxt,
                        rejected_a=a_cur,
                        pair_id=f"thinking::{base_id(e)}::t{i}",
                        q_t=q_c,
                        w_t=w_n,
                        e_t=e_c or "thinking-wrong",
                        pair_weight=min(2.0, pw),
                    )
                    if row:
                        pools["thinking_wrong"].append(row)

            # anti_loop
            if len(pools["anti_loop"]) < MINE_CAP["anti_loop"] and i >= 2:
                a0, a1 = raw_of(steps[i - 2]), raw_of(steps[i - 1])
                if a0 == a1 == a_cur and a_nxt != a_cur and w_n >= w_c:
                    row = make_pair_row(
                        kind="anti_loop",
                        ctx=ctx,
                        chosen_a=a_nxt,
                        rejected_a=a_cur,
                        pair_id=f"antiloop::{base_id(e)}::t{i}",
                        q_t=q_n,
                        w_t=w_n,
                        e_t=e_c or "loop",
                        pair_weight=1.25,
                    )
                    if row:
                        pools["anti_loop"].append(row)
    return pools


def bad_holding_action(chosen: str, key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    m = re.match(r"^(put(?:\s+in)?)\s+(\S+)$", chosen.strip(), re.I)
    obj = m.group(2) if m else ""
    cands = ["end", "observe", "move forward"]
    if obj:
        cands.append(f"pickup {obj}")
    for r in RECEPTACLES:
        if r.lower() != obj.lower():
            cands.append(f"put {r}")
    for k in range(len(cands)):
        a = cands[(h + k) % len(cands)]
        if normalize_put_raw(a) != normalize_put_raw(chosen):
            return normalize_put_raw(a)
    return "end"


def bad_seg2_action(chosen: str, key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return normalize_put_raw(["observe", "end", "move forward"][h % 3])


def build_skill_inherit(bad: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"bridge": [], "holding": [], "seg2": [], "anti_loop": []}
    pos_b = [e for e in load_examples("pos_bridge_after_seg1.json") if not is_forbidden(e, bad)]
    neg_b = [
        e
        for e in load_examples("neg_bridge_stall_strict.json")
        if (e.get("skill_tags") or {}).get("stall_synth") and not is_forbidden(e, bad)
    ]
    pos_map = {base_id(e): e for e in pos_b}
    neg_map = {base_id(e): e for e in neg_b}
    for bid in sorted(set(pos_map) & set(neg_map)):
        p, n = pos_map[bid], neg_map[bid]
        tags = n.get("skill_tags") or {}
        t0 = tags.get("first_put_ok_t")
        if t0 is None:
            t0 = tags.get("seg1_end_t")
        if t0 is None:
            continue
        t0 = int(t0)
        ti = t0 + 1
        if ti >= len(p.get("steps") or []) or ti >= len(n.get("steps") or []):
            continue
        if not all(raw_of(p["steps"][i]) == raw_of(n["steps"][i]) for i in range(t0 + 1)):
            continue
        st = p["steps"][ti]
        row = make_pair_row(
            kind="bridge",
            ctx=history_fields(p, ti, DATA_ROOT),
            chosen_a=raw_of(st),
            rejected_a=raw_of(n["steps"][ti]),
            pair_id=f"bridge::{bid}::t{ti}",
            q_t=q_of(st),
            w_t=w_of(st),
            e_t=e_of(st) or "thinking-wrong",
            pair_weight=1.15,
        )
        if row:
            out["bridge"].append(row)

    for e in load_examples("pos_holding_second_put.json"):
        if is_forbidden(e, bad):
            continue
        sp = (e.get("skill_tags") or {}).get("second_put_t")
        if sp is None:
            continue
        sp = int(sp)
        steps = e.get("steps") or []
        if sp < 0 or sp >= len(steps):
            continue
        chosen_a = raw_of(steps[sp])
        row = make_pair_row(
            kind="holding",
            ctx=history_fields(e, sp, DATA_ROOT),
            chosen_a=chosen_a,
            rejected_a=bad_holding_action(chosen_a, e.get("id") or str(sp)),
            pair_id=f"holding::{base_id(e)}::t{sp}",
            q_t=q_of(steps[sp]),
            w_t=w_of(steps[sp]),
            e_t=e_of(steps[sp]) or "thinking-wrong",
            pair_weight=1.15,
        )
        if row:
            out["holding"].append(row)

    # spam → anti_loop
    seen = set()
    for src, examples in [
        ("switch", load_examples("switch_local_pref.json")),
        ("spam", load_examples("neg_repeat_spam.json")),
    ]:
        for e in examples:
            if is_forbidden(e, bad):
                continue
            steps = e.get("steps") or []
            for ev in (e.get("skill_tags") or {}).get("events") or []:
                if not ev.get("good_switch"):
                    continue
                st = int(ev["start_t"])
                sw = int(ev["switch_t"])
                if st < 0 or sw < 0 or sw >= len(steps) or st >= len(steps):
                    continue
                chosen_a = normalize_put_raw(ev.get("next_raw") or "") or raw_of(steps[sw])
                rejected_a = raw_of(steps[st])
                if sw - 1 > st:
                    mid = raw_of(steps[sw - 1])
                    if mid.split()[:1] == rejected_a.split()[:1]:
                        rejected_a = mid
                pid = f"spam::{src}::{base_id(e)}::t{sw}"
                if pid in seen:
                    continue
                row = make_pair_row(
                    kind="anti_loop",
                    ctx=history_fields(e, sw, DATA_ROOT),
                    chosen_a=chosen_a,
                    rejected_a=rejected_a,
                    pair_id=pid,
                    q_t=q_of(steps[sw]),
                    w_t=max(w_of(steps[sw]), 1.0),
                    e_t="loop",
                    pair_weight=1.25,
                )
                if row:
                    seen.add(pid)
                    out["anti_loop"].append(row)

    if SEG2_PACK.exists():
        examples = json.loads(SEG2_PACK.read_text()).get("examples") or []
        for e in examples:
            if is_forbidden(e, bad) or int(e.get("R_L2") or 0) != 1:
                continue
            bounds = e.get("segment_boundaries")
            if not isinstance(bounds, list) or len(bounds) < 2:
                continue
            ti = int(bounds[1])
            steps = e.get("steps") or []
            if ti < 0 or ti >= len(steps):
                continue
            chosen_a = raw_of(steps[ti])
            if not chosen_a or chosen_a in ("end", "init"):
                continue
            row = make_pair_row(
                kind="seg2",
                ctx=history_fields(e, ti, DATA_ROOT),
                chosen_a=chosen_a,
                rejected_a=bad_seg2_action(chosen_a, e.get("id") or str(ti)),
                pair_id=f"seg2::{base_id(e)}::t{ti}",
                q_t=q_of(steps[ti]),
                w_t=w_of(steps[ti]),
                e_t=e_of(steps[ti]) or "thinking-wrong",
                pair_weight=1.1,
            )
            if row:
                out["seg2"].append(row)

    # anti-illegal open loops → closerep_seeing rejects
    if ANTI.exists():
        for e in json.loads(ANTI.read_text()).get("examples") or []:
            if is_forbidden(e, bad):
                continue
            steps = e.get("steps") or []
            if len(steps) < 2:
                continue
            # chosen = a plausible progress action from original context: pickup/navigate
            # rejected = repeated open
            rejected_a = raw_of(steps[1]) if raw_of(steps[1]).lower().startswith("open ") else raw_of(steps[0])
            chosen_a = "pickup Apple"  # placeholder overwritten below if possible
            # Prefer next non-open from step0's receptacle family: navigate away / pickup
            chosen_a = "observe"
            for cand in ("pickup Knife", "pickup Apple", "navigate to CounterTop", "close Drawer"):
                if normalize_put_raw(cand) != normalize_put_raw(rejected_a):
                    chosen_a = cand
                    break
            # If step0 is open X, chosen=close X is a legal recovery
            a0 = raw_of(steps[0])
            if a0.lower().startswith("open "):
                chosen_a = "close " + a0.split()[-1]
            row = make_pair_row(
                kind="closerep_seeing",
                ctx=history_fields(e, 1, DATA_ROOT),
                chosen_a=chosen_a,
                rejected_a=rejected_a,
                pair_id=f"closerep_illegal::{base_id(e)}::t1",
                q_t=q_of(steps[1]),
                w_t=0.2,
                e_t="seeing-wrong",
                pair_weight=1.2,
            )
            if row:
                out.setdefault("closerep_seeing", [])
                out["closerep_seeing"].append(row)
    return out


def apply_quotas(pools: dict[str, list[dict]], rng: random.Random) -> list[dict]:
    targets = {k: int(round(TARGET_TOTAL * frac)) for k, frac in QUOTAS.items()}
    # ensure sum ~= TARGET_TOTAL
    diff = TARGET_TOTAL - sum(targets.values())
    targets["closerep_seeing"] = targets.get("closerep_seeing", 0) + diff

    selected: list[dict] = []
    fill_report = {}
    for kind, n_tgt in targets.items():
        rows = list(pools.get(kind) or [])
        rng.shuffle(rows)
        take = rows[:n_tgt]
        fill_report[kind] = {"target": n_tgt, "available": len(rows), "taken": len(take)}
        selected.extend(take)

    # If some buckets underfilled, top up from thinking_wrong then closerep_seeing then holding
    need = TARGET_TOTAL - len(selected)
    if need > 0:
        used = {r["pair_id"] for r in selected}
        for kind in ("thinking_wrong", "closerep_seeing", "holding", "bridge", "seg2", "anti_loop"):
            if need <= 0:
                break
            extra = [r for r in pools.get(kind) or [] if r["pair_id"] not in used]
            rng.shuffle(extra)
            add = extra[:need]
            selected.extend(add)
            used.update(r["pair_id"] for r in add)
            need -= len(add)
            fill_report.setdefault(kind, {})
            fill_report[kind]["topup"] = fill_report[kind].get("topup", 0) + len(add)

    # hard enforce seeing_wrong cap even after topup
    max_seeing = int(round(TARGET_TOTAL * QUOTAS["seeing_wrong"]))
    seeing = [r for r in selected if r["pair_kind"] == "seeing_wrong"]
    others = [r for r in selected if r["pair_kind"] != "seeing_wrong"]
    if len(seeing) > max_seeing:
        rng.shuffle(seeing)
        seeing = seeing[:max_seeing]
        selected = others + seeing
        fill_report["seeing_wrong"]["hard_capped_to"] = max_seeing

    # dedupe
    uniq = {}
    for r in selected:
        uniq[r["pair_id"]] = r
    selected = list(uniq.values())
    rng.shuffle(selected)
    return selected, fill_report, targets


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bad = forbidden_identities()
    rng = random.Random(SEED)
    print(f"forbidden_eval_ids={len(bad)}")

    trajs: list[dict] = []
    shared_path = SHARED_V28 if SHARED_V28.exists() else SHARED_V24
    trajs.extend(json.loads(shared_path.read_text()).get("examples") or [])
    for name in ("pos_bridge_after_seg1.json", "pos_holding_second_put.json", "pos_switch_outcome.json"):
        p = PACK / name
        if p.exists():
            trajs.extend(json.loads(p.read_text()).get("examples") or [])

    pools = build_attr_pools(trajs, bad)
    inherit = build_skill_inherit(bad)
    for k, rows in inherit.items():
        pools.setdefault(k, []).extend(rows)

    mined = {k: len(v) for k, v in pools.items()}
    selected, fill_report, targets = apply_quotas(pools, rng)

    leak = [r["pair_id"] for r in selected if r.get("identity") in bad]
    if leak:
        raise SystemExit(f"TEST_LEAK n={len(leak)} eg={leak[:5]}")
    missing = [r["pair_id"] for r in selected if not Path(r["image"]).is_file()]
    if missing:
        raise SystemExit(f"missing images n={len(missing)} eg={missing[:3]}")
    assert len(selected) >= 800, f"too few pairs: {len(selected)}"

    kinds = Counter(r["pair_kind"] for r in selected)
    # family rolls
    c_skill = kinds.get("holding", 0) + kinds.get("bridge", 0) + kinds.get("seg2", 0)
    frac = {k: round(v / max(1, len(selected)), 4) for k, v in kinds.items()}

    df = pd.DataFrame(selected)
    train_path = OUT_DIR / "dpo_train.parquet"
    df.to_parquet(train_path, index=False)
    meta = {
        "version": "v28_attr_dpo_quota",
        "n_total": len(df),
        "kinds": {k: int(v) for k, v in kinds.items()},
        "fractions": frac,
        "family": {
            "C_skill_holding_bridge_seg2": c_skill,
            "C_skill_frac": round(c_skill / len(df), 4),
            "closerep_seeing": kinds.get("closerep_seeing", 0),
            "closerep_frac": round(kinds.get("closerep_seeing", 0) / len(df), 4),
            "seeing_wrong_cap_frac": round(kinds.get("seeing_wrong", 0) / len(df), 4),
        },
        "quotas_target": targets,
        "fill_report": fill_report,
        "mined_before_quota": mined,
        "action_only": True,
        "test_set_used": False,
        "forbidden_eval_ids": len(bad),
        "sources": [str(shared_path), "skill_pack", "seg2_trainpack", "closerep_anti_illegal"],
        "thresholds": {"Q_LOW": Q_LOW, "Q_HIGH": Q_HIGH},
        "train_parquet": str(train_path),
        "narrative": "attribution defines pairs; quotas favor C thinking + closerep seeing; generic seeing capped",
        "chosen_top": df["chosen_a"].value_counts().head(10).to_dict(),
        "rejected_top": df["rejected_a"].value_counts().head(10).to_dict(),
    }
    (OUT_DIR / "DPO_META.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    preview = [
        {k: r.get(k) for k in ("pair_id", "pair_kind", "chosen_a", "rejected_a", "q_t", "w_t", "e_t", "t", "scene", "tasktype")}
        for r in selected[:50]
    ]
    (OUT_DIR / "dpo_pairs_preview.json").write_text(json.dumps(preview, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("WROTE", train_path)


if __name__ == "__main__":
    main()
