#!/usr/bin/env python3
"""v28 Stage2 pack: keep v24 positive skill22 shared + closerep upsample/window clips.

Hard rules:
  - Never inject eval identities (C105 / n806 / test809).
  - Stage2 CE stays positive-only (no skill_neg pools).
  - Closerep aug = success trajs with closerep tasktypes + open→pickup→close windows.
"""
from __future__ import annotations

import os

import copy
import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(os.environ["RECREDIT_ROOT"])
SKILL22 = BASE / "data/wave23_shared_p123_skill22"
OUT_DIR = BASE / "data/wave23_shared_p123_v28"
DATA_ROOT = Path(os.environ.get("EMBODIED_REASONER_ROOT", "/path/to/embodied_reasoner"))
TEST_C = BASE / "data/eval/test_C_105.json"
TEST_809 = Path(os.environ.get("EMBODIED_REASONER_ROOT", "/path/to/embodied_reasoner")) / "test_809.json"
N806 = BASE / "data/eval/test_809_n806_no_emulator3.json"

# Prefer scarce / high-signal closerep families first.
CLOSEREP_TT = (
    "single_search_from_closerep",
    "single_pickup_from_closerep",
    "pickup_from_closerep_and_put",
    "pickup_from_closerep_and_put_in_closerep",
    "pickup_and_put_in_closerep",
    "single_search_from_closerep_open_fault",
    "pickup_from_closerep_and_put_close_fault",
    "pickup_from_closerep_and_put_in_closerep_close_fault",
    "pickup_from_closerep_and_put_in_closerep_open_fault",
)
UPSAMPLE = {
    "single_search_from_closerep": 4,
    "single_pickup_from_closerep": 3,
    "pickup_from_closerep_and_put": 2,
    "pickup_from_closerep_and_put_in_closerep": 2,
    "pickup_and_put_in_closerep": 1,
}
WINDOW_PAD = 1  # steps around open/pickup/close core
MAX_ANTI_ILLEGAL = 200


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    x = json.loads(path.read_text())
    items = x if isinstance(x, list) else (x.get("examples") or x.get("data") or [])
    return {str(e.get("identity")) for e in items if e.get("identity") is not None}


def forbidden() -> set[str]:
    return load_ids(TEST_C) | load_ids(TEST_809) | load_ids(N806)


def identity_of(ex: dict) -> str | None:
    if ex.get("identity") is not None:
        return str(ex["identity"])
    m = re.match(r"^(\d+)_", str(ex.get("id") or ""))
    return m.group(1) if m else None


def is_forbidden(ex: dict, bad: set[str]) -> bool:
    ident = identity_of(ex)
    return bool(ident and ident in bad)


def raw_of(step: dict) -> str:
    a = step.get("action")
    if isinstance(a, dict):
        return (a.get("raw") or "").strip()
    return str(a or "").strip()


def is_closerep_tt(tt: str) -> bool:
    t = (tt or "").lower()
    return "closerep" in t or t in {x.lower() for x in CLOSEREP_TT}


def window_clip(ex: dict) -> dict | None:
    """Clip around first open→…→pickup (or open→close) span for denser closerep CE."""
    steps = list(ex.get("steps") or [])
    if len(steps) < 3:
        return None
    open_i = next((i for i, s in enumerate(steps) if raw_of(s).lower().startswith("open ")), None)
    if open_i is None:
        return None
    pickup_i = next(
        (i for i, s in enumerate(steps) if i > open_i and raw_of(s).lower().startswith("pickup ")),
        None,
    )
    close_i = next(
        (i for i, s in enumerate(steps) if i > open_i and raw_of(s).lower().startswith("close ")),
        None,
    )
    end_i = max(x for x in (pickup_i, close_i, open_i + 2) if x is not None)
    lo = max(0, open_i - WINDOW_PAD)
    hi = min(len(steps), end_i + WINDOW_PAD + 1)
    if hi - lo < 3:
        return None
    out = copy.deepcopy(ex)
    out["steps"] = steps[lo:hi]
    # reindex t if present
    for j, s in enumerate(out["steps"]):
        if "t" in s:
            s["t"] = j
    out["id"] = f"{ex.get('id')}__v28_closerep_win_{lo}_{hi}"
    tags = dict(out.get("skill_tags") or {})
    tags["v28_from"] = "closerep_window"
    tags["window"] = [lo, hi]
    out["skill_tags"] = tags
    out["thor_train_pool"] = "v28_closerep_window"
    out["skill22_from"] = "pos_closerep_window"
    return out


def upsample_copies(ex: dict, k: int, tag: str) -> list[dict]:
    if k <= 1:
        e = copy.deepcopy(ex)
        tags = dict(e.get("skill_tags") or {})
        tags["v28_from"] = tag
        e["skill_tags"] = tags
        e["thor_train_pool"] = e.get("thor_train_pool") or "v28_closerep_full"
        return [e]
    rows = []
    for i in range(k):
        e = copy.deepcopy(ex)
        e["id"] = f"{ex.get('id')}__v28_up{i}"
        tags = dict(e.get("skill_tags") or {})
        tags["v28_from"] = tag
        tags["upsample_i"] = i
        e["skill_tags"] = tags
        e["thor_train_pool"] = "v28_closerep_upsample"
        e["skill22_from"] = "pos_closerep_upsample"
        rows.append(e)
    return rows


def anti_illegal_open_loop(ex: dict, bad: set[str]) -> dict | None:
    """Short rejected-style clip: duplicate open of same receptacle (for CE soft or DPO mining).

    Kept as R_L2=0 annotated failure fragment; convert uses neg_scale=0 so it will be dropped
    from CE weights if negative — we still keep only as optional DPO source file, not CE.
    """
    if is_forbidden(ex, bad):
        return None
    steps = list(ex.get("steps") or [])
    for i, s in enumerate(steps[:-1]):
        a = raw_of(s)
        if not a.lower().startswith("open "):
            continue
        # synthesize a one-step illegal repeat after this open, using same image as next if possible
        nxt = steps[min(i + 1, len(steps) - 1)]
        syn = copy.deepcopy(ex)
        core = [copy.deepcopy(steps[i]), copy.deepcopy(nxt), copy.deepcopy(nxt)]
        core[1]["action"] = {"raw": a} if isinstance(nxt.get("action"), dict) else a
        if isinstance(core[1].get("action"), dict):
            core[1]["action"]["raw"] = a
        else:
            core[1]["action"] = a
        # mark as failure fragment
        for j, st in enumerate(core):
            st["t"] = j
            st["q_t"] = float(st.get("q_t") or st.get("q") or 0.2)
            st["w_t"] = float(st.get("w_t") or 0.05)
            st["e_t"] = "seeing-wrong"
        syn["steps"] = core
        syn["R_L2"] = 0
        syn["id"] = f"{ex.get('id')}__v28_anti_illegal_open"
        syn["thor_train_pool"] = "v28_closerep_anti_illegal"
        syn["skill22_from"] = "neg_closerep_anti_illegal"
        tags = dict(syn.get("skill_tags") or {})
        tags["v28_from"] = "anti_illegal_open_loop"
        syn["skill_tags"] = tags
        return syn
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bad = forbidden()
    base_train = json.loads((SKILL22 / "shared_train.json").read_text())
    base_val = json.loads((SKILL22 / "shared_val.json").read_text())
    train = list(base_train.get("examples") or [])
    val = list(base_val.get("examples") or [])

    # Drop any accidental neg pools if present
    train = [
        e
        for e in train
        if "skill_neg_" not in str(e.get("thor_train_pool") or "")
        and not str(e.get("skill22_from") or "").startswith("neg_")
        and not is_forbidden(e, bad)
    ]
    val = [e for e in val if not is_forbidden(e, bad)]

    existing_ids = {e.get("id") for e in train}
    closerep_src = [e for e in train if is_closerep_tt(str(e.get("tasktype") or ""))]
    # Also pull extras from s2fix if available
    extra_path = BASE / "data/wave23_shared_p123_s2fix/shared_train.json"
    extra_added = 0
    if extra_path.exists():
        for e in json.loads(extra_path.read_text()).get("examples") or []:
            if is_forbidden(e, bad):
                continue
            if not is_closerep_tt(str(e.get("tasktype") or "")):
                continue
            if int(e.get("R_L2") or 0) != 1:
                continue
            if e.get("id") in existing_ids:
                continue
            e2 = copy.deepcopy(e)
            e2["thor_train_pool"] = "v28_closerep_extra_s2fix"
            e2["skill22_from"] = "pos_closerep_extra"
            tags = dict(e2.get("skill_tags") or {})
            tags["v28_from"] = "closerep_extra_s2fix"
            e2["skill_tags"] = tags
            train.append(e2)
            existing_ids.add(e2.get("id"))
            closerep_src.append(e2)
            extra_added += 1

    aug: list[dict] = []
    tt_counts = Counter()
    for e in closerep_src:
        if int(e.get("R_L2") or 0) != 1:
            continue
        tt = str(e.get("tasktype") or "")
        tt_counts[tt] += 1
        k = UPSAMPLE.get(tt, 1)
        aug.extend(upsample_copies(e, k, "closerep_upsample"))
        win = window_clip(e)
        if win is not None:
            aug.append(win)

    # Anti-illegal fragments → separate file for DPO only (not merged into CE train)
    anti: list[dict] = []
    for e in closerep_src:
        if len(anti) >= MAX_ANTI_ILLEGAL:
            break
        syn = anti_illegal_open_loop(e, bad)
        if syn is not None:
            anti.append(syn)

    # Merge CE train: base + aug (dedupe by id)
    merged = {e.get("id"): e for e in train}
    for e in aug:
        merged[e.get("id")] = e
    train_out = list(merged.values())

    out_train = {
        "schema_version": "recredit_r1.v28_stage2",
        "examples": train_out,
    }
    out_val = {
        "schema_version": "recredit_r1.v28_stage2",
        "examples": val,
    }
    (OUT_DIR / "shared_train.json").write_text(json.dumps(out_train, ensure_ascii=False))
    (OUT_DIR / "shared_val.json").write_text(json.dumps(out_val, ensure_ascii=False))
    (OUT_DIR / "closerep_anti_illegal.json").write_text(
        json.dumps({"schema_version": "recredit_r1.v28_anti_illegal", "examples": anti}, ensure_ascii=False)
    )

    pools = Counter(str(e.get("thor_train_pool") or "") for e in train_out)
    v28_from = Counter(str((e.get("skill_tags") or {}).get("v28_from") or "") for e in train_out)
    meta = {
        "version": "v28_stage2_closerep",
        "n_train": len(train_out),
        "n_val": len(val),
        "n_base_pos": len(train),
        "n_aug_rows": len(aug),
        "n_closerep_src": len(closerep_src),
        "extra_s2fix_added": extra_added,
        "n_anti_illegal_dpo_only": len(anti),
        "closerep_tt_counts": dict(tt_counts),
        "pools": pools.most_common(20),
        "v28_from": v28_from.most_common(20),
        "forbidden_eval_ids": len(bad),
        "upsample": UPSAMPLE,
        "narrative": "v24 pos skill22 + closerep upsample/window; anti-illegal kept DPO-only",
    }
    (OUT_DIR / "PACK_META.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("WROTE", OUT_DIR)


if __name__ == "__main__":
    main()
