#!/usr/bin/env python3
"""Audit whether object=0 / cache-only are parse misses, not model misses."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_action_metrics import (  # noqa: E402
    DECISION_RE,
    load_json,
    normalize_object,
    parse_prediction,
)

PAPER = HERE.parents[1]
PRED = PAPER / "results/rh20t_cfg3_pc_n100"
MET = PAPER / "results/rh20t_metrics"
OUT = MET / "audits"
OBJ_ALIAS = load_json(HERE / "object_aliases.json")["alias_to_canonical"]
VERB_ALIAS = load_json(HERE / "verb_aliases.json")["alias_to_canonical"]
REQ = {"pick", "place"}
FREE = {"move", "end"}
DM_LOOSE = re.compile(r"<DecisionMaking>(.*?)</?[^>]{0,40}>", re.I | re.S)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def objects_in_text(text: str) -> list[str]:
    found = []
    seen = set()
    low = (text or "").lower()
    for key in sorted(OBJ_ALIAS, key=len, reverse=True):
        if key in low and OBJ_ALIAS[key] not in seen:
            seen.add(OBJ_ALIAS[key])
            found.append(OBJ_ALIAS[key])
    return found


def q2_dpo_cache_and_hits() -> dict:
    rows = load_jsonl(MET / "q2_dpo_ol/per_step_normalized.jsonl")
    cache = [r for r in rows if r.get("recovery") == "cache_only"]
    obj_ok = [r for r in rows if r.get("object_correct") is True]
    return {
        "n_cache_only": len(cache),
        "cache_only": [
            {
                "episode_id": r["episode_id"],
                "step_id": r["step_id"],
                "gt_verb": r["gt_verb"],
                "pred_verb": r["pred_verb"],
                "pred_object": r["pred_object"],
                "raw_prediction": r["raw_prediction"],
                "response_head": (r.get("full_response") or "")[:200],
                "format_error": r.get("format_error"),
                "object_correct": r.get("object_correct"),
                "is_key_action": r.get("is_key_action"),
            }
            for r in cache
        ],
        "object_correct_rows": [
            {
                "episode_id": r["episode_id"],
                "step_id": r["step_id"],
                "gt_verb": r["gt_verb"],
                "pred_verb": r["pred_verb"],
                "gt_object": r["gt_object"],
                "pred_object": r["pred_object"],
                "recovery": r.get("recovery"),
                "decision": (DECISION_RE.findall(r.get("full_response") or "") or [None])[-1],
                "joint_correct": r.get("joint_correct"),
            }
            for r in obj_ok
        ],
    }


def reconstruct_key_matches() -> dict:
    from evaluate_action_metrics import lcs_match

    rows = load_jsonl(MET / "q2_dpo_ol/per_step_normalized.jsonl")
    by_ep: dict[str, list] = {}
    for r in rows:
        by_ep.setdefault(r["episode_id"], []).append(r)
    hits = []
    total = 0
    for eid, steps in by_ep.items():
        steps = sorted(steps, key=lambda x: x["step_id"])
        pred_pairs = [(x["pred_verb"], x["pred_object"]) for x in steps]
        gt_keys = [(x["gt_verb"], x["gt_object"]) for x in steps if x.get("is_key_action")]
        n = lcs_match(pred_pairs, gt_keys)
        total += n
        if n:
            hits.append({
                "episode_id": eid,
                "n_matched": n,
                "gt_keys": gt_keys,
                "pred_pairs": [(a, b) for a, b in pred_pairs if a in REQ],
            })
    return {"micro_matched": total, "episodes_with_match": hits}


def audit_object_required(variant: str, sample_n: int = 40) -> dict:
    rows = load_jsonl(MET / variant / "per_step_normalized.jsonl")
    req = [r for r in rows if r.get("object_required")]
    missed_in_dm = []
    dm_has_obj_parser_none = []
    no_dm = []
    parser_vs_loose = []
    for r in req:
        full = r.get("full_response") or ""
        head = ""
        pred_path = PRED / variant / r["episode_id"] / "result.json"
        if pred_path.exists():
            doc = json.loads(pred_path.read_text())
            for s in doc.get("steps") or []:
                if int(s["t"]) == int(r["step_id"]):
                    head = s.get("response_head") or ""
                    if not full:
                        full = s.get("full_response") or ""
                    break
        dms = DECISION_RE.findall(full) or DECISION_RE.findall(head)
        loose = DM_LOOSE.findall(full) or DM_LOOSE.findall(head)
        last = (dms[-1].strip() if dms else "")
        if not dms:
            no_dm.append({
                "episode_id": r["episode_id"],
                "step_id": r["step_id"],
                "pred_verb": r.get("pred_verb"),
                "raw_prediction": r.get("raw_prediction"),
                "head_or_full_tail": (full or head)[-180:],
                "recovery": r.get("recovery"),
            })
        objs_dm = objects_in_text(last)
        if objs_dm and r.get("pred_object") is None:
            dm_has_obj_parser_none.append({
                "episode_id": r["episode_id"],
                "step_id": r["step_id"],
                "decision": last,
                "alias_hits_in_decision": objs_dm,
                "pred_verb": r.get("pred_verb"),
                "gt_object": r.get("gt_object"),
            })
        if r.get("gt_object") and r.get("gt_object") in objects_in_text(last) and not r.get("object_correct"):
            missed_in_dm.append({
                "episode_id": r["episode_id"],
                "step_id": r["step_id"],
                "decision": last,
                "gt_object": r["gt_object"],
                "pred_object": r.get("pred_object"),
            })
        if loose and not dms:
            parser_vs_loose.append({
                "episode_id": r["episode_id"],
                "step_id": r["step_id"],
                "loose_last": loose[-1][:160],
                "pred_verb": r.get("pred_verb"),
            })

    # stratified sample of object-required steps for manual dump
    rng_idx = list(range(len(req)))
    rng_idx.sort(key=lambda i: (req[i]["episode_id"], req[i]["step_id"]))
    # take first 8 per pred_verb + 8 format-ok if any
    by = {}
    for r in req:
        by.setdefault(r.get("pred_verb") or "none", []).append(r)
    sample = []
    for k, items in sorted(by.items()):
        sample.extend(items[: max(1, sample_n // max(1, len(by)))])
    sample = sample[:sample_n]
    sample_out = []
    for r in sample:
        full = r.get("full_response") or ""
        dms = DECISION_RE.findall(full)
        sample_out.append({
            "episode_id": r["episode_id"],
            "step_id": r["step_id"],
            "gt_verb": r["gt_verb"],
            "gt_object": r["gt_object"],
            "pred_verb": r["pred_verb"],
            "pred_object": r["pred_object"],
            "object_correct": r["object_correct"],
            "format_error": r.get("format_error"),
            "recovery": r.get("recovery"),
            "n_decision_tags": len(dms),
            "last_decision": (dms[-1].strip() if dms else None),
            "alias_in_last_decision": objects_in_text(dms[-1]) if dms else [],
        })

    return {
        "variant": variant,
        "n_object_required": len(req),
        "n_object_correct": sum(1 for r in req if r.get("object_correct")),
        "n_no_strict_decision_tag": len(no_dm),
        "n_decision_has_alias_but_parser_none": len(dm_has_obj_parser_none),
        "n_gt_object_in_decision_but_scored_false": len(missed_in_dm),
        "n_loose_tag_only": len(parser_vs_loose),
        "examples_parser_miss": dm_has_obj_parser_none[:20],
        "examples_gt_in_dm_missed": missed_in_dm[:20],
        "examples_no_dm": no_dm[:15],
        "examples_loose_only": parser_vs_loose[:15],
        "manual_sample": sample_out,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q2 = {
        "cache_and_object_hits": q2_dpo_cache_and_hits(),
        "key_lcs": reconstruct_key_matches(),
    }
    (OUT / "q2_dpo_rare_positives.json").write_text(json.dumps(q2, indent=2, ensure_ascii=False))
    q3 = audit_object_required("q3_full_ol")
    (OUT / "q3_full_object_zero_audit.json").write_text(json.dumps(q3, indent=2, ensure_ascii=False))
    allv = {v: audit_object_required(v, sample_n=8) for v in (
        "q2_stage1_ol", "q2_full_ol", "q2_dpo_ol", "q3_stage1_ol", "q3_full_ol", "q3_dpo_ol"
    )}
    (OUT / "six_variant_object_parse_audit.json").write_text(json.dumps({
        k: {kk: vv for kk, vv in v.items() if kk != "manual_sample"}
        for k, v in allv.items()
    }, indent=2, ensure_ascii=False))
    print(json.dumps({
        "q2_dpo_cache_only": q2["cache_and_object_hits"]["n_cache_only"],
        "q2_dpo_object_hits": len(q2["cache_and_object_hits"]["object_correct_rows"]),
        "q2_dpo_key_micro": q2["key_lcs"]["micro_matched"],
        "q3_full": {
            "object_correct": q3["n_object_correct"],
            "no_dm": q3["n_no_strict_decision_tag"],
            "parser_miss": q3["n_decision_has_alias_but_parser_none"],
            "gt_in_dm_missed": q3["n_gt_object_in_decision_but_scored_false"],
            "loose_only": q3["n_loose_tag_only"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
