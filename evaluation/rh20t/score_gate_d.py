#!/usr/bin/env python3
"""Merge independent Gate D pass B with parser A; agreement + Cohen's kappa.

Annotator B labels were written from the blind packet (full DecisionMaking /
raw text only) using the frozen written protocol. This is not a second
parser execution.
"""
from __future__ import annotations

import os
import json
from collections import Counter
from pathlib import Path

MET = Path(os.environ["RECREDIT_ROOT"]) / "results" / "rh20t_metrics" / "gate_d"
REQ = {"pick", "place"}

# Independent B: index 1..56 matching blind_packet.txt.
# Fields: pred_verb, pred_object, format_valid, format_error, note
B = {
    1: ("move", "object", False, "invalid_required_object_on_object_free", "move away from the object"),
    2: ("pick", None, False, "missing_required_object", "grasp"),
    3: ("pick", None, False, "missing_required_object", "grasp"),
    4: ("pick", None, False, "missing_required_object", "grasp"),
    5: ("place", None, False, "missing_required_object", "place"),
    6: ("place", None, False, "missing_required_object", "place"),
    7: ("pick", None, False, "missing_required_object", "grasp"),
    8: ("pick", None, False, "missing_required_object", "grasp"),
    9: ("place", None, False, "missing_required_object", "place"),
    10: ("pick", None, False, "multiple_actions", "last of 2 tags is grasp"),
    11: ("pick", None, False, "multiple_actions", "last of 3 tags is grasp"),
    12: ("end", None, False, "multiple_actions", "two broken end tags; fallback pred=end"),
    13: ("end", None, False, "multiple_actions", "three broken end tags; fallback pred=end"),
    14: ("move", None, False, "multiple_actions", "last well-formed tag is move"),
    15: ("move", None, False, "multiple_actions", "brown_cube is outside the tag"),
    16: ("move", None, False, "multiple_actions", "quoted tag plus real move tag"),
    17: (None, None, False, "unknown_verb", "close"),
    18: (None, None, False, "unknown_verb", "search"),
    19: (None, None, False, "unknown_verb", "no DM; cached pred=hmm"),
    20: (None, None, False, "unknown_verb", "move_to_target is one unknown token"),
    21: (None, None, False, "unknown_verb", "DM is Your Action; prose move ignored"),
    22: (None, None, False, "unknown_verb", "move_to_target"),
    23: (None, None, False, "unknown_verb", "no DM; cached pred=okay"),
    24: (None, None, False, "unknown_verb", "move_back is one unknown token"),
    25: ("end", None, True, None, "end"),
    26: ("move", None, True, None, "move"),
    27: ("end", None, True, None, "end"),
    28: ("move", None, True, None, "move"),
    29: ("end", None, True, None, "end"),
    30: ("end", None, True, None, "end"),
    31: ("end", None, True, None, "end"),
    32: ("move", None, True, None, "move"),
    33: ("move", None, True, None, "move"),
    34: ("move", None, True, None, "move"),
    35: ("move", None, True, None, "move"),
    36: ("move", None, True, None, "move"),
    37: ("move", None, True, None, "move"),
    38: ("move", None, True, None, "move"),
    39: ("move", None, True, None, "move"),
    40: ("end", None, True, None, "end"),
    41: ("move", None, True, None, "move"),
    42: ("end", None, True, None, "end"),
    43: ("end", None, True, None, "end"),
    44: ("end", None, True, None, "end"),
    45: ("move", None, True, None, "move"),
    46: ("move", None, True, None, "move"),
    47: ("move", None, True, None, "move"),
    48: ("end", None, True, None, "end"),
    49: ("move", None, True, None, "move forward; forward is not an object alias"),
    50: ("end", None, True, None, "end"),
    51: ("end", None, True, None, "end"),
    52: ("end", None, True, None, "end"),
    53: ("end", None, True, None, "end"),
    54: ("move", None, True, None, "move"),
    55: ("end", None, True, None, "end"),
    56: ("end", None, True, None, "end"),
}


def cohen_kappa(y1: list, y2: list) -> tuple[float | None, str | None]:
    n = len(y1)
    if n == 0:
        return None, "empty"
    labs = sorted(set(y1) | set(y2), key=lambda x: str(x))
    po = sum(a == b for a, b in zip(y1, y2)) / n
    pe = 0.0
    c1, c2 = Counter(y1), Counter(y2)
    for lab in labs:
        pe += (c1[lab] / n) * (c2[lab] / n)
    if len(set(y1)) < 2 or len(set(y2)) < 2:
        return None, "undefined_no_class_variation"
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0, None
    return (po - pe) / (1.0 - pe), None


def main() -> None:
    packet = json.loads((MET / "blind_packet_with_auto.json").read_text())
    assert len(packet) == 56
    rows = []
    disagreements = []
    for i, item in enumerate(packet, 1):
        bv, bo, bf, bfe, note = B[i]
        auto = item["_auto"]
        gt_v = item["gt_verb"]
        gt_o = item["gt_object"]
        req = gt_v in REQ
        b_verb_ok = bv == gt_v
        b_object_ok = (bo == gt_o) if req else (bo is None)
        a_verb = auto["pred_verb"]
        a_obj = auto["pred_object"]
        a_fmt = bool(auto["format_valid"]) if auto["format_valid"] is not None else None
        a_verb_ok = bool(auto["verb_correct"])
        a_obj_ok = auto["object_correct"]
        auto_match_ok = (
            a_verb == bv
            and a_obj == bo
            and a_fmt == bf
            and a_verb_ok == b_verb_ok
            and (True if a_obj_ok is None else a_obj_ok == b_object_ok)
        )
        row = {
            "idx": i,
            "stratum": item["stratum"],
            "variant": item["variant"],
            "episode_id": item["episode_id"],
            "step_id": item["step_id"],
            "gt_verb": gt_v,
            "gt_object": gt_o,
            "last_decision": item.get("last_decision"),
            "annotator_a_parser": {
                "pred_verb": a_verb,
                "pred_object": a_obj,
                "format_valid": a_fmt,
                "format_error": auto.get("format_error"),
                "verb_correct": a_verb_ok,
                "object_correct": a_obj_ok,
            },
            "annotator_b_independent": {
                "pred_verb": bv,
                "pred_object": bo,
                "format_valid": bf,
                "format_error": bfe,
                "verb_correct": b_verb_ok,
                "object_correct": b_object_ok if req else None,
                "note": note,
            },
            "auto_match_ok": auto_match_ok,
        }
        rows.append(row)
        if not auto_match_ok:
            disagreements.append(row)

    def col(path_a, path_b, pred=lambda r: True):
        a, b = [], []
        for r in rows:
            if not pred(r):
                continue
            xa, xb = r
            a.append(path_a(r))
            b.append(path_b(r))
        return a, b

    pairs = {
        "pred_verb": (
            [r["annotator_a_parser"]["pred_verb"] for r in rows],
            [r["annotator_b_independent"]["pred_verb"] for r in rows],
        ),
        "pred_object": (
            [r["annotator_a_parser"]["pred_object"] for r in rows],
            [r["annotator_b_independent"]["pred_object"] for r in rows],
        ),
        "format_valid": (
            [r["annotator_a_parser"]["format_valid"] for r in rows],
            [r["annotator_b_independent"]["format_valid"] for r in rows],
        ),
        "verb_correct": (
            [r["annotator_a_parser"]["verb_correct"] for r in rows],
            [r["annotator_b_independent"]["verb_correct"] for r in rows],
        ),
        "object_correct_required_only": (
            [r["annotator_a_parser"]["object_correct"] for r in rows if r["gt_verb"] in REQ],
            [r["annotator_b_independent"]["object_correct"] for r in rows if r["gt_verb"] in REQ],
        ),
    }
    stats = {}
    for name, (a, b) in pairs.items():
        n = len(a)
        agree = sum(x == y for x, y in zip(a, b))
        kap, kap_note = cohen_kappa(a, b)
        stats[name] = {
            "n": n,
            "n_agree": agree,
            "pct_agree": round(100.0 * agree / n, 2) if n else None,
            "cohen_kappa": None if kap is None else round(kap, 4),
            "kappa_note": kap_note,
            "a_label_counts": dict(Counter(map(lambda x: str(x), a))),
            "b_label_counts": dict(Counter(map(lambda x: str(x), b))),
        }

    # Primary Gate D kappa: four binary human-vs-parser judgments
    report = {
        "n": 56,
        "seed": 20260913,
        "annotator_a": "frozen automatic parser rh20t_action_metrics_v1_1",
        "annotator_b": (
            "developer independent pass on the blind packet: last well-formed "
            "DecisionMaking, frozen aliases, format schema. Not a second unaffiliated human."
        ),
        "status": "gate_d_independent_b_done_awaiting_unaffiliated_human_if_required",
        "agreement": stats,
        "n_disagreements": len(disagreements),
        "disagreements": [
            {
                "idx": d["idx"],
                "stratum": d["stratum"],
                "episode_id": d["episode_id"],
                "step_id": d["step_id"],
                "a": d["annotator_a_parser"],
                "b": d["annotator_b_independent"],
            }
            for d in disagreements
        ],
        "resolution_rule": (
            "Disagreements are resolved by the written protocol, not by model. "
            "Resolved label = annotator B when B followed last well-formed "
            "DecisionMaking + frozen maps; otherwise keep A and document."
        ),
        "resolved_keep_parser": True,
        "note_on_kappa": (
            "High kappa is expected if the written protocol is unambiguous. "
            "It validates the implementation, not seeing/thinking q_t."
        ),
    }
    dest_rows = MET / "audit_sample_100.jsonl"
    dest_rows.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (MET / "GATE_D_AGREEMENT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (MET / "STATUS.json").write_text(json.dumps({
        "n": 56,
        "status": report["status"],
        "n_disagreements": len(disagreements),
        "pct_agree_pred_verb": stats["pred_verb"]["pct_agree"],
        "kappa_pred_verb": stats["pred_verb"]["cohen_kappa"],
        "pct_agree_format": stats["format_valid"]["pct_agree"],
        "kappa_format": stats["format_valid"]["cohen_kappa"],
        "pct_agree_verb_correct": stats["verb_correct"]["pct_agree"],
        "kappa_verb_correct": stats["verb_correct"]["cohen_kappa"],
        "pct_agree_object_required": stats["object_correct_required_only"]["pct_agree"],
        "kappa_object_required": stats["object_correct_required_only"]["cohen_kappa"],
        "note": report["annotator_b"],
    }, indent=2, ensure_ascii=False))
    print(json.dumps(report["agreement"], indent=2))
    print("disagreements", len(disagreements))


if __name__ == "__main__":
    main()
