#!/usr/bin/env python3
"""RH20T offline action-matching metrics (rh20t_action_metrics_v1).

Scores saved openloop_multiturn_teacher_force_v2 predictions against the
100-episode external test pack. This is not closed-loop robot success.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_SCHEMA = HERE / "action_schema.json"
DEFAULT_VERB = HERE / "verb_aliases.json"
DEFAULT_OBJECT = HERE / "object_aliases.json"

from recover_from_vlm_logs import classify_request_error  # noqa: E402

DECISION_RE = re.compile(r"<DecisionMaking>(.*?)</DecisionMaking>", re.I | re.S)
TOKEN_RE = re.compile(r"[a-z0-9_]+")
MULTI_DECISION_RE = re.compile(r"<DecisionMaking>", re.I)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalize_verb(raw: str | None, alias: dict[str, str]) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if not s:
        return None
    return alias.get(s)


def normalize_object(raw: str | None, alias: dict[str, str]) -> str | None:
    if raw is None:
        return None
    s = " ".join(str(raw).strip().lower().replace("-", " ").replace("_", " ").split())
    if not s:
        return None
    if s in alias:
        return alias[s]
    # longest alias match on the whole string or last token
    for key in sorted(alias, key=len, reverse=True):
        if key in s:
            return alias[key]
    return None


def _object_from_tokens(tokens: list[str], obj_alias: dict[str, str]) -> str | None:
    if not tokens:
        return None
    joined = " ".join(tokens)
    return normalize_object(joined, obj_alias)


def parse_prediction(
    pred: str | None,
    response_head: str | None,
    verb_alias: dict[str, str],
    obj_alias: dict[str, str],
    object_required_verbs: set[str],
    object_free_verbs: set[str],
    full_response: str | None = None,
) -> dict[str, Any]:
    raw_pred = "" if pred is None else str(pred)
    raw_head = "" if response_head is None else str(response_head)
    raw_full = "" if full_response is None else str(full_response)
    parse_text = raw_full or raw_head
    errors: list[str] = []

    text = raw_pred.strip()
    n_tags = len(MULTI_DECISION_RE.findall(parse_text))
    if n_tags > 1:
        errors.append("multiple_actions")
    dm = DECISION_RE.findall(parse_text)
    if dm:
        text = dm[-1].strip() or text
        if len(dm) > 1:
            if "multiple_actions" not in errors:
                errors.append("multiple_actions")

    if not text:
        return {
            "pred_verb": None,
            "pred_object": None,
            "pred_pose": None,
            "format_valid": False,
            "format_error": "empty",
            "format_errors": ["empty"],
        }

    tokens = TOKEN_RE.findall(text.lower().replace("-", "_"))
    if not tokens:
        return {
            "pred_verb": None,
            "pred_object": None,
            "pred_pose": None,
            "format_valid": False,
            "format_error": "unparseable",
            "format_errors": ["unparseable"],
        }

    verb = normalize_verb(tokens[0], verb_alias)
    extra = tokens[1:]
    obj = _object_from_tokens(extra, obj_alias)
    # If the first token is unknown, try the whole string as a verb.
    if verb is None and len(tokens) == 1:
        verb = normalize_verb(tokens[0], verb_alias)

    if verb is None:
        errors.append("unknown_verb")
    elif verb in object_required_verbs and obj is None:
        errors.append("missing_required_object")
    elif verb in object_free_verbs and obj is not None:
        errors.append("invalid_required_object_on_object_free")

    # Numeric / pose checks: this protocol does not emit poses.
    if re.search(r"\bnan\b", text, re.I):
        errors.append("nan_pose")

    format_valid = len(errors) == 0
    return {
        "pred_verb": verb,
        "pred_object": obj,
        "pred_pose": None,
        "format_valid": format_valid,
        "format_error": None if format_valid else errors[0],
        "format_errors": errors,
    }


def gt_object_for_episode(instruction: str, instr_map: dict[str, str], obj_alias: dict[str, str]) -> str:
    if instruction in instr_map:
        return instr_map[instruction]
    obj = normalize_object(instruction, obj_alias)
    if obj is None:
        raise KeyError(f"unmapped instruction: {instruction!r}")
    return obj


def lcs_match(pred_pairs: list[tuple[str | None, str | None]], gt_pairs: list[tuple[str, str]]) -> int:
    """One-to-one monotonic LCS over (verb, object) pairs."""
    n, m = len(pred_pairs), len(gt_pairs)
    if m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_pairs[i - 1] == gt_pairs[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def geodesic_angle_deg(q1: list[float], q2: list[float]) -> float:
    """Quaternion geodesic angle in degrees. q and -q are equivalent."""
    a = [float(x) for x in q1]
    b = [float(x) for x in q2]
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    a = [x / na for x in a]
    b = [x / nb for x in b]
    dot = max(-1.0, min(1.0, abs(sum(x * y for x, y in zip(a, b)))))
    return math.degrees(2.0 * math.acos(dot))


def _pct(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return 100.0 * num / den


def _ci(samples: list[float]) -> list[float] | None:
    if not samples:
        return None
    xs = sorted(samples)
    lo = xs[int(0.025 * (len(xs) - 1))]
    hi = xs[int(0.975 * (len(xs) - 1))]
    return [lo, hi]


def bootstrap_episode_metrics(
    per_ep: list[dict[str, Any]],
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(per_ep)
    keys = [
        "verb_acc",
        "object_acc",
        "joint_acc",
        "key_action_recall",
        "format_failure_rate",
    ]
    buckets = {k: [] for k in keys}
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        v_ok = v_n = o_ok = o_n = j_ok = j_n = k_ok = k_n = f_bad = f_n = 0
        recalls = []
        for i in idx:
            e = per_ep[i]
            v_ok += e["n_verb_correct"]
            v_n += e["n_steps"]
            o_ok += e["n_object_correct"]
            o_n += e["n_object_required"]
            j_ok += e["n_joint_correct"]
            j_n += e["n_steps"]
            k_ok += e["n_key_matched"]
            k_n += e["n_key"]
            f_bad += e["n_format_fail"]
            f_n += e["n_steps"]
            if e["n_key"] > 0:
                recalls.append(e["n_key_matched"] / e["n_key"])
        buckets["verb_acc"].append(100.0 * v_ok / v_n if v_n else 0.0)
        buckets["object_acc"].append(100.0 * o_ok / o_n if o_n else 0.0)
        buckets["joint_acc"].append(100.0 * j_ok / j_n if j_n else 0.0)
        buckets["key_action_recall"].append(100.0 * (sum(recalls) / len(recalls) if recalls else 0.0))
        buckets["format_failure_rate"].append(100.0 * f_bad / f_n if f_n else 0.0)
    return {k: {"ci95": _ci(v)} for k, v in buckets.items()}


def load_recovered(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if path is None or not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[(row["episode_id"], int(row["step_id"]))] = row
    return out


def score_variant(
    pred_root: Path,
    gt_pack: dict[str, Any],
    schema: dict[str, Any],
    verb_cfg: dict[str, Any],
    obj_cfg: dict[str, Any],
    bootstrap_seed: int,
    bootstrap_samples: int,
    recovered: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verb_alias = {k.lower(): v for k, v in verb_cfg["alias_to_canonical"].items()}
    obj_alias = {k.lower(): v for k, v in obj_cfg["alias_to_canonical"].items()}
    instr_map = obj_cfg["instruction_to_object"]
    object_required = set(schema["object_required_verbs"])
    object_free = set(schema["object_free_verbs"])

    gt_by_id = {ex["id"]: ex for ex in gt_pack["examples"]}
    rows: list[dict[str, Any]] = []
    per_ep: list[dict[str, Any]] = []
    missing_eps: list[str] = []
    verb_class = defaultdict(lambda: {"ok": 0, "n": 0})
    format_cat = Counter()

    for ex in gt_pack["examples"]:
        eid = ex["id"]
        pred_path = pred_root / eid / "result.json"
        if not pred_path.exists():
            missing_eps.append(eid)
            continue
        pred_doc = load_json(pred_path)
        pred_steps = {int(s["t"]): s for s in pred_doc.get("steps") or []}
        instruction = ex["instruction"]
        gt_obj = gt_object_for_episode(instruction, instr_map, obj_alias)

        ep = {
            "episode_id": eid,
            "n_steps": 0,
            "n_verb_correct": 0,
            "n_object_required": 0,
            "n_object_correct": 0,
            "n_joint_correct": 0,
            "n_key": 0,
            "n_key_matched": 0,
            "n_format_fail": 0,
            "n_object_free": 0,
            "n_request_fail": 0,
            "n_joint_object_required": 0,
            "n_joint_object_free": 0,
        }
        pred_pairs: list[tuple[str | None, str | None]] = []
        gt_key_pairs: list[tuple[str, str]] = []

        for step in ex["steps"]:
            t = int(step["t"])
            ps = pred_steps.get(t)
            raw_gt = (step.get("action") or {}).get("raw") or (step.get("action") or {}).get("verb")
            gt_verb = normalize_verb(raw_gt, verb_alias)
            if gt_verb is None:
                raise ValueError(f"{eid} t={t} unknown GT verb {raw_gt!r}")
            is_key = False
            raw_pred = None
            raw_head = None
            raw_err = None
            if ps is not None:
                is_key = bool(ps.get("is_key"))
                raw_pred = ps.get("pred")
                raw_head = ps.get("response_head")
                raw_err = ps.get("error")
            rec = (recovered or {}).get((eid, t), {})
            req = rec.get("request_error") or classify_request_error(raw_err)
            full = rec.get("full_response") or (ps or {}).get("full_response")
            if req:
                parsed = {
                    "pred_verb": None,
                    "pred_object": None,
                    "pred_pose": None,
                    "format_valid": None,
                    "format_error": None,
                    "format_errors": [],
                }
            else:
                parsed = parse_prediction(
                    raw_pred, raw_head, verb_alias, obj_alias, object_required, object_free,
                    full_response=full,
                )
            requires_obj = gt_verb in object_required
            gt_step_obj = gt_obj if requires_obj else None
            pred_verb = parsed["pred_verb"]
            pred_obj = parsed["pred_object"]
            verb_ok = pred_verb == gt_verb
            obj_ok = (pred_obj == gt_step_obj) if requires_obj else True
            if not requires_obj:
                # object-free: verb must match and pred must not invent a required object
                joint_ok = verb_ok and pred_obj is None
            else:
                joint_ok = verb_ok and obj_ok

            if req:
                format_cat["request_failed"] += 1
            elif parsed["format_errors"]:
                for e in parsed["format_errors"]:
                    format_cat[e] += 1
            else:
                format_cat["ok"] += 1

            row = {
                "split": "test",
                "episode_id": eid,
                "step_id": t,
                "instruction": instruction,
                "is_key_action": is_key,
                "raw_gt": raw_gt,
                "raw_prediction": raw_pred,
                "full_response": full,
                "recovery": rec.get("recovery"),
                "request_error": req,
                "gt_verb": gt_verb,
                "pred_verb": pred_verb,
                "gt_object": gt_step_obj,
                "pred_object": pred_obj,
                "gt_pose": None,
                "pred_pose": None,
                "format_valid": parsed["format_valid"],
                "format_error": parsed["format_error"],
                "verb_correct": verb_ok,
                "object_correct": obj_ok if requires_obj else None,
                "joint_correct": joint_ok,
                "object_required": requires_obj,
            }
            rows.append(row)

            ep["n_steps"] += 1
            verb_class[gt_verb]["n"] += 1
            if verb_ok:
                ep["n_verb_correct"] += 1
                verb_class[gt_verb]["ok"] += 1
            if requires_obj:
                ep["n_object_required"] += 1
                if obj_ok:
                    ep["n_object_correct"] += 1
                if joint_ok:
                    ep["n_joint_object_required"] += 1
            else:
                ep["n_object_free"] += 1
                if joint_ok:
                    ep["n_joint_object_free"] += 1
            if joint_ok:
                ep["n_joint_correct"] += 1
            if req:
                ep["n_request_fail"] += 1
            elif parsed["format_valid"] is False:
                ep["n_format_fail"] += 1
            pred_pairs.append((pred_verb, pred_obj))
            if is_key:
                ep["n_key"] += 1
                gt_key_pairs.append((gt_verb, gt_obj))

        ep["n_key_matched"] = lcs_match(pred_pairs, gt_key_pairs)
        per_ep.append(ep)

    n_steps = sum(e["n_steps"] for e in per_ep)
    n_verb = sum(e["n_verb_correct"] for e in per_ep)
    n_obj_req = sum(e["n_object_required"] for e in per_ep)
    n_obj_ok = sum(e["n_object_correct"] for e in per_ep)
    n_joint = sum(e["n_joint_correct"] for e in per_ep)
    n_key = sum(e["n_key"] for e in per_ep)
    n_key_ok = sum(e["n_key_matched"] for e in per_ep)
    n_fmt = sum(e["n_format_fail"] for e in per_ep)
    n_obj_free = sum(e["n_object_free"] for e in per_ep)
    n_req = sum(e["n_request_fail"] for e in per_ep)
    n_joint_req = sum(e["n_joint_object_required"] for e in per_ep)
    n_joint_free = sum(e["n_joint_object_free"] for e in per_ep)
    recovered_ok = {
        "log_startswith_response_head",
        "log_normalized_startswith",
        "log_unique_fragment",
        "result_json_full_response",
    }
    n_recovered = sum(1 for r in rows if r.get("recovery") in recovered_ok)
    n_cache_only = sum(1 for r in rows if r.get("recovery") == "cache_only")
    ep_recalls = [e["n_key_matched"] / e["n_key"] for e in per_ep if e["n_key"] > 0]
    macro_key = sum(ep_recalls) / len(ep_recalls) if ep_recalls else 0.0
    boot = bootstrap_episode_metrics(per_ep, bootstrap_samples, bootstrap_seed)

    metrics = {
        "evaluator_version": "rh20t_action_metrics_v1_1",
        "protocol": "openloop_multiturn_teacher_force_v2",
        "split": "external_test_100",
        "n_episodes_scored": len(per_ep),
        "n_episodes_missing": len(missing_eps),
        "missing_episode_ids": missing_eps,
        "n_steps": n_steps,
        "n_files_present_not_inference_success": (
            "n_missing=0 means 100 result.json files exist; it does not mean 1131 steps succeeded"
        ),
        "request_failure": {
            "invalid": n_req,
            "denominator": n_steps,
            "pct": _pct(n_req, n_steps),
            "by_category": dict(Counter(r["request_error"] for r in rows if r.get("request_error"))),
            "note": "service/transport failures; not model format failures; remain in semantic denominators",
        },
        "recovery": {
            "from_full_log": n_recovered,
            "cache_only": n_cache_only,
            "request_failed": n_req,
        },
        "comparison_eligible": n_req == 0,
        "verb_accuracy": {
            "correct": n_verb,
            "denominator": n_steps,
            "pct": _pct(n_verb, n_steps),
            "ci95": boot["verb_acc"]["ci95"],
            "per_class": {
                k: {"correct": v["ok"], "denominator": v["n"], "pct": _pct(v["ok"], v["n"])}
                for k, v in sorted(verb_class.items())
            },
        },
        "object_accuracy": {
            "correct": n_obj_ok,
            "denominator": n_obj_req,
            "pct": _pct(n_obj_ok, n_obj_req),
            "ci95": boot["object_acc"]["ci95"],
            "object_free_steps_excluded": n_obj_free,
        },
        "joint_accuracy": {
            "correct": n_joint,
            "denominator": n_steps,
            "pct": _pct(n_joint, n_steps),
            "ci95": boot["joint_acc"]["ci95"],
            "object_required": {
                "correct": n_joint_req,
                "denominator": n_obj_req,
                "pct": _pct(n_joint_req, n_obj_req),
            },
            "object_free": {
                "correct": n_joint_free,
                "denominator": n_obj_free,
                "pct": _pct(n_joint_free, n_obj_free),
            },
            "note": (
                "All-step joint keeps the 1131 denominator. "
                "Nonzero all-step joint is often move/end only and is not grasp/place task completion."
            ),
        },
        "key_action_recall": {
            "episode_macro": {
                "mean": 100.0 * macro_key,
                "n_episodes_with_keys": len(ep_recalls),
                "ci95": boot["key_action_recall"]["ci95"],
            },
            "micro": {
                "matched": n_key_ok,
                "denominator": n_key,
                "pct": _pct(n_key_ok, n_key),
            },
        },
        "pose_tolerance_accuracy": {
            "status": "N/A",
            "reason": "predictions do not contain comparable continuous poses; GT world_point is annotation-only",
            "unconditional": None,
            "conditional": None,
            "thresholds_cm_deg": {"strict": [2, 5], "standard": [5, 15], "loose": [10, 30]},
        },
        "format_failure_rate": {
            "invalid": n_fmt,
            "denominator": n_steps,
            "pct": _pct(n_fmt, n_steps),
            "ci95": boot["format_failure_rate"]["ci95"],
            "by_category": dict(format_cat),
            "note": "request failures are excluded from this numerator",
        },
        "interpretation": (
            "Offline teacher-forced audit on 100 external RH20T test trajectories "
            "found cached request failures and missing action-object fields in some "
            "saved replies. Affected metrics are not used for model comparison until "
            "refill completes. Reported action/object/format scores are not closed-loop "
            "robot success rates. Pose accuracy is N/A without comparable continuous poses."
        ),
        "per_episode": per_ep,
    }
    return metrics, rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_audit_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_verb: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors = [r for r in rows if r.get("format_valid") is False]
    for r in rows:
        by_verb[r["gt_verb"]].append(r)
    sample: list[dict[str, Any]] = []
    per_verb = max(1, n // max(1, len(by_verb)))
    for verb, items in by_verb.items():
        take = min(per_verb, len(items))
        sample.extend(rng.sample(items, take))
    remain = [r for r in errors if r not in sample]
    extra = min(max(0, n - len(sample)), len(remain))
    if extra:
        sample.extend(rng.sample(remain, extra))
    # unique preserve order
    seen = set()
    out = []
    for r in sample:
        key = (r["episode_id"], r["step_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out[:n]


def score_prediction_tree(
    pred_root: Path,
    gt_path: Path,
    output_dir: Path,
    schema_path: Path = DEFAULT_SCHEMA,
    verb_path: Path = DEFAULT_VERB,
    object_path: Path = DEFAULT_OBJECT,
    bootstrap_seed: int = 20260912,
    bootstrap_samples: int = 10000,
    recovered_jsonl: Path | None = None,
) -> dict[str, Any]:
    schema = load_json(schema_path)
    verb_cfg = load_json(verb_path)
    obj_cfg = load_json(object_path)
    gt_pack = load_json(gt_path)
    metrics, rows = score_variant(
        pred_root, gt_pack, schema, verb_cfg, obj_cfg, bootstrap_seed, bootstrap_samples,
        recovered=load_recovered(recovered_jsonl),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_step_normalized.jsonl", rows)
    (output_dir / "metrics_test100.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    audit = build_audit_sample(rows, 80, bootstrap_seed)
    write_jsonl(output_dir / "audit_sample.jsonl", audit)
    slim = {k: v for k, v in metrics.items() if k != "per_episode"}
    (output_dir / "metrics_test100_slim.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False))
    return slim


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-root", type=Path, help="Directory of <episode>/result.json")
    p.add_argument("--predictions", type=Path, default=None, help="Unused alias for interface spec")
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--verb-aliases", type=Path, default=DEFAULT_VERB)
    p.add_argument("--object-aliases", type=Path, default=DEFAULT_OBJECT)
    p.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap-seed", type=int, default=20260912)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--recovered-jsonl", type=Path, default=None)
    args = p.parse_args()
    pred_root = args.pred_root or args.predictions
    if pred_root is None:
        raise SystemExit("provide --pred-root")
    slim = score_prediction_tree(
        pred_root=pred_root,
        gt_path=args.ground_truth,
        output_dir=args.output_dir,
        schema_path=args.schema,
        verb_path=args.verb_aliases,
        object_path=args.object_aliases,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        recovered_jsonl=args.recovered_jsonl,
    )
    print(json.dumps({k: slim[k] for k in [
        "n_episodes_scored", "n_episodes_missing", "n_steps",
        "verb_accuracy", "object_accuracy", "joint_accuracy",
        "key_action_recall", "pose_tolerance_accuracy", "format_failure_rate",
    ] if k in slim}, indent=2))


if __name__ == "__main__":
    main()
