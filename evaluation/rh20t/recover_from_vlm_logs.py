#!/usr/bin/env python3
"""Recover full VLM replies from serving logs and classify request failures.

The cached result.json stores pred=verb and a ~240-char response_head. Object
arguments often appear later in <DecisionMaking>. This writes a recovered
overlay used by the evaluator. It does not shrink the 1131-step denominator.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

RESP_RE = re.compile(r"^\d+\s+(\[.*\])\s*$")


def classify_request_error(err: str | None) -> str | None:
    if not err:
        return None
    e = str(err)
    if "Connection refused" in e or "Errno 111" in e:
        return "connection_refused"
    if any(k in e for k in ("RemoteDisconnected", "Connection aborted", "ConnectionReset", "Broken pipe")):
        return "connection_interrupted"
    if "HTTPConnectionPool" in e or "Max retries" in e or "Read timed out" in e:
        return "http_transport"
    return "other_request_error"


def extract_vlm_texts(log_dir: Path, variant: str) -> list[str]:
    texts: list[str] = []
    for p in sorted(log_dir.glob(f"vlm_{variant}_g*_p*.log")):
        for line in p.read_text(errors="replace").splitlines():
            m = RESP_RE.match(line.strip())
            if not m:
                continue
            try:
                obj = ast.literal_eval(m.group(1))
            except Exception:
                continue
            if isinstance(obj, list) and obj and isinstance(obj[0], str):
                texts.append(obj[0])
            elif isinstance(obj, str):
                texts.append(obj)
    return texts


def _normalize_head(s: str) -> str:
    return " ".join((s or "").replace("\r\n", "\n").split())


def recover_variant(pred_root: Path, log_dir: Path, variant: str) -> dict:
    pool = extract_vlm_texts(log_dir, variant)
    used = [False] * len(pool)
    pending: list[dict] = []
    stats = Counter()
    for ep_path in sorted((pred_root / variant).glob("*/result.json")):
        doc = json.loads(ep_path.read_text())
        eid = doc.get("id") or ep_path.parent.name
        for s in doc.get("steps") or []:
            req = classify_request_error(s.get("error"))
            head = s.get("response_head") or ""
            on_disk = s.get("full_response") or None
            pending.append({
                "episode_id": eid,
                "step_id": int(s["t"]),
                "pred_cached": s.get("pred"),
                "response_head": head,
                "full_response": on_disk,
                "recovery": None,
                "request_error": req,
                "raw_error": s.get("error"),
            })

    # Steps that already store a full reply (local refill) must not consume remote VLM logs.
    for row in pending:
        if row["request_error"] or not row.get("full_response"):
            continue
        row["recovery"] = "result_json_full_response"
        stats["result_json_full_response"] += 1

    # Longest head first so a short prefix cannot steal a unique garbled reply.
    order = sorted(
        (
            i for i, row in enumerate(pending)
            if not row["request_error"] and row["response_head"] and not row.get("recovery")
        ),
        key=lambda i: len(pending[i]["response_head"]),
        reverse=True,
    )
    for i in order:
        row = pending[i]
        head = row["response_head"]
        hits = [j for j, t in enumerate(pool) if (not used[j]) and t.startswith(head)]
        how = None
        full = None
        if hits:
            j = min(hits, key=lambda k: len(pool[k]))
            used[j] = True
            full = pool[j]
            how = "log_startswith_response_head"
        else:
            nh = _normalize_head(head)
            hits = [
                j for j, t in enumerate(pool)
                if (not used[j]) and _normalize_head(t).startswith(nh)
            ]
            if hits:
                j = min(hits, key=lambda k: len(pool[k]))
                used[j] = True
                full = pool[j]
                how = "log_normalized_startswith"
            elif len(head) >= 24:
                frag = _normalize_head(head[:80])
                hits = [j for j, t in enumerate(pool) if (not used[j]) and frag in _normalize_head(t)]
                if len(hits) == 1:
                    used[hits[0]] = True
                    full = pool[hits[0]]
                    how = "log_unique_fragment"
        if how:
            row["full_response"] = full
            row["recovery"] = how
            stats[how] += 1
        elif row.get("full_response"):
            row["recovery"] = "result_json_full_response"
            stats["result_json_full_response"] += 1
        else:
            row["recovery"] = "cache_only"
            stats["unrecovered_cache_only"] += 1

    steps = []
    for row in pending:
        if row["request_error"]:
            row["recovery"] = "request_failed"
            stats[row["request_error"]] += 1
        elif not row["recovery"]:
            if row.get("full_response"):
                row["recovery"] = "result_json_full_response"
                stats["result_json_full_response"] += 1
            else:
                row["recovery"] = "cache_only"
                stats["unrecovered_cache_only"] += 1
        steps.append(row)
    n = len(steps)
    return {
        "variant": variant,
        "n_steps": n,
        "n_log_texts": len(pool),
        "n_request_failed": sum(stats[k] for k in stats if k.startswith("connection") or k in {"http_transport", "other_request_error"}),
        "counts": dict(stats),
        "steps": steps,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variants", nargs="+", default=[
        "q2_stage1_ol", "q2_full_ol", "q2_dpo_ol",
        "q3_stage1_ol", "q3_full_ol", "q3_dpo_ol",
    ])
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for v in args.variants:
        rec = recover_variant(args.pred_root, args.log_dir, v)
        dest = args.output_dir / f"{v}.jsonl"
        with dest.open("w") as f:
            for row in rec["steps"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        slim = {k: rec[k] for k in rec if k != "steps"}
        (args.output_dir / f"{v}_summary.json").write_text(json.dumps(slim, indent=2))
        summary.append(slim)
        print(json.dumps(slim, ensure_ascii=False))
    (args.output_dir / "RECOVERY_SUMMARY.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
