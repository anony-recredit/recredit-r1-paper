#!/usr/bin/env python3
"""Build Stage-I U2 parquet: mass-match REAL Geo token weights.

Geo token weight (must match Qwen3 ``recredit_r1_qwen3/dataset.py``):

  - tokenize ``response`` with offset_mapping
  - grounding-span tokens → ``perc_weight``
  - other masked response tokens → ``reas_weight``

Per trajectory τ:

  c_τ = (Σ_{i∈τ} Σ_{j∈M_i} w^{Geo}_{ij}) / (Σ_{i∈τ} |M_i|)
  perc_weight = reas_weight = c_τ

so U2 total weighted token mass equals Geo; only within-traj spatial allocation
is removed. Matching ``q_t`` alone is FORBIDDEN (that under-scales mass ~0.58×).

Requires ``--tokenizer`` (Qwen3) for production builds. Exits non-zero if any
traj fails |mass_U2 − mass_Geo| > --atol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _char_span_in_text(haystack: str, needle: str) -> tuple[int, int] | None:
    """Byte-identical to recredit_r1_qwen3/dataset.py::_char_span_in_text."""
    if not needle:
        return None
    i = haystack.find(needle)
    if i < 0:
        return None
    return i, i + len(needle)


def geo_token_weights_for_response(
    tokenizer,
    response: str,
    think_t: str,
    g_t: str,
    a_t: str,
    perc_weight: float,
    reas_weight: float,
) -> tuple[list[float], list[int]]:
    """Return (weights, mask) for response tokens — Stage-I positive branch.

    Logic mirrored from recredit_r1_qwen3/dataset.py::_token_weights_for_response
    for non-negative perc/reas (Stage I). Negative-credit branch is unused here.
    """
    enc = tokenizer(response or "", add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(enc["offset_mapping"])
    g_span = _char_span_in_text(response or "", g_t) if g_t else None
    think_span = _char_span_in_text(response or "", think_t) if think_t else None
    a_span = _char_span_in_text(response or "", a_t) if a_t else None

    weights: list[float] = []
    mask: list[int] = []
    for s, e in offsets:
        if e <= s:
            weights.append(0.0)
            mask.append(0)
            continue
        use_perc = g_span is not None and s < g_span[1] and e > g_span[0]
        w = float(perc_weight) if use_perc else float(reas_weight)
        if think_span is None and a_span is None and g_span is None:
            w = float(reas_weight)
        weights.append(w)
        mask.append(1)
    return weights, mask


def row_geo_mass(
    tokenizer,
    row: dict[str, Any],
) -> tuple[float, int]:
    w, m = geo_token_weights_for_response(
        tokenizer,
        str(row.get("response") or ""),
        str(row.get("think_t") or ""),
        str(row.get("g_t") or ""),
        str(row.get("a_t") or ""),
        float(row["perc_weight"]),
        float(row["reas_weight"]),
    )
    mass = sum(wi for wi, mi in zip(w, m) if mi)
    n = sum(m)
    return float(mass), int(n)


def compute_c_tau_by_traj(
    df: pd.DataFrame,
    tokenizer,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    mass_geo: dict[str, float] = defaultdict(float)
    n_mask: dict[str, float] = defaultdict(float)
    for row in df.to_dict(orient="records"):
        tid = str(row["traj_id"])
        m, n = row_geo_mass(tokenizer, row)
        mass_geo[tid] += m
        n_mask[tid] += n

    c_map: dict[str, float] = {}
    audit: dict[str, dict[str, float]] = {}
    for tid, nsum in n_mask.items():
        if nsum <= 0:
            raise ValueError(f"traj {tid}: zero masked-token count")
        c = mass_geo[tid] / nsum
        c_map[tid] = c
        audit[tid] = {
            "mass_geo": mass_geo[tid],
            "n_mask": nsum,
            "c_tau": c,
            "mass_u2": c * nsum,
        }
    return c_map, audit


def assert_mass_match(audit: dict[str, dict[str, float]], *, atol: float) -> None:
    bad = [
        f"{tid}: |{a['mass_u2']}-{a['mass_geo']}|={abs(a['mass_u2']-a['mass_geo'])}>{atol}"
        for tid, a in audit.items()
        if abs(a["mass_u2"] - a["mass_geo"]) > atol
    ]
    if bad:
        raise AssertionError("U2 mass match failed:\n" + "\n".join(bad[:30]))


def verify_u2_against_geo(
    geo_df: pd.DataFrame,
    u2_df: pd.DataFrame,
    tokenizer,
    *,
    atol: float,
) -> dict[str, Any]:
    """Re-tokenize both frames and confirm per-traj mass equality."""
    c_geo, audit_geo = compute_c_tau_by_traj(geo_df, tokenizer)
    # U2 rows must have perc=reas=c; recompute mass with same span rule
    mass_u2: dict[str, float] = defaultdict(float)
    n_u2: dict[str, float] = defaultdict(float)
    for row in u2_df.to_dict(orient="records"):
        tid = str(row["traj_id"])
        m, n = row_geo_mass(tokenizer, row)
        mass_u2[tid] += m
        n_u2[tid] += n

    ratios = []
    n_matched = 0
    max_rel = 0.0
    for tid, a in audit_geo.items():
        g = a["mass_geo"]
        u = mass_u2[tid]
        if g == 0:
            ok = abs(u) <= atol
            rel = 0.0
        else:
            rel = abs(u - g) / abs(g)
            ok = abs(u - g) <= atol
        if ok:
            n_matched += 1
        max_rel = max(max_rel, rel)
        ratios.append(u / g if g else float("nan"))

    import statistics

    finite = [r for r in ratios if r == r]
    return {
        "n_traj": len(audit_geo),
        "n_mass_matched": n_matched,
        "max_rel_diff": max_rel,
        "median_u2_over_geo": statistics.median(finite) if finite else None,
        "mean_u2_over_geo": statistics.mean(finite) if finite else None,
        "ok": n_matched == len(audit_geo),
        "c_tau_n": len(c_geo),
    }


def transform_frame(df: pd.DataFrame, tokenizer, *, atol: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"traj_id", "response", "think_t", "g_t", "a_t", "perc_weight", "reas_weight"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"parquet missing columns: {sorted(missing)}")

    c_map, audit = compute_c_tau_by_traj(df, tokenizer)
    assert_mass_match(audit, atol=atol)

    out = df.copy()
    constants = [c_map[str(t)] for t in out["traj_id"].tolist()]
    out["perc_weight"] = constants
    out["reas_weight"] = constants
    out["u2_c_tau"] = constants
    out["credit_mode"] = "u2_mass_matched_uniform_geo_token"

    # Post-check: U2 mass via same span rule
    v = verify_u2_against_geo(df, out, tokenizer, atol=atol)
    if not v["ok"]:
        raise AssertionError(f"post-verify mass match failed: {v}")

    meta = {
        "n_rows": int(len(out)),
        "n_traj": int(df["traj_id"].nunique()),
        "atol": atol,
        "max_abs_mass_err": max(abs(a["mass_u2"] - a["mass_geo"]) for a in audit.values()),
        "definition": (
            "c_tau = sum_i sum_{j in M_i} w^Geo_ij / sum_i |M_i|; "
            "w^Geo from perc/reas + g_span mask (Qwen3 dataset.py)"
        ),
        "verify": v,
    }
    return out, meta


def convert_split(src: Path, dst: Path, *, tokenizer_path: str, atol: float) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    df = pd.read_parquet(src)
    out, meta = transform_frame(df, tok, atol=atol)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)
    meta.update(
        {
            "src": str(src),
            "dst": str(dst),
            "src_sha256": _sha256_file(src),
            "dst_sha256": _sha256_file(dst),
            "tokenizer": tokenizer_path,
        }
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=str, required=True, help="HF tokenizer (Qwen3)")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["stage1_train.parquet", "stage1_val.parquet"],
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"splits": {}, "ok": True}
    try:
        for name in args.splits:
            src = args.src_dir / name
            if not src.exists():
                raise FileNotFoundError(src)
            meta = convert_split(src, args.out_dir / name, tokenizer_path=args.tokenizer, atol=args.atol)
            report["splits"][name] = meta
            print(json.dumps(meta, indent=2))
    except Exception as e:
        report["ok"] = False
        report["error"] = str(e)
        (args.out_dir / "U2_BUILD_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
        print("FAIL:", e, file=sys.stderr)
        sys.exit(1)

    (args.out_dir / "U2_BUILD_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print("OK wrote", args.out_dir)


if __name__ == "__main__":
    main()
