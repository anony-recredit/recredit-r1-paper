#!/usr/bin/env python3
"""Build failctrl_v3_gr_mass_matched parquets.

Success rows are copied from the frozen Full filt pack.
Failure coefficients use a shared step budget then route:

    m_t = min(eta * S * w_t, 0.5)
    UNI:  (m/2, m/2)
    TYPE: (m(1-q), m q)

Think-empty failure steps are dropped from both arms together.
This script never reads n781 / C105 / RH20T as training rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

M_MAX = 0.5
TEST_ID_RE = ("RH20T_",)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _identity(row) -> str:
    for k in ("identity", "traj_id", "id", "episode_id"):
        if k in row and pd.notna(row[k]):
            return str(row[k])
    return ""


def _think_text(row) -> str:
    for k in ("think_t", "think", "spans_think_t"):
        if k in row and pd.notna(row[k]):
            return str(row[k]).strip()
    spans = row.get("spans")
    if isinstance(spans, dict):
        return str(spans.get("think_t") or "").strip()
    if isinstance(spans, str) and spans:
        try:
            return str(json.loads(spans).get("think_t") or "").strip()
        except Exception:
            return ""
    return ""


def _q_w(row) -> tuple[float, float]:
    q = float(row["q_t"]) if "q_t" in row and pd.notna(row["q_t"]) else float("nan")
    if "w_t" in row and pd.notna(row["w_t"]):
        w = float(row["w_t"])
    elif "s_t" in row and pd.notna(row["s_t"]):
        w = float(row["s_t"])
    else:
        w = float("nan")
    return q, w


def load_forbidden_ids(paths: list[Path]) -> set[str]:
    forbidden: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        items = raw if isinstance(raw, list) else (raw.get("examples") or raw.get("data") or [])
        for e in items:
            if not isinstance(e, dict):
                continue
            for k in ("identity", "id", "traj_id"):
                if e.get(k):
                    forbidden.add(str(e[k]))
    return forbidden


def assign_failure_weights(df: pd.DataFrame, eta: float, scale: float, routing: str) -> pd.DataFrame:
    q = df["q_t"].astype(float).to_numpy()
    w = df["w_t"].astype(float).to_numpy()
    raw = eta * scale * np.maximum(w, 0.0)
    m = np.minimum(raw, M_MAX)
    sat = raw > M_MAX
    if routing == "uniform_half":
        cg = cr = m / 2.0
    elif routing == "q_proxy":
        cg = m * (1.0 - q)
        cr = m * q
    else:
        raise ValueError(routing)
    out = df.copy()
    out["perc_weight"] = -cg
    out["reas_weight"] = -cr
    out["m_t"] = m
    out["mass_saturated"] = sat
    out["routing"] = routing
    out["eta"] = eta
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--success-train", required=True, type=Path)
    p.add_argument("--success-val", required=True, type=Path)
    p.add_argument("--failure-src", required=True, type=Path, help="Existing failctrl parquet or fail-only table")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--eta", type=float, required=True)
    p.add_argument("--weight-scale", type=float, default=1.0)
    p.add_argument("--n781-json", type=Path, default=None)
    p.add_argument("--c105-json", type=Path, default=None)
    p.add_argument("--rh20t-json", type=Path, default=None)
    args = p.parse_args()

    suc_tr = pd.read_parquet(args.success_train)
    suc_va = pd.read_parquet(args.success_val)
    fail_src = pd.read_parquet(args.failure_src)

    # Failure candidates: non-success / already-negative rows in the src mix.
    if "R_L2" in fail_src.columns:
        fail = fail_src[fail_src["R_L2"].astype(float) <= 0].copy()
    elif "perc_weight" in fail_src.columns:
        fail = fail_src[(fail_src["perc_weight"].astype(float) < 0) | (fail_src["reas_weight"].astype(float) < 0)].copy()
    else:
        raise SystemExit("failure src has neither R_L2 nor negative weights")

    if "q_t" not in fail.columns:
        raise SystemExit("failure src missing q_t")
    if "w_t" not in fail.columns:
        if "s_t" in fail.columns:
            fail["w_t"] = fail["s_t"].astype(float)
        else:
            # recover a non-negative step weight from existing |perc|+|reas| / eta / scale if present
            if "perc_weight" in fail.columns:
                fail["w_t"] = (fail["perc_weight"].astype(float).abs() + fail["reas_weight"].astype(float).abs()) / max(args.eta, 1e-12) / max(args.weight_scale, 1e-12)
            else:
                raise SystemExit("cannot recover w_t")

    think = fail.apply(_think_text, axis=1)
    fail = fail.assign(think_len=think.str.len())
    n_no_think = int((fail["think_len"] <= 0).sum())
    fail = fail[fail["think_len"] > 0].copy()

    forbidden = load_forbidden_ids([p for p in [args.n781_json, args.c105_json, args.rh20t_json] if p])
    suc_ids = set(suc_tr.apply(_identity, axis=1))
    fail_ids = fail.apply(_identity, axis=1)
    leak_fail = sorted({i for i in fail_ids if i in forbidden or i.startswith(TEST_ID_RE)})
    leak_suc = sorted({i for i in suc_ids if i in forbidden or str(i).startswith(TEST_ID_RE)})
    if leak_fail or leak_suc:
        raise SystemExit(f"LEAK: success={len(leak_suc)} fail={len(leak_fail)} sample={ (leak_suc+leak_fail)[:8]}")

    # Success IDs must match Full.
    report = {
        "protocol": "failctrl_v3_gr_mass_matched",
        "eta": args.eta,
        "weight_scale": args.weight_scale,
        "n_success_train": int(len(suc_tr)),
        "n_success_val": int(len(suc_va)),
        "n_failure_src": int(len(fail_src)),
        "n_failure_eligible": int(len(fail)),
        "n_failure_dropped_empty_think": n_no_think,
        "n_leak_success": 0,
        "n_leak_failure": 0,
        "success_train_sha256": _sha256_file(args.success_train),
        "success_val_sha256": _sha256_file(args.success_val),
        "failure_src_sha256": _sha256_file(args.failure_src),
    }

    for routing, arm in (("uniform_half", "neg_uniform"), ("q_proxy", "type_neg")):
        fail_w = assign_failure_weights(fail, args.eta, args.weight_scale, routing)
        # Keep success weights exactly as Full; do not mix into one train parquet.
        dest = args.out_root / f"eta{int(round(args.eta * 100)):03d}" / arm
        dest.mkdir(parents=True, exist_ok=True)
        suc_tr.to_parquet(dest / "success_train.parquet", index=False)
        suc_va.to_parquet(dest / "success_val.parquet", index=False)
        fail_w.to_parquet(dest / "failure_train.parquet", index=False)
        mass_u = float(fail_w["m_t"].sum())
        sat = float(fail_w["mass_saturated"].mean())
        report[arm] = {
            "n_failure": int(len(fail_w)),
            "sum_m_t": mass_u,
            "mean_m_t": float(fail_w["m_t"].mean()),
            "saturation_frac": sat,
            "sum_abs_perc": float(fail_w["perc_weight"].abs().sum()),
            "sum_abs_reas": float(fail_w["reas_weight"].abs().sum()),
        }

    uni_m = report["neg_uniform"]["sum_m_t"]
    typ_m = report["type_neg"]["sum_m_t"]
    rel = abs(uni_m - typ_m) / max(uni_m, 1e-12)
    report["mass_rel_diff"] = rel
    if rel > 1e-12:
        raise SystemExit(f"arm mass mismatch {rel}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_rep = args.out_root / f"PREPARE_REPORT_eta{int(round(args.eta * 100)):03d}.json"
    out_rep.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
