#!/usr/bin/env python3
"""Audit v28 Stage2 pack + Attr-DPO quotas. Exit 1 on hard gate failure."""
from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(os.environ["RECREDIT_ROOT"])
PACK = BASE / "data/wave23_shared_p123_v28"
CE = BASE / "data/parquet_p123_shared/P3_recredit_ablation_v28_stage2"
DPO = BASE / "data/parquet_p123_shared/P3_recredit_ablation_v28_dpo"
TEST_C = BASE / "data/eval/test_C_105.json"
N806 = BASE / "data/eval/test_809_n806_no_emulator3.json"
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"OK  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"FAIL {name}" + (f"  [{detail}]" if detail else ""))
        FAIL.append(name)


def ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    x = json.loads(path.read_text())
    items = x if isinstance(x, list) else (x.get("examples") or x.get("data") or [])
    return {str(e["identity"]) for e in items if "identity" in e}


def main() -> int:
    bad = ids(TEST_C) | ids(N806)

    # --- Stage2 pack ---
    meta_p = PACK / "PACK_META.json"
    check("pack_meta", meta_p.exists())
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        check("n_train>=4000", meta.get("n_train", 0) >= 4000, str(meta.get("n_train")))
        check("has_closerep_aug", meta.get("n_aug_rows", 0) > 0, str(meta.get("n_aug_rows")))
        check("anti_illegal_separate", meta.get("n_anti_illegal_dpo_only", 0) > 0)

    train_json = PACK / "shared_train.json"
    if train_json.exists():
        ex = json.loads(train_json.read_text())["examples"]
        neg = sum(
            1
            for e in ex
            if "skill_neg_" in str(e.get("thor_train_pool") or "")
            or str(e.get("skill22_from") or "").startswith("neg_")
        )
        check("ce_no_skill_neg", neg == 0, str(neg))
        leak = [e.get("id") for e in ex if str(e.get("identity")) in bad]
        check("ce_no_eval_leak", len(leak) == 0, str(len(leak)))

    # --- CE parquet ---
    if (CE / "stage2_train.parquet").exists():
        df = pd.read_parquet(CE / "stage2_train.parquet", columns=["perc_weight", "reas_weight"])
        pw = df["perc_weight"].astype(float)
        rw = df["reas_weight"].astype(float)
        check("ce_pos_only", int((pw < 0).sum()) == 0 and int((rw < 0).sum()) == 0)
        check("ce_n>=20000", len(df) >= 20000, str(len(df)))
    else:
        print("SKIP ce_parquet (not built yet)")

    # --- DPO quotas ---
    dpo_pq = DPO / "dpo_train.parquet"
    if dpo_pq.exists():
        df = pd.read_parquet(dpo_pq)
        n = len(df)
        kinds = df["pair_kind"].value_counts().to_dict()
        check("dpo_n>=800", n >= 800, str(n))
        check("attr_cols", {"q_t", "w_t", "e_t", "pair_weight"}.issubset(df.columns))
        c_skill = kinds.get("holding", 0) + kinds.get("bridge", 0) + kinds.get("seg2", 0)
        cr = kinds.get("closerep_seeing", 0)
        see = kinds.get("seeing_wrong", 0)
        check("C_skill_frac>=0.30", c_skill / n >= 0.30, f"{c_skill}/{n}={c_skill/n:.3f}")
        check("closerep_frac>=0.20", cr / n >= 0.20, f"{cr}/{n}={cr/n:.3f}")
        check("seeing_frac<=0.15", see / n <= 0.15, f"{see}/{n}={see/n:.3f}")
        leak = [i for i in df.get("identity", []).astype(str).tolist() if i in bad]
        check("dpo_no_eval_leak", len(leak) == 0, str(len(leak)))
        meta = json.loads((DPO / "DPO_META.json").read_text()) if (DPO / "DPO_META.json").exists() else {}
        check("meta_test_false", meta.get("test_set_used") is False)
        print("KINDS", kinds)
        print("FAMILY", {"C_skill": c_skill, "closerep": cr, "seeing": see})
    else:
        print("SKIP dpo_parquet (not built yet)")

    ref = DPO / "dpo_train_with_ref.parquet"
    if ref.exists():
        rdf = pd.read_parquet(ref)
        check("ref_n_match", len(rdf) == len(pd.read_parquet(dpo_pq)))
        check("has_ref_logp", {"ref_logp_chosen", "ref_logp_rejected"}.issubset(rdf.columns))

    print("FAILS", len(FAIL), FAIL)
    report = {"ok": len(FAIL) == 0, "fails": FAIL}
    (DPO if DPO.exists() else PACK).mkdir(parents=True, exist_ok=True)
    out = PACK / "AUDIT_REPORT.json"
    out.write_text(json.dumps(report, indent=2))
    print("WROTE", out)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
