#!/usr/bin/env python3
"""Score the six frozen RH20T OL variants with one parser."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
sys.path.insert(0, str(HERE))
from evaluate_action_metrics import score_prediction_tree  # noqa: E402
from recover_from_vlm_logs import recover_variant  # noqa: E402

GT = PAPER / "data/rh20t_recredit_cfg3_pc_n100/all.json"
PRED = PAPER / "results/rh20t_cfg3_pc_n100"
LOGS = PRED / "logs"
OUT = PAPER / "results/rh20t_metrics"
REC = OUT / "recovered"

VARIANTS = [
    ("RH-Q2-S1", "q2_stage1_ol", "Qwen2 Stage I"),
    ("RH-Q2-F", "q2_full_ol", "Qwen2 Full CE"),
    ("RH-Q2-D", "q2_dpo_ol", "Qwen2 DPO"),
    ("RH-Q3-S1", "q3_stage1_ol", "Qwen3 Stage I"),
    ("RH-Q3-F", "q3_full_ol", "Qwen3 Full CE"),
    ("RH-Q3-D", "q3_dpo_ol", "Qwen3 DPO"),
]


def main() -> None:
    table = []
    for run_id, dirname, label in VARIANTS:
        pred_root = PRED / dirname
        out_dir = OUT / dirname
        rec = recover_variant(PRED, LOGS, dirname)
        rec_path = REC / f"{dirname}.jsonl"
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in rec["steps"]))
        print(f"== {run_id} {dirname} recovered={rec['counts']}", flush=True)
        slim = score_prediction_tree(
            pred_root=pred_root,
            gt_path=GT,
            output_dir=out_dir,
            bootstrap_seed=20260912,
            bootstrap_samples=10000,
            recovered_jsonl=rec_path,
        )
        row = {
            "run_id": run_id,
            "label": label,
            "dirname": dirname,
            "n_episodes": slim["n_episodes_scored"],
            "n_missing": slim["n_episodes_missing"],
            "n_steps": slim["n_steps"],
            "comparison_eligible": slim.get("comparison_eligible"),
            "request_failure": slim.get("request_failure"),
            "recovery": slim.get("recovery"),
            "verb": slim["verb_accuracy"],
            "object": slim["object_accuracy"],
            "joint": slim["joint_accuracy"],
            "key_recall_macro": slim["key_action_recall"]["episode_macro"],
            "key_recall_micro": slim["key_action_recall"]["micro"],
            "pose": slim["pose_tolerance_accuracy"]["status"],
            "format": slim["format_failure_rate"],
        }
        table.append(row)
        print(
            f"  verb {row['verb']['correct']}/{row['verb']['denominator']} "
            f"joint {row['joint']['correct']}/{row['joint']['denominator']} "
            f"fmt {row['format']['invalid']}/{row['format']['denominator']}",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "TABLE_six_variants.json").write_text(json.dumps(table, indent=2, ensure_ascii=False))
    print(f"wrote {OUT / 'TABLE_six_variants.json'}")


if __name__ == "__main__":
    main()
