#!/usr/bin/env python3
"""Convert schema recredit_r1.v1 JSON into verl parquet rows (one OTA step per sample)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from discourse import hat_A_reas, language_action_consistency, mode_tag, mode_weight

USER_PROMPT = (
    "You are Recredit-R1, an embodied household agent in AI2-THOR.\n"
    "Given the current first-person RGB observation, continue the observation-thought-action "
    "(OTA) loop for the instruction below.\n"
    "Write a thought, then emit a single high-level action inside "
    "<DecisionMaking>...</DecisionMaking>.\n\n"
    "Instruction: {instruction}"
)


def _load_examples(path: Path) -> list[dict[str, Any]]:
    blob = json.loads(path.read_text())
    if isinstance(blob, dict) and "examples" in blob:
        return list(blob["examples"])
    if isinstance(blob, list):
        return blob
    raise ValueError(f"unrecognized json layout: {path}")


def _resolve_image(img: str, data_root: Path) -> str:
    p = Path(img)
    if p.is_absolute() and p.exists():
        return str(p)
    cand = data_root / img
    if cand.exists():
        return str(cand.resolve())
    cand2 = data_root / "data" / img
    if cand2.exists():
        return str(cand2.resolve())
    # last resort: keep relative, trainer will skip missing
    return str((data_root / img).resolve())


def scene_split(examples: list[dict[str, Any]], val_frac: float, seed: int) -> tuple[set[str], set[str]]:
    scenes = sorted({ex.get("scene") or "unknown" for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_val = max(1, int(round(len(scenes) * val_frac))) if len(scenes) > 1 else 0
    val = set(scenes[:n_val])
    train = set(scenes[n_val:]) if n_val else set(scenes)
    if not train:
        train, val = set(scenes), set()
    return train, val


def step_to_row(
    ex: dict[str, Any],
    step: dict[str, Any],
    *,
    data_root: Path,
    stage: str,
    use_discourse: bool,
) -> dict[str, Any] | None:
    spans = step.get("spans") or {}
    think = spans.get("think_t") or ""
    g_t = spans.get("g_t") or ""
    a_t = spans.get("a_t") or (step.get("action") or {}).get("raw") or ""
    y_t = spans.get("y_t") or ""
    if not y_t:
        if a_t:
            y_t = f"{think}<DecisionMaking>{a_t}</DecisionMaking>"
        else:
            y_t = think
    if not y_t.strip():
        return None

    q_t = float(step.get("q_t") or 0.0)
    R_L2 = int(ex.get("R_L2") if ex.get("R_L2") is not None else step.get("R_L2") or 0)
    A_perc = float(step.get("A_t_perc") or 0.0)
    A_reas = float(step.get("A_t_reas") or 0.0)
    w_t = float(step.get("w_t") or 0.0)
    obj = (step.get("action") or {}).get("objectType") or g_t

    m_t = language_action_consistency(think, obj)
    M_t = mode_tag(think)
    mu_t = mode_weight(R_L2, q_t, M_t)
    A_reas_hat = hat_A_reas(A_reas, m_t, mu_t) if use_discourse else A_reas

    if stage == "stage1":
        perc_w, reas_w = q_t, q_t
    else:
        perc_w, reas_w = A_perc, A_reas_hat

    img = _resolve_image(step.get("image") or "", data_root)
    instruction = ex.get("instruction") or ""
    return {
        "id": f"{ex.get('id')}_t{step.get('t')}",
        "traj_id": ex.get("id"),
        "scene": ex.get("scene"),
        "t": int(step.get("t") or 0),
        "n_steps": int(ex.get("n_steps") or len(ex.get("steps") or [])),
        "instruction": instruction,
        "prompt": USER_PROMPT.format(instruction=instruction),
        "response": y_t,
        "image": img,
        "think_t": think,
        "g_t": g_t,
        "a_t": a_t,
        "q_t": q_t,
        "w_t": w_t,
        "R_L2": R_L2,
        "A_t_perc": A_perc,
        "A_t_reas": A_reas,
        "A_t_reas_hat": A_reas_hat,
        "m_t": m_t,
        "M_t": M_t,
        "mu_t": mu_t,
        "perc_weight": perc_w,
        "reas_weight": reas_w,
        "error_type": step.get("error_type"),
        "stage": stage,
        "schema_version": ex.get("schema_version") or "recredit_r1.v1",
    }


def convert(
    json_path: Path,
    data_root: Path,
    out_dir: Path,
    *,
    stage: str,
    val_frac: float,
    seed: int,
    max_horizon: int | None,
    use_discourse: bool,
) -> None:
    import pandas as pd

    examples = _load_examples(json_path)
    train_scenes, val_scenes = scene_split(examples, val_frac, seed)
    rows_train, rows_val = [], []
    skipped = 0
    for ex in examples:
        n = int(ex.get("n_steps") or len(ex.get("steps") or []))
        if max_horizon is not None and n > max_horizon:
            skipped += 1
            continue
        scene = ex.get("scene") or "unknown"
        dest = rows_val if scene in val_scenes else rows_train
        for step in ex.get("steps") or []:
            row = step_to_row(ex, step, data_root=data_root, stage=stage, use_discourse=use_discourse)
            if row is None:
                skipped += 1
                continue
            dest.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{stage}_train.parquet"
    val_path = out_dir / f"{stage}_val.parquet"
    pd.DataFrame(rows_train).to_parquet(train_path, index=False)
    pd.DataFrame(rows_val).to_parquet(val_path, index=False)
    meta = {
        "json": str(json_path),
        "stage": stage,
        "n_train_steps": len(rows_train),
        "n_val_steps": len(rows_val),
        "n_skipped": skipped,
        "train_scenes": sorted(train_scenes),
        "val_scenes": sorted(val_scenes),
        "max_horizon": max_horizon,
        "use_discourse": use_discourse,
    }
    (out_dir / f"{stage}_split.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("wrote", train_path, val_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path, help="Embodied-Reasoner root that contains data/images")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-horizon", type=int, default=None, help="Stage I: keep short/mid clips, e.g. 8")
    p.add_argument("--no-discourse", action="store_true")
    args = p.parse_args()
    convert(
        args.json,
        args.data_root,
        args.out_dir,
        stage=args.stage,
        val_frac=args.val_frac,
        seed=args.seed,
        max_horizon=args.max_horizon,
        use_discourse=not args.no_discourse,
    )


if __name__ == "__main__":
    main()
