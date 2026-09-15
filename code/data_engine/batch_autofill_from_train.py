#!/usr/bin/env python3
"""Convert ER train_multiturn_9390.json → recredit_r1.v1 (full or partial).

Iterates every trajectory in the multiturn JSON, resolves RGB paths from
`images[]`, and skips items missing scene_metadata or local files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recredit_data_engine.autofill_core import (  # noqa: E402
    build_one,
    load_train,
    resolve_image_paths,
    to_view,
)
from recredit_data_engine.batch_autofill_local import pack  # noqa: E402


def run(
    *,
    er_root: Path,
    train_path: Path,
    meta_dir: Path,
    out_json: Path,
    out_view: Path,
    require_images: bool,
    limit: int | None,
) -> int:
    train = load_train(train_path)
    examples: list[dict] = []
    skipped: list[dict] = []

    for i, item in enumerate(train):
        if limit is not None and len(examples) >= limit:
            break
        resolved = resolve_image_paths(er_root, item, require_exists=require_images)
        if resolved is None:
            key = f"idx_{i}"
            skipped.append({"index": i, "traj": key, "reason": "no_images_or_missing_files"})
            continue
        tt, traj, image_paths = resolved
        try:
            ex = build_one(er_root, meta_dir, tt, traj, image_paths, item)
        except Exception as e:
            skipped.append({"index": i, "traj": traj, "reason": f"exception:{e}"})
            continue
        if ex is None:
            skipped.append({"index": i, "traj": traj, "reason": "no_scene_meta_or_empty"})
            continue
        examples.append(ex)
        if len(examples) % 100 == 0 or len(examples) <= 5:
            print(f"ok [{len(examples)}] {ex['id']} R={ex['R_L2']} steps={ex['n_steps']}")

    blob = {
        "paper": "Recredit-R1",
        "n": len(examples),
        "n_success": sum(1 for e in examples if e["R_L2"] == 1),
        "n_failure": sum(1 for e in examples if e["R_L2"] == 0),
        "examples": examples,
        "skipped": skipped,
        "schema_version": "recredit_r1.v1",
        "image_source": "embodied_reasoner_original",
        "source": "embodied_reasoner_train_multiturn_autofill",
        "train_path": str(train_path),
        "n_train_total": len(train),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(blob, ensure_ascii=False, indent=2))
    out_view.write_text(json.dumps(to_view(blob), ensure_ascii=False, indent=2))
    print(f"wrote {out_json} n={len(examples)} skip={len(skipped)} / train={len(train)}")

    if examples:
        tar = pack(examples, {"skipped": skipped, "mode": "from_train", "train_total": len(train)})
        print(f"pack tar: {tar}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="ER multiturn → recredit_r1.v1 autofill")
    p.add_argument("--er-root", type=Path, default=ROOT, help="Embodied-Reasoner dataset root")
    p.add_argument("--train", type=Path, default=ROOT / "train_multiturn_9390.json")
    p.add_argument("--meta", type=Path, default=ROOT / "scene_metadata")
    p.add_argument("--out", type=Path, default=ROOT / "recredit_r1_autofill_full.json")
    p.add_argument("--out-view", type=Path, default=ROOT / "recredit_r1_autofill_full_view.json")
    p.add_argument(
        "--require-images",
        action="store_true",
        default=True,
        help="Skip trajectories whose RGB files are missing locally (default: on)",
    )
    p.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Keep trajectories even if RGB files are absent (paths only)",
    )
    p.add_argument("--limit", type=int, default=None, help="Stop after N successful examples")
    args = p.parse_args()
    require_images = not args.allow_missing_images
    return run(
        er_root=args.er_root,
        train_path=args.train,
        meta_dir=args.meta,
        out_json=args.out,
        out_view=args.out_view,
        require_images=require_images,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
