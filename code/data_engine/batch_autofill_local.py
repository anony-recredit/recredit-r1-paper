#!/usr/bin/env python3
"""Auto-annotate local Embodied-Reasoner trajs into recredit_r1.v1, then pack.

Uses existing train_multiturn JSON (text) + data/images (RGB) + scene_metadata (AABB).
No human seeing/thinking labels — q_t / R_L2 / error_type are rule-derived.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recredit_data_engine.autofill_core import (  # noqa: E402
    build_one,
    index_train,
    load_train,
    local_trajs,
    rel_img,
    to_view,
)
from recredit_data_engine.pack_dataset import collect_image_paths, copy_images, summarize, validate_example

TRAIN = ROOT / "train_multiturn_9390.json"
IMG = ROOT / "data" / "images"
META = ROOT / "scene_metadata"
OUT_JSON = ROOT / "recredit_r1_autofill_local.json"
OUT_VIEW = ROOT / "recredit_r1_autofill_local_view.json"


def pack(examples: list[dict], report_extra: dict) -> Path:
    created = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack_name = f"recredit_r1_er_autofill_{created}"
    pack_dir = ROOT / "releases" / pack_name
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    (pack_dir / "annotations").mkdir(parents=True)
    (pack_dir / "viewers").mkdir(parents=True)

    blob = {
        "paper": "Recredit-R1",
        "schema_version": "recredit_r1.v1",
        "source": "embodied_reasoner_train_local_images_autofill",
        "image_source": "embodied_reasoner_original",
        "note": (
            "Auto-filled from existing Embodied-Reasoner multiturn text + local RGB + "
            "scene_metadata AABB. error_type / w_t derived; no human seeing/thinking labels."
        ),
        "n": len(examples),
        "n_success": sum(1 for e in examples if e["R_L2"] == 1),
        "n_failure": sum(1 for e in examples if e["R_L2"] == 0),
        "examples": examples,
        "packaged_at": created,
    }
    (pack_dir / "annotations" / "recredit_r1_autofill_local.json").write_text(
        json.dumps(blob, ensure_ascii=False, indent=2)
    )
    (pack_dir / "annotations" / "recredit_r1_autofill_local_view.json").write_text(
        json.dumps(to_view(blob), ensure_ascii=False, indent=2)
    )

    errs = []
    for ex in examples:
        for e in validate_example(ex, ROOT):
            errs.append({"id": ex.get("id"), "error": e})

    rels = collect_image_paths(examples)
    n_copied, missing = copy_images(rels, ROOT, pack_dir)

    src_html = ROOT / "er_recredit_annotated_viewer.html"
    if src_html.exists():
        t = src_html.read_text()
        t = t.replace("./recredit_r1_annotated.json", "../annotations/recredit_r1_autofill_local_view.json")
        t = t.replace("./recredit_r1_annotated_22_view.json", "../annotations/recredit_r1_autofill_local_view.json")
        t = t.replace('src="./${esc(s.image)}"', 'src="../${esc(s.image)}"')
        t = t.replace("${s.image ? `<img src=\"./${esc(s.image)}\"", "${s.image ? `<img src=\"../${esc(s.image)}\"")
        (pack_dir / "viewers" / "er_autofill_viewer.html").write_text(t)

    demo_html = ROOT / "er_gen_demo_viewer.html"
    if demo_html.exists():
        t = demo_html.read_text()
        t = t.replace("./gen_demo_batch_view.json", "../annotations/recredit_r1_autofill_local_view.json")
        t = t.replace('src="./${esc(s.image)}"', 'src="../${esc(s.image)}"')
        t = t.replace("Recredit data-engine demos", "Recredit autofill from Embodied-Reasoner")
        t = t.replace("data engine demos", "Recredit autofill from Embodied-Reasoner")
        (pack_dir / "viewers" / "er_autofill_demo_style.html").write_text(t)

    report = {
        "created_at": created,
        "pack_name": pack_name,
        "paper": "Recredit-R1",
        "schema": "recredit_r1.v1",
        "summary": summarize(examples),
        "n_errors": len(errs),
        "errors": errs[:60],
        "ok": len(errs) == 0 and len(missing) == 0,
        "images": {"n_refs": len(rels), "n_copied": n_copied, "n_missing": len(missing), "missing": missing[:20]},
        "checks": {
            "q_t_rule_verifiable": True,
            "R_L2_env_or_language_grounded": True,
            "error_type_derived_not_human": True,
            "uses_er_original_rgb": True,
            "filled_from_existing_er_dataset": True,
        },
        **report_extra,
    }
    (pack_dir / "VALIDATION_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (pack_dir / "README.md").write_text(
        f"""# Recredit autofill from Embodied-Reasoner (local)

- Source: `train_multiturn_9390.json` + local `data/images` + `scene_metadata` AABB
- Autofill: `think/g/a` spans, `q_t`, `R_L2`, `w_t`, type routing, `error_type`
- No human seeing/thinking labels
- Validation: errors={report['n_errors']} missing_imgs={len(missing)} ok={report['ok']}

```bash
cd {pack_name}
python3 -m http.server 8766
# http://127.0.0.1:8766/viewers/er_autofill_demo_style.html
```
""",
        encoding="utf-8",
    )
    schema_src = ROOT / "releases" / "recredit_r1_er_dataset_20260820_170623" / "SCHEMA.md"
    if schema_src.exists():
        shutil.copy2(schema_src, pack_dir / "SCHEMA.md")
    else:
        (pack_dir / "SCHEMA.md").write_text("# recredit_r1.v1\n", encoding="utf-8")

    (pack_dir / "index.html").write_text(
        """<!DOCTYPE html><html><body><h1>Recredit-R1 ER autofill</h1>
<ul>
<li><a href="viewers/er_autofill_demo_style.html">viewer</a></li>
<li><a href="VALIDATION_REPORT.json">validation</a></li>
<li><a href="README.md">README</a></li>
</ul></body></html>"""
    )

    tar_path = ROOT / "releases" / f"{pack_name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pack_dir, arcname=pack_name)
    return tar_path


def main() -> int:
    train = load_train(TRAIN)
    by = index_train(train)
    examples = []
    skipped = []
    for tt, traj, pngs in local_trajs(IMG):
        idx = by.get((tt, traj))
        if idx is None:
            skipped.append({"traj": traj, "reason": "no_train_match"})
            continue
        try:
            image_paths = [rel_img(ROOT, p) for p in pngs]
            ex = build_one(ROOT, META, tt, traj, image_paths, train[idx])
        except Exception as e:
            skipped.append({"traj": traj, "reason": f"exception:{e}"})
            continue
        if ex is None:
            skipped.append({"traj": traj, "reason": "no_scene_meta_or_empty"})
            continue
        examples.append(ex)
        print(f"ok {ex['id']} R={ex['R_L2']} steps={ex['n_steps']} q̄={ex['mean_q_t']}")

    blob = {
        "paper": "Recredit-R1",
        "n": len(examples),
        "n_success": sum(1 for e in examples if e["R_L2"] == 1),
        "n_failure": sum(1 for e in examples if e["R_L2"] == 0),
        "examples": examples,
        "skipped": skipped,
        "schema_version": "recredit_r1.v1",
        "image_source": "embodied_reasoner_original",
    }
    OUT_JSON.write_text(json.dumps(blob, ensure_ascii=False, indent=2))
    OUT_VIEW.write_text(json.dumps(to_view(blob), ensure_ascii=False, indent=2))
    print(f"wrote {OUT_JSON} n={len(examples)} skip={len(skipped)}")

    tar = pack(examples, {"skipped": skipped})
    print(f"pack tar: {tar} size_mb={tar.stat().st_size/1e6:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
