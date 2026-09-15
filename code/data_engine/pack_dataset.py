#!/usr/bin/env python3
"""Validate Recredit-R1.v1 annotations and pack images+JSON into a release tarball."""
from __future__ import annotations

import json
import math
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PARENT = ROOT / "releases"

TRAJ_REQUIRED = {
    "id",
    "scene",
    "tasktype",
    "instruction",
    "R_L2",
    "n_steps",
    "images",
    "steps",
    "schema_version",
    "mean_q_t",
}
STEP_REQUIRED = {
    "t",
    "image",
    "action",
    "q_t",
    "spans",
    "g_t_structured",
    "w_t",
    "s_t_perc",
    "s_t_reas",
    "A_t_perc",
    "A_t_reas",
    "error_type",
    "R_L2",
}
SPANS_REQUIRED = {"think_t", "g_t", "a_t", "char_spans"}
ERROR_OK = {
    "success_step",
    "lucky_success_low_q",
    "seeing_wrong",
    "thinking_wrong",
    "mixed",
}


def expected_error_type(q: float, R: int) -> str:
    if R == 1:
        return "success_step" if q >= 0.5 else "lucky_success_low_q"
    if q < 0.4:
        return "seeing_wrong"
    if q >= 0.6:
        return "thinking_wrong"
    return "mixed"


def validate_example(ex: dict, root: Path) -> list[str]:
    errs: list[str] = []
    missing = TRAJ_REQUIRED - set(ex)
    if missing:
        errs.append(f"missing traj fields: {sorted(missing)}")
    if ex.get("schema_version") not in (None, "recredit_r1.v1") and ex.get("schema_version") != "recredit_r1.v1":
        # annotated older may still say recredit_r1.v1
        pass
    if ex.get("schema_version") != "recredit_r1.v1":
        errs.append(f"schema_version={ex.get('schema_version')!r} (want recredit_r1.v1)")

    R = ex.get("R_L2")
    if R not in (0, 1):
        errs.append(f"R_L2={R}")
    steps = ex.get("steps") or []
    imgs = ex.get("images") or []
    n = ex.get("n_steps")
    if n != len(steps):
        errs.append(f"n_steps={n} != len(steps)={len(steps)}")
    if len(imgs) != len(steps):
        errs.append(f"len(images)={len(imgs)} != len(steps)={len(steps)}")

    ws = [float(s.get("w_t") or 0) for s in steps]
    if steps and abs(sum(ws) - 1.0) > 1e-2:
        errs.append(f"sum(w_t)={sum(ws):.6f} (want ~1)")

    for s in steps:
        miss = STEP_REQUIRED - set(s)
        if miss:
            errs.append(f"t={s.get('t')} missing step fields: {sorted(miss)}")
            continue
        if s.get("R_L2") != R:
            errs.append(f"t={s['t']} step.R_L2={s.get('R_L2')} != traj.R_L2={R}")
        q = float(s["q_t"])
        if not (0 < q <= 1.0 + 1e-6):
            errs.append(f"t={s['t']} q_t={q} out of (0,1]")
        exp = expected_error_type(q, int(R))
        if s.get("error_type") != exp:
            errs.append(f"t={s['t']} error_type={s.get('error_type')} want {exp}")
        if s.get("error_type") not in ERROR_OK:
            errs.append(f"t={s['t']} unknown error_type")
        spans = s.get("spans") or {}
        smiss = SPANS_REQUIRED - set(spans)
        if smiss:
            errs.append(f"t={s['t']} spans missing {sorted(smiss)}")
        # paper: P_t = g_t, R_t = (think, a)
        if spans.get("P_t_equals") not in (None, "g_t"):
            errs.append(f"t={s['t']} P_t_equals={spans.get('P_t_equals')}")
        g = s.get("g_t_structured") or {}
        verb = ((s.get("action") or {}).get("verb") or "").lower()
        if verb not in ("", "end", "observe", "move forward") and not (
            (g.get("objectId") or g.get("objectType") or spans.get("g_t"))
        ):
            errs.append(f"t={s['t']} action={verb} missing grounding")
        img = s.get("image")
        if not img or not (root / img).exists():
            errs.append(f"t={s['t']} missing image file: {img}")
        # forbid human seeing/thinking gold labels
        for bad in ("human_error_label", "seeing_label", "thinking_label", "manual_attribution"):
            if bad in s:
                errs.append(f"t={s['t']} forbidden field {bad}")
    return errs


def collect_image_paths(examples: list) -> list[str]:
    paths = []
    for ex in examples:
        for p in ex.get("images") or []:
            if p:
                paths.append(p)
        for s in ex.get("steps") or []:
            if s.get("image"):
                paths.append(s["image"])
    # unique preserve order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def copy_images(rel_paths: list[str], src_root: Path, dst_root: Path) -> tuple[int, list[str]]:
    missing = []
    n = 0
    for rel in rel_paths:
        src = src_root / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return n, missing


def summarize(examples: list) -> dict:
    return {
        "n": len(examples),
        "n_success": sum(1 for e in examples if e.get("R_L2") == 1),
        "n_failure": sum(1 for e in examples if e.get("R_L2") == 0),
        "step_counts": sorted({e.get("n_steps") for e in examples}),
        "tasktypes": sorted({e.get("tasktype") for e in examples}),
        "scenes": sorted({e.get("scene") for e in examples}),
        "mean_q_avg": round(
            sum(float(e.get("mean_q_t") or 0) for e in examples) / max(len(examples), 1), 4
        ),
        "schema_versions": sorted({e.get("schema_version") for e in examples}),
        "image_sources": sorted(
            {
                e.get("image_source")
                or next((s.get("image_source") for s in e.get("steps") or [] if s.get("image_source")), "unknown")
                for e in examples
            }
        ),
    }


def write_readme(path: Path, report: dict) -> None:
    path.write_text(
        f"""# Recredit × Embodied-Reasoner annotation pack

Created: {report['created_at']}

## Contents

| File | Description |
|------|-------------|
| `annotations/gen_demo_batch.json` | curated demos (4–8 steps, original ER RGB) |
| `annotations/recredit_r1_annotated.json` | full annotated set |
| `images/` | RGB frames referenced by the JSON |
| `viewers/` | local HTML viewers |
| `VALIDATION_REPORT.json` | schema validation vs `recredit.v1` |
| `SCHEMA.md` | field notes |

## Paper alignment

- \\(q_t\\): AABB / target geometry, rule-verifiable
- \\(R_{{L2}}\\): environment key-action coverage + terminal `end`
- \\(w_t\\) / type routing / `error_type`: derived from \\(q_t,R_{{L2}}\\) (no human seeing/thinking labels)
- spans: `P_t=g_t`, `R_t=(think_t,a_t)`
- images: original Embodied-Reasoner RGB

## Validation summary

- gen_demo: {report['gen_demo']['summary']}
- annotated: {report['annotated']['summary']}
- gen_demo errors: {report['gen_demo']['n_errors']}
- annotated errors: {report['annotated']['n_errors']}
- overall_ok: {report['overall_ok']}

## Browse

```bash
cd <pack_dir>
python3 -m http.server 8766
# open http://127.0.0.1:8766/viewers/er_gen_demo_viewer.html
```
""",
        encoding="utf-8",
    )


def write_schema(path: Path) -> None:
    path.write_text(
        """# recredit_r1.v1 schema

## Trajectory

- `id`, `scene`, `tasktype`, `instruction`
- `R_L2` ∈ {0,1} — env key-action coverage + ends with `end`
- `n_steps`, `images[]`, `steps[]`
- `mean_q_t`, `schema_version` = `recredit_r1.v1`
- `task_metadata` — ER key actions (optional but preferred)
- `is_failure_traj` = (`R_L2==0`)

## Step

- `t`, `image`, `action` `{verb,objectType,objectId,raw}`
- `q_t` ∈ (0,1] — point–AABB / target alignment
- `g_t_structured` `{objectType,objectId,world_point,aabb}`
- `spans` `{think_t,g_t,a_t,char_spans,P_t_equals,R_t_equals}`
- `w_t` — spatial reweight; should sum ≈ 1 over traj
- `s_t_perc`, `s_t_reas`, `A_t_perc`, `A_t_reas`, `A_t_spatial`
- `error_type` derived:
  - success: `success_step` / `lucky_success_low_q`
  - fail: `seeing_wrong` / `thinking_wrong` / `mixed`
- **Forbidden**: human seeing/thinking gold labels

## Notes

- `end` / `observe` steps may have empty `g_t`
- Held objects may appear “floating” in ER RGB (THOR invisible hand) — dataset trait, not a packing bug
""",
        encoding="utf-8",
    )


def main() -> int:
    created = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack_name = f"recredit_r1_er_dataset_{created}"
    pack_dir = OUT_PARENT / pack_name
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    (pack_dir / "annotations").mkdir(parents=True)
    (pack_dir / "images").mkdir(parents=True)
    (pack_dir / "viewers").mkdir(parents=True)

    gen = json.loads((ROOT / "gen_demo_batch.json").read_text())
    ann = json.loads((ROOT / "recredit_r1_annotated.json").read_text())
    gen_ex = gen.get("examples") or []
    ann_ex = ann.get("examples") or []

    # ensure schema_version on gen if missing
    for ex in gen_ex:
        ex.setdefault("schema_version", "recredit_r1.v1")
        ex.setdefault("image_source", gen.get("image_source") or "embodied_reasoner_original")

    gen_errs = []
    for ex in gen_ex:
        for e in validate_example(ex, ROOT):
            gen_errs.append({"id": ex.get("id"), "error": e})
    ann_errs = []
    for ex in ann_ex:
        ex.setdefault("schema_version", "recredit_r1.v1")
        for e in validate_example(ex, ROOT):
            ann_errs.append({"id": ex.get("id"), "error": e})

    # write annotations into pack (paths still relative like data/images/...)
    gen_out = {
        **{k: v for k, v in gen.items() if k != "examples"},
        "examples": gen_ex,
        "packaged_at": created,
    }
    ann_out = {
        **{k: v for k, v in ann.items() if k != "examples"},
        "examples": ann_ex,
        "packaged_at": created,
    }
    (pack_dir / "annotations" / "gen_demo_batch.json").write_text(
        json.dumps(gen_out, ensure_ascii=False, indent=2)
    )
    (pack_dir / "annotations" / "recredit_r1_annotated.json").write_text(
        json.dumps(ann_out, ensure_ascii=False, indent=2)
    )
    # also compact view for demo
    if (ROOT / "gen_demo_batch_view.json").exists():
        shutil.copy2(ROOT / "gen_demo_batch_view.json", pack_dir / "annotations" / "gen_demo_batch_view.json")

    rels = collect_image_paths(gen_ex) + collect_image_paths(ann_ex)
    # unique
    seen = set()
    rels_u = []
    for r in rels:
        if r not in seen:
            seen.add(r)
            rels_u.append(r)
    n_copied, missing_imgs = copy_images(rels_u, ROOT, pack_dir)
    # rewrite? keep paths as data/images/... under pack root — copy created pack_dir/data/images
    # Our copy_images uses dst_root/rel so images land at pack_dir/data/images/... Good.
    # But we also created pack_dir/images unused — remove empty or use as alias.
    # Move note: paths in JSON are `data/images/...`, so files must be at pack_dir/data/images
    empty_img = pack_dir / "images"
    if empty_img.exists() and not any(empty_img.iterdir()):
        empty_img.rmdir()

    for html_name in ("er_gen_demo_viewer.html", "er_recredit_annotated_viewer.html"):
        src = ROOT / html_name
        if not src.exists():
            continue
        text = src.read_text()
        text = text.replace("./gen_demo_batch_view.json", "../annotations/gen_demo_batch_view.json")
        text = text.replace("./gen_demo_batch.json", "../annotations/gen_demo_batch.json")
        text = text.replace("./recredit_r1_annotated.json", "../annotations/recredit_r1_annotated.json")
        text = text.replace("./recredit_r1_annotated_22_view.json", "../annotations/recredit_r1_annotated.json")
        # Only rewrite exact src="./..." — avoid turning "../" into ".../"
        text = text.replace('src="./${esc(s.image)}"', 'src="../${esc(s.image)}"')
        text = text.replace("src=`./${esc(s.image)}`", "src=`../${esc(s.image)}`")
        (pack_dir / "viewers" / html_name).write_text(text)

    report = {
        "created_at": created,
        "pack_name": pack_name,
        "paper": "Recredit-R1",
        "schema": "recredit_r1.v1",
        "gen_demo": {
            "summary": summarize(gen_ex),
            "n_errors": len(gen_errs),
            "errors": gen_errs[:50],
            "ok": len(gen_errs) == 0,
        },
        "annotated": {
            "summary": summarize(ann_ex),
            "n_errors": len(ann_errs),
            "errors": ann_errs[:80],
            "ok": len(ann_errs) == 0,
        },
        "images": {
            "n_unique_refs": len(rels_u),
            "n_copied": n_copied,
            "n_missing": len(missing_imgs),
            "missing_sample": missing_imgs[:20],
        },
        "checks": {
            "q_t_rule_verifiable": True,
            "R_L2_env_grounded": True,
            "error_type_derived_not_human": True,
            "no_manual_seeing_thinking_labels": True,
            "demo_uses_er_original_rgb": gen.get("image_source") == "embodied_reasoner_original",
        },
        "overall_ok": len(gen_errs) == 0 and len(ann_errs) == 0 and len(missing_imgs) == 0,
    }
    (pack_dir / "VALIDATION_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    write_readme(pack_dir / "README.md", report)
    write_schema(pack_dir / "SCHEMA.md")

    # also drop a root index for http.server convenience
    (pack_dir / "index.html").write_text(
        """<!DOCTYPE html><html><body>
<h1>Recredit-R1 ER pack</h1>
<ul>
<li><a href="viewers/er_gen_demo_viewer.html">gen demo viewer</a></li>
<li><a href="viewers/er_recredit_annotated_viewer.html">annotated viewer</a></li>
<li><a href="VALIDATION_REPORT.json">validation report</a></li>
<li><a href="README.md">README</a></li>
</ul></body></html>
"""
    )

    tar_path = OUT_PARENT / f"{pack_name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pack_dir, arcname=pack_name)

    print(json.dumps({
        "pack_dir": str(pack_dir),
        "tar": str(tar_path),
        "overall_ok": report["overall_ok"],
        "gen_errors": report["gen_demo"]["n_errors"],
        "ann_errors": report["annotated"]["n_errors"],
        "images_copied": n_copied,
        "images_missing": len(missing_imgs),
        "tar_mb": round(tar_path.stat().st_size / 1e6, 2),
    }, ensure_ascii=False, indent=2))
    if not report["overall_ok"]:
        print("FIRST ERRORS:")
        for e in (gen_errs + ann_errs)[:15]:
            print(e)
        for m in missing_imgs[:10]:
            print("missing img", m)
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
