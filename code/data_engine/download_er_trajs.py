#!/usr/bin/env python3
"""Download selected ER trajectory RGB from HF mirror zips (selective extract)."""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HF_MIRROR = "https://huggingface.co/datasets/zwq2018/embodied_reasoner/resolve/main"


def traj_key_from_item(item: dict) -> tuple[str, str] | None:
    imgs = item.get("images") or []
    if not imgs:
        return None
    parts = str(imgs[0]).replace("\\", "/").split("/")
    if "images" not in parts:
        return None
    j = parts.index("images")
    if len(parts) <= j + 2:
        return None
    return parts[j + 1], parts[j + 2]


def local_have(img_root: Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    if not img_root.exists():
        return out
    for tt_dir in img_root.iterdir():
        if not tt_dir.is_dir():
            continue
        for traj_dir in tt_dir.iterdir():
            if traj_dir.is_dir() and list(traj_dir.glob("*.png")):
                out.add((tt_dir.name, traj_dir.name))
    return out


def select_candidates(
    train: list[dict],
    meta_dir: Path,
    have: set[tuple[str, str]],
    *,
    limit: int,
    prefer_tasktypes: set[str] | None = None,
) -> list[dict]:
    meta = {p.stem for p in meta_dir.glob("FloorPlan*.json")}
    cands: list[tuple[int, tuple[str, str], int]] = []
    for i, item in enumerate(train):
        k = traj_key_from_item(item)
        if not k or k in have:
            continue
        m = re.match(r"(FloorPlan\d+)", k[1])
        if not m or m.group(1) not in meta:
            continue
        if prefer_tasktypes and k[0] not in prefer_tasktypes:
            continue
        cands.append((i, k, len(item.get("images") or [])))

    if prefer_tasktypes:
        by_tt: dict[str, list] = defaultdict(list)
        for c in cands:
            by_tt[c[1][0]].append(c)
        selected: list[tuple[int, tuple[str, str], int]] = []
        for tt in sorted(prefer_tasktypes, key=lambda t: len(by_tt.get(t, []))):
            xs = sorted(by_tt.get(tt, []), key=lambda x: abs(x[2] - 6))
            selected.extend(xs)
        selected = selected[:limit]
    else:
        # round-robin across tasktypes for diversity
        by_tt: dict[str, list] = defaultdict(list)
        for c in cands:
            by_tt[c[1][0]].append(c)
        for tt in by_tt:
            by_tt[tt] = sorted(by_tt[tt], key=lambda x: abs(x[2] - 6))
        tts = sorted(by_tt.keys(), key=lambda t: len(by_tt[t]))
        selected = []
        while len(selected) < limit:
            added = False
            for tt in tts:
                if by_tt[tt] and len(selected) < limit:
                    selected.append(by_tt[tt].pop(0))
                    added = True
            if not added:
                break

    return [
        {"index": i, "tasktype": k[0], "traj": k[1], "n_images": n}
        for i, k, n in selected
    ]


def download_zip(tasktype: str, zip_dir: Path) -> Path:
    zip_dir.mkdir(parents=True, exist_ok=True)
    name = f"{tasktype}.zip"
    dest = zip_dir / name
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    url = f"{HF_MIRROR}/data/images/{name}"
    print(f"downloading {url} -> {dest}")
    dest_part = dest.with_suffix(".zip.part")
    subprocess.run(
        ["curl", "-fL", "--retry", "3", "-C", "-", "-o", str(dest_part), url],
        check=True,
    )
    dest_part.rename(dest)
    return dest


def extract_trajs(zip_path: Path, tasktype: str, trajs: set[str], img_root: Path) -> tuple[int, int]:
    """HF zips use `{tasktype}/{traj}/*.png` (no `data/images/` prefix)."""
    prefixes = (f"{tasktype}/", f"data/images/{tasktype}/")
    n_files = 0
    done_trajs: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".png") or "_mix_res" in name:
                continue
            if not any(name.startswith(p) for p in prefixes):
                continue
            rel_parts = name.split("/")
            traj = rel_parts[-2]
            if traj not in trajs:
                continue
            dest = img_root / tasktype / traj / Path(name).name
            if dest.exists() and dest.stat().st_size > 0:
                n_files += 1
                done_trajs.add(traj)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))
            n_files += 1
            done_trajs.add(traj)
    return len(done_trajs), n_files


def main() -> int:
    p = argparse.ArgumentParser(description="Download ER trajectory images from HF mirror zips")
    p.add_argument("--er-root", type=Path, default=ROOT)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument(
        "--broad",
        action="store_true",
        help="Select across all tasktypes (not only small zips); use for limit>50",
    )
    p.add_argument("--manifest", type=Path, default=None, help="JSON manifest from prior selection")
    p.add_argument("--zip-dir", type=Path, default=ROOT / "zips")
    args = p.parse_args()

    er_root = args.er_root
    train = json.loads((er_root / "train_multiturn_9390.json").read_text())
    img_root = er_root / "data" / "images"
    meta_dir = er_root / "scene_metadata"
    have = local_have(img_root)

    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
    else:
        small = {
            "single_search_from_closerep_open_fault",
            "single_toggle_navigate_fault",
            "pickup_and_put_navigate_fault",
            "navigate1open1pickup0",
            "pickup_from_closerep_and_put_close_fault",
            "pickup_from_closerep_and_put_in_closerep_open_fault",
            "pickup_from_closerep_and_put_in_closerep_close_fault",
        }
        use_broad = args.broad or args.limit > 80
        manifest = select_candidates(
            train,
            meta_dir,
            have,
            limit=args.limit,
            prefer_tasktypes=None if use_broad else small,
        )
        out_manifest = er_root / "zips" / f"download_{args.limit}_manifest.json"
        out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"wrote manifest {out_manifest} ({len(manifest)} trajs)")

    by_tt: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        by_tt[row["tasktype"]].add(row["traj"])

    report = {"requested": len(manifest), "by_tasktype": {}, "extracted_trajs": []}
    for tasktype, trajs in sorted(by_tt.items()):
        zpath = download_zip(tasktype, args.zip_dir)
        n_traj, n_files = extract_trajs(zpath, tasktype, trajs, img_root)
        report["by_tasktype"][tasktype] = {"zip": str(zpath), "n_traj": n_traj, "n_files": n_files}
        report["extracted_trajs"].extend(sorted(trajs))
        print(f"extracted {tasktype}: {n_traj}/{len(trajs)} trajs, {n_files} pngs")

    report["n_local_trajs"] = len(local_have(img_root))
    status_path = er_root / "zips" / "DOWNLOAD_STATUS.json"
    status = {}
    if status_path.exists():
        status = json.loads(status_path.read_text())
    status.update(
        {
            "last_batch": report,
            "note": f"download_er_trajs limit={args.limit}",
        }
    )
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
