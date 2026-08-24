#!/usr/bin/env python3
"""Give the campaign's flat output the layout `SegFolderDataset` reads.

The campaign writes one flat store -- `masks/` and `images/` -- because a camera-date unit
is the unit of work and a worker should not have to know about splits. The two loaders
this dataset feeds want two different layouts, and they disagree about where images live::

    SegFolderDataset   root/images/<split>/**  +  root/annotations/<split>/**
    CocoDetDataset     root/<split>/**         +  root/annotations/instances_<split>.json

so this builds BOTH with relative SYMLINKS into the flat store. No bytes are copied and
the flat store stays the single source: a mask fixed in place is fixed in every split that
points at it, and a rebuild after a resume costs nothing. Building only one of the two
layouts is how a multitask dataset ends up training its segmentation head on nine cameras
and its detection head on nothing.

    python tools/site30k/materialize_splits.py --root datasets/site30k_v1
    python tools/site30k/materialize_splits.py --root datasets/site30k_v1 \
        --exclude Tao-Hsin-cam03

The assignment is inherited per CAMERA from `datasets/site30k/split.json` and is not a
decision this script makes. A camera missing from that file is an error and not a default:
`train` is the wrong guess to make silently, because a camera quietly landing in train is
a camera that can never be used to measure anything.

It also prints the camera identities per split, not only the counts. A split's frame count
answers "is there enough"; only the camera list answers "can this measure anything", and
that is the question that comes next -- site30k_v1's test split is one camera.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
# `<camera>__<slot>__<frame>` -- the campaign's own naming, see recipe.py.
STEM = re.compile(r"^(?P<camera>.+?)__(?P<slot>\d{8}-\d{6})__(?P<frame>\d+)$")


def link(src: Path, dst: Path) -> bool:
    """Point dst at src by a relative symlink. True if anything changed."""
    rel = os.path.relpath(src, dst.parent)
    if dst.is_symlink():
        if dst.readlink() == Path(rel):
            return False
        dst.unlink()
    elif dst.exists():
        raise SystemExit(f"{dst} exists and is not a symlink; refusing to replace it")
    dst.symlink_to(rel)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("datasets/site30k_v1"))
    ap.add_argument("--split-json", type=Path, default=Path("datasets/site30k/split.json"))
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="camera to leave out of every split; repeatable. Use it for a camera whose "
        "labels are known bad -- excluding is a decision, so it is recorded in the "
        "manifest rather than done by deleting files.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    assign = json.loads(args.split_json.read_text())["assign"]
    masks = sorted((args.root / "masks").glob("*.png"))
    if not masks:
        raise SystemExit(f"no masks under {args.root / 'masks'}")

    by_split: dict[str, list[str]] = defaultdict(list)
    cameras: dict[str, set[str]] = defaultdict(set)
    unknown: set[str] = set()
    excluded = 0
    no_image = []

    for m in masks:
        hit = STEM.match(m.stem)
        if not hit:
            raise SystemExit(f"unparseable mask name: {m.name}")
        cam = hit["camera"]
        if cam in args.exclude:
            excluded += 1
            continue
        split = assign.get(cam)
        if split is None:
            unknown.add(cam)
            continue
        if not (args.root / "images" / f"{m.stem}.jpg").exists():
            no_image.append(m.stem)
            continue
        by_split[split].append(m.stem)
        cameras[split].add(cam)

    if unknown:
        raise SystemExit(
            f"no split assignment for {sorted(unknown)} in {args.split_json}. "
            "Assign them there; this script will not guess."
        )
    if no_image:
        raise SystemExit(
            f"{len(no_image)} masks have no image, first: {no_image[:3]}. "
            "Run tools/site30k/check_dataset.py -- a half-written pair is a campaign bug."
        )

    changed = 0
    for split, stems in sorted(by_split.items()):
        img_dir = args.root / "images" / split  # SegFolderDataset
        ann_dir = args.root / "annotations" / split
        det_dir = args.root / split  # CocoDetDataset
        if not args.dry_run:
            for d in (img_dir, ann_dir, det_dir):
                d.mkdir(parents=True, exist_ok=True)
            for stem in stems:
                changed += link(args.root / "images" / f"{stem}.jpg", img_dir / f"{stem}.jpg")
                changed += link(args.root / "masks" / f"{stem}.png", ann_dir / f"{stem}.png")
                changed += link(args.root / "images" / f"{stem}.jpg", det_dir / f"{stem}.jpg")

    total = sum(len(v) for v in by_split.values())
    print(f"{len(masks)} masks -> {total} linked, {excluded} excluded, {changed} links written")
    for split in SPLITS:
        stems = by_split.get(split, [])
        cams = sorted(cameras.get(split, ()))
        share = 100 * len(stems) / total if total else 0.0
        print(
            f"  {split:5s} {len(stems):6d} frames  {share:5.1f}%  "
            f"{len(cams)} cameras: {', '.join(cams) or '-'}"
        )

    if len(cameras.get("test", ())) < 2:
        print(
            "\n  !! the test split is fewer than two cameras. A single-camera test split "
            "measures that camera, not the model, and it is the split that cannot be "
            "repaired after the fact.",
            file=sys.stderr,
        )

    manifest = {
        "root": str(args.root),
        "split_json": str(args.split_json),
        "excluded_cameras": sorted(args.exclude),
        "frames": {s: len(by_split.get(s, [])) for s in SPLITS},
        "cameras": {s: sorted(cameras.get(s, ())) for s in SPLITS},
    }
    if not args.dry_run:
        (args.root / "splits.json").write_text(json.dumps(manifest, indent=1))
        print(f"\nwrote {args.root / 'splits.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
