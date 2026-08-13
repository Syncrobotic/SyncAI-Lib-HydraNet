"""Turn ADEChallengeData2016 into the seg_folder layout, keeping indoor scenes only.

    hydranet-prepare-ade20k --src datasets/ADEChallengeData2016 --dst datasets/ADE20K

ADE20K ships 20k images spanning 1055 scene categories, most of which are outdoor.
Rather than curating a category whitelist, this filters on annotation content: keep
frames where walkable floor is clearly visible and sky is not. That selects exactly the
ground-level indoor viewpoint a quadruped sees, and it stays correct even for scenes
whose category label is "misc".

Output uses symlinks, so it costs no extra disk and re-running is cheap.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

# ADE20K class ids used for the filter (verified against objectInfo150.txt).
# Deliberately narrower than the label map's floor_hard: road/sidewalk/path also map to
# walkable ground, but selecting on them lets street scenes through. Indoor floor and
# rug are the reliable signal that a frame was shot inside.
FLOOR_IDS = (4, 29)  # floor, rug
SKY_ID = 3
VEGETATION_IDS = (5, 10, 18, 17, 35)  # tree, grass, plant, mountain, rock

SPLITS = {"training": "train", "validation": "val"}


def score(ann_path: Path) -> tuple[float, float, float]:
    """Return the floor, sky and vegetation pixel fractions for one annotation."""
    raw = np.asarray(Image.open(ann_path))
    if raw.ndim == 3:
        raw = raw[..., 0]
    counts = np.bincount(raw.ravel(), minlength=151).astype(np.float64)
    total = max(counts.sum(), 1.0)
    return (
        counts[list(FLOOR_IDS)].sum() / total,
        counts[SKY_ID] / total,
        counts[list(VEGETATION_IDS)].sum() / total,
    )


def _score_one(p: str):
    return (p, *score(Path(p)))


def load_scene_categories(src: Path) -> dict[str, str]:
    f = src / "sceneCategories.txt"
    if not f.is_file():
        return {}
    out = {}
    with f.open(encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter=" "):
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-prepare-ade20k", description=__doc__)
    ap.add_argument("--src", required=True, help="extracted ADEChallengeData2016 directory")
    ap.add_argument("--dst", required=True, help="output root, e.g. datasets/ADE20K")
    ap.add_argument(
        "--min-floor", type=float, default=0.08, help="minimum floor pixel fraction"
    )
    ap.add_argument("--max-sky", type=float, default=0.02, help="maximum sky pixel fraction")
    ap.add_argument(
        "--max-vegetation", type=float, default=0.05, help="maximum vegetation pixel fraction"
    )
    ap.add_argument("--workers", type=int, default=0, help="0 means os.cpu_count() - 2")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if not (src / "images").is_dir():
        raise SystemExit(f"{src}/images not found; point --src at the extracted dataset")
    scenes = load_scene_categories(src)
    workers = args.workers or max((os.cpu_count() or 4) - 2, 1)

    for ade_split, out_split in SPLITS.items():
        anns = sorted((src / "annotations" / ade_split).glob("*.png"))
        if not anns:
            print(f"{ade_split}: no annotations, skipping")
            continue

        with ProcessPoolExecutor(max_workers=workers) as ex:
            scored = list(ex.map(_score_one, [str(p) for p in anns], chunksize=64))

        img_dir = dst / "images" / out_split
        ann_dir = dst / "annotations" / out_split
        for d in (img_dir, ann_dir):
            d.mkdir(parents=True, exist_ok=True)
            for old in d.iterdir():
                old.unlink()

        kept, kept_scenes = 0, Counter()
        for path_str, floor, sky, veg in scored:
            if floor < args.min_floor or sky > args.max_sky or veg > args.max_vegetation:
                continue
            ann = Path(path_str)
            img = src / "images" / ade_split / f"{ann.stem}.jpg"
            if not img.is_file():
                continue
            (img_dir / img.name).symlink_to(img)
            (ann_dir / ann.name).symlink_to(ann)
            kept += 1
            kept_scenes[scenes.get(ann.stem, "unknown")] += 1

        pct = 100 * kept / len(scored)
        print(f"{ade_split} -> {out_split}: kept {kept}/{len(scored)} ({pct:.1f}%)")
        top = ", ".join(f"{k}={v}" for k, v in kept_scenes.most_common(8))
        print(f"  top scenes: {top}")

    print(f"\nwrote {dst}")
    print("point the config at it with:  root: " + str(dst))


if __name__ == "__main__":
    main()
