#!/usr/bin/env python3
"""Integrity gate for the campaign output. Run it before anything trains on this.

Every check here is one that a long unattended run can fail silently:

  pairing      a mask with no image (or the reverse) is a frame that will either crash a
               loader or be dropped without a count
  class ids    a value outside the taxonomy means a composition bug, and it would train
               as a real class
  emptiness    an all-IGNORE mask contributes nothing but occupies a split; a mask that
               is 97% one class usually means that clip's plate went wrong
  duplicates   the same image bytes under two names is a leak between splits when a clip
               was pulled twice under different names

Exits non-zero if anything failed, so it can gate a training run.
"""
import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

VALID = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255}
IGNORE = 255


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("datasets/site30k_v1"))
    ap.add_argument("--sample", type=int, default=0,
                    help="check only N masks, spread evenly (0 = all)")
    ap.add_argument("--hash-images", action="store_true",
                    help="also hash every image to find duplicates (slow)")
    args = ap.parse_args()

    all_masks = sorted((args.root / "masks").glob("*.png"))
    images = {p.stem for p in (args.root / "images").glob("*.jpg")}
    masks = all_masks
    if args.sample and len(masks) > args.sample:
        step = len(masks) / args.sample
        masks = [masks[int(i * step)] for i in range(args.sample)]

    missing_image = [m.stem for m in masks if m.stem not in images]
    orphan_image = sorted(images - {m.stem for m in all_masks})
    bad_ids: list[tuple[str, set]] = []
    empty: list[str] = []
    lopsided: list[tuple[str, float]] = []
    per_camera: Counter = Counter()
    labelled: list[float] = []

    for m in masks:
        a = np.asarray(Image.open(m))
        ids = set(np.unique(a).tolist())
        if not ids <= VALID:
            bad_ids.append((m.stem, ids - VALID))
        lab = float((a != IGNORE).mean())
        labelled.append(lab)
        if lab < 0.05:
            empty.append(m.stem)
        else:
            top = max((a == c).mean() for c in ids if c != IGNORE)
            if top > 0.97:
                lopsided.append((m.stem, round(100 * top, 1)))
        per_camera[m.stem.split("__")[0]] += 1

    dupes: list[tuple[str, str]] = []
    if args.hash_images:
        seen: dict[str, str] = {}
        for p in sorted((args.root / "images").glob("*.jpg")):
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h in seen:
                dupes.append((seen[h], p.stem))
            else:
                seen[h] = p.stem

    print(f"{len(masks)} masks checked in {args.root}")
    print(f"  labelled share: mean {100 * np.mean(labelled):.1f}%  "
          f"min {100 * np.min(labelled):.1f}%  max {100 * np.max(labelled):.1f}%")
    print("  per camera: " + ", ".join(f"{c} {n}" for c, n in sorted(per_camera.items())))
    problems = 0
    for name, items in (("masks with no image", missing_image),
                        ("images with no mask", orphan_image),
                        ("masks with ids outside the taxonomy", bad_ids),
                        ("masks under 5% labelled", empty),
                        ("masks over 97% one class", lopsided),
                        ("duplicate image bytes", dupes)):
        if items:
            problems += len(items)
            print(f"  !! {name}: {len(items)}")
            for it in items[:5]:
                print(f"       {it}")
    if not problems:
        print("  no problems found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
