#!/usr/bin/env python3
"""CrowdPose -> the `keypoints_{split}.json` layout `PoseKeypointsDataset` reads.

Writes a dataset root the existing loader can open unchanged:

    <out>/
      annotations/keypoints_{train,val}.json
      images/  -> symlink to the CrowdPose images directory

`PoseKeypointsDataset` hard-codes `reshape(17, 3)`, so the conversion happens here
rather than in the loader: one file to audit, and no branch in the hot path that could
apply the wrong schema to `site_pose`.

The 14 -> 17 mapping and the three traps it has to respect are documented in
`syncai_hydranet.data.label_maps_crowdpose`. The short version: 12 joints map by name,
`head` and `neck` are dropped rather than forced into `nose`, and the five face channels
arrive as v=0 -- which only means "not labelled" if the loss masks the whole channel.

Usage:
    python tools/pose/crowdpose_to_coco17.py \\
        --src /data/open_data/CrowdPose --out datasets/crowdpose_c17
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from syncai_hydranet.data.label_maps_crowdpose import to_coco17  # noqa: E402


def convert(src: Path, out: Path, splits: tuple[str, ...]) -> None:
    images_dir = src / "images"
    if not images_dir.is_dir():
        raise SystemExit(f"{images_dir} missing -- unzip images.zip first")

    (out / "annotations").mkdir(parents=True, exist_ok=True)
    link = out / "images"
    if not link.exists():
        # Symlink, not a copy: 20,000 JPEGs are the same bytes either way and the
        # source tree stays the one place they live.
        link.symlink_to(images_dir)

    for split in splits:
        ann_in = src / "annotations" / f"crowdpose_{split}.json"
        if not ann_in.is_file():
            print(f"  {split:9} skipped -- {ann_in.name} not present")
            continue
        data = json.loads(ann_in.read_text())

        kept, dropped_empty = [], 0
        for a in data["annotations"]:
            kp = to_coco17(a["keypoints"])[0]
            # A person with no labelled joint supervises nothing and still costs a
            # heatmap render. CrowdPose has these: fully-occluded people kept for the
            # box annotations.
            if not (kp[:, 2] > 0).any():
                dropped_empty += 1
                continue
            kept.append(
                {
                    "id": a["id"],
                    "image_id": a["image_id"],
                    "category_id": 1,
                    "bbox": [float(v) for v in a["bbox"]],
                    "iscrowd": a.get("iscrowd", 0),
                    "num_keypoints": int((kp[:, 2] > 0).sum()),
                    "keypoints": [float(v) for v in kp.reshape(-1)],
                }
            )

        with_ann = {a["image_id"] for a in kept}
        payload = {
            "info": {
                "description": f"CrowdPose {split} remapped to COCO-17",
                "source": str(ann_in),
                "note": (
                    "nose/eyes/ears are v=0 (not labelled), not absent. "
                    "The loss must mask those channels; see label_maps_crowdpose."
                ),
            },
            "categories": data["categories"],
            "images": [im for im in data["images"] if im["id"] in with_ann],
            "annotations": kept,
        }
        dst = out / "annotations" / f"keypoints_{split}.json"
        dst.write_text(json.dumps(payload))

        v = np.concatenate([np.asarray(a["keypoints"]).reshape(-1, 3)[:, 2] for a in kept])
        print(
            f"  {split:9} {len(payload['images']):>6} images  {len(kept):>7} people  "
            f"(dropped {dropped_empty} with no labelled joint)  "
            f"v0={int((v == 0).sum())} v1={int((v == 1).sum())} v2={int((v == 2).sum())}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("/data/open_data/CrowdPose"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    a = p.parse_args()
    convert(a.src, a.out, tuple(a.splits))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
