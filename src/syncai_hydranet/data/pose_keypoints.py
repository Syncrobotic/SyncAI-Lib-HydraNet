"""Teacher keypoints as a dataset: `keypoints_{split}.json` -> per-image (N, 17, 3).

The file is what `tools/pose/vitpose_teacher.py` writes -- ViTPose over the Gold boxes,
image ids inherited from `instances_all_{split}.json` so the by-camera split discipline
carries over without a second bookkeeping system. Keypoints ride the same geometry the
boxes do (`transforms.py` scales, shifts and flips them together), and arrive at the
loss in network-input pixels, which is the coordinate frame `PoseHeatmapLoss.render`
expects.

The teacher's boxes ride along as `targets["boxes"]`. They supervise nothing -- this
dataset declares `pose` alone -- but validation cannot score keypoints without them:
`decode_boxes` reads person i's heatmap window out of box i, and scoring against boxes
the detection head produced would move the pose curve every time detection moved. They
are parallel arrays with the keypoints, one row per person, and `transforms._paste`
drops both together when a crop loses a person.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import GEOM_IDENTITY, Sample, build_transforms


class PoseKeypointsDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str,
        split: str,
        input_size,
        train: bool,
        supervises=("pose",),
        letterbox: bool = False,
        augment: dict | None = None,
        head_name: str = "pose",
    ):
        self.root = Path(root)
        ann_file = self.root / "annotations" / f"keypoints_{split}.json"
        if not ann_file.is_file():
            raise FileNotFoundError(
                f"{ann_file} does not exist -- run tools/pose/vitpose_teacher.py {split}"
            )
        data = json.loads(ann_file.read_text())
        self.img_dir = self.root / "images"
        by_image: dict[int, list] = {}
        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            by_image.setdefault(a["image_id"], []).append(
                (
                    np.asarray(a["keypoints"], dtype=np.float32).reshape(17, 3),
                    np.asarray([x, y, x + w, y + h], dtype=np.float32),
                )
            )
        self.entries = [
            (
                im["file_name"],
                np.stack([kp for kp, _ in by_image[im["id"]]]),
                np.stack([bx for _, bx in by_image[im["id"]]]),
                im["id"],
            )
            for im in data["images"]
            if im["id"] in by_image
        ]
        self.head_name = head_name
        self.supervises = list(supervises)
        self.transform = build_transforms(
            input_size, train, letterbox=letterbox, augment=augment
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int):
        file_name, kps, boxes, img_id = self.entries[index]
        img = Image.open(self.img_dir / file_name).convert("RGB")
        # `labels` is all zeros and means nothing: the transforms index it alongside
        # `boxes` when a crop drops a person, so the pair has to exist even though this
        # dataset supervises no detection head.
        s = Sample(
            image=img,
            pose=kps.copy(),
            boxes=boxes.copy(),
            labels=np.zeros(len(boxes), dtype=np.int64),
        )
        s = self.transform(s)
        return {
            "image": s["image"],
            "targets": {
                self.head_name: torch.from_numpy(s["pose"]).float(),
                "boxes": torch.from_numpy(np.asarray(s["boxes"], dtype=np.float32)),
            },
            "supervises": self.supervises,
            "image_id": img_id,
            "geom": s.get("geom", GEOM_IDENTITY),
        }
