"""AI-reviewed instance subsets with explicit empty regions, never implicit background."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .store_split import camera_store, fold_split
from .studioa_autolabel import decode
from .studioa_review import digest, write_json
from .studioa_supervision import _relative, check_supervision
from .transforms import Sample, build_transforms

SCHEMA = "studioa.partial-instances.v1"
CLASSES = (
    "laptop",
    "phone",
    "tablet",
    "boxed_stock",
    "cardboard_box",
    "speaker",
    "poster",
    "fire_equipment",
    "chair",
    "person",
)


def export_instances(source: Path, reviews: Path, out: Path) -> dict:
    """Export only source train/val frames explicitly inspected by the AI reviewer."""
    manifest = check_supervision(source)
    review = json.loads(reviews.read_text())
    if review.get("reviewer_kind") != "ai" or review.get("source_manifest_sha256") != digest(
        source / "manifest.json"
    ):
        raise ValueError("instance review needs bound source and AI provenance")
    held_out = review["held_out"]
    source_frames = {f["id"]: f for f in manifest["frames"]}
    assignments = manifest["folds"][held_out]["assignments"]
    prepared = []
    seen = set()
    for item in review["frames"]:
        fid = item["frame_id"]
        if fid in seen or fid not in source_frames or assignments[fid] not in ("train", "val"):
            raise ValueError("duplicate, unknown or held-out/excluded instance review frame")
        seen.add(fid)
        frame = source_frames[fid]
        if item["image_sha256"] != frame["image_sha256"] or item["companion_sha256"] != digest(
            source / frame["companion"]
        ):
            raise ValueError("instance review image/annotation changed")
        data = json.loads((source / frame["companion"]).read_text())
        candidates = {r["id"]: r for r in data["entities"] + data["unresolved_candidates"]}
        w, h = frame["image_size_px"]
        occupied = np.zeros((h, w), dtype=bool)
        boxes, labels, identities = [], [], []
        for decision in item["positives"]:
            identity, entity = decision["id"], decision["entity"]
            if identity in identities or entity not in CLASSES or not decision.get("reason"):
                raise ValueError("invalid or duplicate instance decision")
            row = candidates[identity]
            if entity not in {row["entity"], *row.get("possible_entities", [])}:
                raise ValueError("reviewed class not supported by source proposal")
            mask = decode(row["segmentation"])
            ys, xs = np.where(mask)
            if not len(xs):
                raise ValueError("empty reviewed instance")
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
            labels.append(CLASSES.index(entity))
            identities.append(identity)
            occupied |= mask
        negative = np.zeros((h, w), dtype=np.uint8)
        for region in item["negative_rects"]:
            rect = region["xyxy"]
            if (
                len(rect) != 4
                or any(type(v) is not int for v in rect)
                or not region.get("reason")
            ):
                raise ValueError("negative region needs integer bounds and review reason")
            x0, y0, x1, y1 = rect
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError("negative region outside image")
            negative[y0:y1, x0:x1] = 1
        if (occupied & (negative == 1)).any():
            raise ValueError("reviewed empty region overlaps a reviewed object")
        prepared.append((frame, boxes, labels, identities, negative))
    if not prepared:
        raise ValueError("empty instance review")
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("images", "annotations", "companions"):
        (out / folder).mkdir()
    shutil.copyfile(reviews, out / "reviews.json")
    shutil.copyfile(__file__, out / "producer.py")
    frames = []
    counts = Counter(dict.fromkeys(CLASSES, 0))
    for frame, boxes, labels, identities, negative in prepared:
        fid = frame["id"]
        shutil.copyfile(source / frame["image"], out / frame["image"])
        shutil.copyfile(source / frame["companion"], out / frame["companion"])
        target_name, negative_name = f"annotations/{fid}.json", f"annotations/{fid}.png"
        write_json(
            out / target_name, {"boxes": boxes, "labels": labels, "source_ids": identities}
        )
        Image.fromarray(negative).save(out / negative_name)
        counts.update(CLASSES[i] for i in labels)
        frames.append(
            {
                **{
                    k: frame[k]
                    for k in (
                        "id",
                        "camera",
                        "store",
                        "original_split",
                        "image_size_px",
                        "image",
                        "image_sha256",
                        "pixel_sha256",
                        "companion",
                    )
                },
                "targets": target_name,
                "negative_mask": negative_name,
                "negative_pixels": int(negative.sum()),
                "instances": len(labels),
            }
        )
    result = {
        "schema": SCHEMA,
        "classes": list(CLASSES),
        "held_out": held_out,
        "frames": frames,
        "folds": {held_out: {"assignments": {f["id"]: assignments[f["id"]] for f in frames}}},
        "source_manifest_sha256": digest(source / "manifest.json"),
        "review_sha256": digest(reviews),
        "instances_by_class": dict(counts),
        "policy": (
            "positive assigned channel only; negatives only in AI-reviewed empty regions; "
            "unknown and padding ignored"
        ),
        "exhaustive_labels": False,
        "independent_accuracy": False,
    }
    write_json(out / "manifest.json", result)
    write_json(
        out / "report.json",
        {
            "status": "completed",
            "outputs": {
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        },
    )
    return result


def check_instances(root: Path) -> dict:
    report = json.loads((root / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("incomplete instance package")
    for name, expected in report["outputs"].items():
        if digest(_relative(root, name)) != expected:
            raise ValueError(f"instance package changed: {name}")
    m = json.loads((root / "manifest.json").read_text())
    if m["schema"] != SCHEMA or m["classes"] != list(CLASSES) or m["exhaustive_labels"]:
        raise ValueError("invalid partial-instance contract")
    seen, cameras, pixels = set(), {}, {}
    assignments = m["folds"][m["held_out"]]["assignments"]
    for f in m["frames"]:
        split = assignments[f["id"]]
        if f["id"] in seen or camera_store(f["id"]) != (f["camera"], f["store"]):
            raise ValueError("invalid instance frame identity")
        seen.add(f["id"])
        if split not in ("train", "val") or split != fold_split(
            f["store"], f["original_split"], m["held_out"]
        ):
            raise ValueError("instance split role changed")
        for mapping, key in ((cameras, f["camera"]), (pixels, f["pixel_sha256"])):
            if mapping.setdefault(key, split) != split:
                raise ValueError("instance camera/content split leak")
    return m


class StudioAInstanceDataset(Dataset):
    partial_detection = True

    def __init__(
        self,
        root: Path,
        held_out: str,
        split: str,
        input_size,
        *,
        train=False,
        augment=None,
        partial_eval=None,
    ):
        m = check_instances(root)
        if (
            held_out != m["held_out"]
            or split not in ("train", "val")
            or (train and split != "train")
        ):
            raise ValueError("invalid instance fold/split/augmentation")
        self.root = root
        if partial_eval not in (None, "reviewed_regions_v1"):
            raise ValueError("unsupported partial evaluation protocol")
        self.partial_detection_evaluation = partial_eval
        self.detection_classes = CLASSES
        self.supervises = ["detection"]
        self.frames = [
            f for f in m["frames"] if m["folds"][held_out]["assignments"][f["id"]] == split
        ]
        if not self.frames:
            raise ValueError("empty reviewed instance split")
        self.transform = build_transforms(
            input_size, train=train, letterbox=True, augment=augment
        )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        f = self.frames[index]
        targets = json.loads((self.root / f["targets"]).read_text())
        with Image.open(self.root / f["image"]) as image:
            rgb = image.convert("RGB")
        with Image.open(self.root / f["negative_mask"]) as image:
            negative = np.array(image)
        sample = self.transform(
            Sample(
                image=rgb,
                masks={"det_negative_mask": negative},
                boxes=np.asarray(targets["boxes"], dtype=np.float32).reshape(-1, 4),
                labels=np.asarray(targets["labels"], dtype=np.int64),
            )
        )
        return {
            "image": sample["image"],
            "supervises": self.supervises,
            "targets": {
                **sample["masks"],
                "boxes": sample["boxes"],
                "labels": sample["labels"],
            },
        }
