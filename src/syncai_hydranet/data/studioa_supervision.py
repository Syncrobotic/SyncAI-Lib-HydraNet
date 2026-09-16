"""Source-bound partial semantic supervision; unknown pixels are never background."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .store_split import STORES, camera_store, fold_split, validate_fold
from .studioa_autolabel import decode, validate_annotation
from .studioa_contract import ENTITY_NAMES
from .studioa_review import digest, write_json
from .transforms import Sample, build_transforms

SCHEMA = "studioa.partial-semantic.v1"
CLASSES = {name: index for index, name in enumerate(ENTITY_NAMES)}
IGNORE = 255


def supervision_policy() -> dict:
    return {
        "schema": SCHEMA,
        "classes": CLASSES,
        "ignore_index": IGNORE,
        "unknown": "ignore; no background class or absence labels",
        "overlaps": "different positive classes or any unresolved mask -> ignore",
        "display_table": "round and long merchandise display tables",
        "display_cabinet": "display cabinets and upright merchandise racks",
        "legacy_uncertain_aliases": {"other_shelf": "display_cabinet"},
        "alias_promotes_candidates": False,
        "scope": "auxiliary semantic supervision; does not replace instance detection",
        "detection_training_ready": False,
    }


def semantic_target(data: dict) -> tuple[np.ndarray, dict]:
    """Conservative visible-pixel target, independent of candidate order or score."""
    shape = tuple(data["image_size_px"][::-1])
    target = np.full(shape, IGNORE, dtype=np.uint8)
    collision = np.zeros(shape, dtype=bool)
    unresolved = np.zeros(shape, dtype=bool)
    for row in data["entities"]:
        if row["entity"] not in CLASSES:
            raise ValueError("positive class is outside the semantic vocabulary")
        mask = decode(row["segmentation"])
        if mask.shape != shape:
            raise ValueError("positive mask shape mismatch")
        cid = CLASSES[row["entity"]]
        collision |= mask & (target != IGNORE) & (target != cid)
        target[mask] = cid
    for row in data["unresolved_candidates"]:
        mask = decode(row["segmentation"])
        if mask.shape != shape:
            raise ValueError("unresolved mask shape mismatch")
        unresolved |= mask
    target[collision | unresolved] = IGNORE
    counts = np.bincount(target.ravel(), minlength=256)
    return target, {
        "pixels_by_class": {name: int(counts[cid]) for name, cid in CLASSES.items()},
        "valid_pixels": int((target != IGNORE).sum()),
        "ignore_pixels": int(counts[IGNORE]),
        "positive_conflict_pixels": int(collision.sum()),
        "unresolved_pixels": int(unresolved.sum()),
    }


def _relative(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("path escapes package")
    return path


def export_supervision(source: Path, out: Path) -> dict:
    parent = json.loads((source / "report.json").read_text())
    job = json.loads((source / "job.json").read_text())
    job_hash = digest(source / "job.json")
    if parent["status"] != "completed" or job_hash != parent["job_sha256"]:
        raise ValueError("export requires completed bound AI annotations")
    for name, expected in parent["outputs"].items():
        if digest(_relative(source, name)) != expected:
            raise ValueError("source annotation changed")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("images", "annotations", "companions", "producer"):
        (out / name).mkdir()
    for path in (
        Path(__file__),
        Path(__file__).with_name("studioa_autolabel.py"),
        Path(__file__).with_name("studioa_contract.py"),
        Path(__file__).with_name("store_split.py"),
    ):
        shutil.copyfile(path, out / "producer" / path.name)
    write_json(out / "status.json", {"status": "exporting", "completed": 0})
    frames, seen = [], set()
    try:
        for index, frame in enumerate(job["frames"]):
            identity = frame["id"]
            if identity in seen or Path(identity).name != identity:
                raise ValueError("duplicate or invalid frame ID")
            seen.add(identity)
            if camera_store(identity) != (frame["camera"], frame["store"]):
                raise ValueError("frame camera/store mismatch")
            original = _relative(source, frame["image"])
            if digest(original) != frame["image_sha256"]:
                raise ValueError("source image changed")
            annotation_name = f"frames/{identity}.json"
            annotation = _relative(source, annotation_name)
            if digest(annotation) != parent["outputs"].get(annotation_name):
                raise ValueError("annotation is not in source report")
            data = json.loads(annotation.read_text())
            if data.get("job_sha256") != job_hash or data.get("frame_id") != identity:
                raise ValueError("foreign annotation job/frame")
            with Image.open(original) as image:
                rgb = np.asarray(image.convert("RGB"))
            if list(rgb.shape[1::-1]) != frame["image_size_px"]:
                raise ValueError("source image dimensions disagree")
            validate_annotation(data, frame["image_sha256"], rgb.shape[:2])
            target, stats = semantic_target(data)
            image_name = f"images/{identity}{original.suffix}"
            mask_name = f"annotations/{identity}.png"
            companion_name = f"companions/{identity}.json"
            shutil.copyfile(original, out / image_name)
            shutil.copyfile(annotation, out / companion_name)
            Image.fromarray(target).save(out / mask_name)
            frames.append(
                {
                    **{
                        key: frame[key]
                        for key in ("id", "camera", "store", "original_split", "image_size_px")
                    },
                    "image": image_name,
                    "mask": mask_name,
                    "companion": companion_name,
                    "source_annotation_sha256": digest(annotation),
                    "image_sha256": digest(original),
                    "pixel_sha256": hashlib.sha256(
                        str(rgb.shape).encode() + rgb.tobytes()
                    ).hexdigest(),
                    **stats,
                }
            )
            write_json(out / "status.json", {"status": "exporting", "completed": index + 1})
        folds = {}
        for store in STORES:
            # Validate decoded content, not just encoded JPEG identity.
            counts = validate_fold(
                [{**f, "image_sha256": f["pixel_sha256"]} for f in frames], store
            )
            assignments = {
                f["id"]: fold_split(f["store"], f["original_split"], store) for f in frames
            }
            pixels = {}
            for split in ("train", "val", "test"):
                total = Counter(dict.fromkeys(CLASSES, 0))
                for frame in frames:
                    if assignments[frame["id"]] == split:
                        total.update(frame["pixels_by_class"])
                pixels[split] = dict(total)
            folds[store] = {
                "counts": counts,
                "assignments": assignments,
                "pixels_by_split": pixels,
                "missing_train_classes": [
                    name for name, n in pixels["train"].items() if n == 0
                ],
            }
        manifest = {
            "schema": SCHEMA,
            "policy": supervision_policy(),
            "frames": frames,
            "folds": folds,
            "source_report_sha256": digest(source / "report.json"),
            "source": str(source.resolve()),
            "annotation_kind": "ai_generated",
            "independent_accuracy": False,
            "exposure": "legacy images; not blind evaluation for previous models",
        }
        write_json(out / "manifest.json", manifest)
        report = {
            "status": "completed",
            "frames": len(frames),
            "classes": CLASSES,
            "valid_pixels": sum(f["valid_pixels"] for f in frames),
            "ignore_pixels": sum(f["ignore_pixels"] for f in frames),
            "detection_training_ready": False,
            "outputs": {
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "status.json"
            },
        }
        write_json(out / "report.json", report)
        write_json(out / "status.json", {"status": "completed", "completed": len(frames)})
        return report
    except Exception as exc:
        write_json(
            out / "status.json",
            {"status": "failed", "completed": len(frames), "error": str(exc)},
        )
        raise


def check_supervision(root: Path) -> dict:
    report = json.loads((root / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("incomplete supervision export")
    for name, expected in report["outputs"].items():
        if digest(_relative(root, name)) != expected:
            raise ValueError(f"supervision output changed: {name}")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest["policy"] != supervision_policy():
        raise ValueError("supervision policy mismatch")
    for store, fold in manifest["folds"].items():
        rows = [{**f, "image_sha256": f["pixel_sha256"]} for f in manifest["frames"]]
        if validate_fold(rows, store) != fold["counts"]:
            raise ValueError("fold counts disagree")
        for frame in rows:
            if fold["assignments"][frame["id"]] != fold_split(
                frame["store"], frame["original_split"], store
            ):
                raise ValueError("fold assignment changed")
    return manifest


class StudioAPartialDataset(Dataset):
    """Explicit whole-store selection; auxiliary 19-class semantic masks only."""

    def __init__(
        self,
        root: Path,
        held_out: str,
        split: str,
        input_size=(256, 384),
        *,
        train: bool = False,
        augment: dict | None = None,
    ):
        manifest = check_supervision(root)
        if train and split != "train":
            raise ValueError("evaluation partitions cannot use training augmentation")
        if held_out not in STORES or split not in ("train", "val", "test"):
            raise ValueError("explicit store and train/val/test split required")
        self.root = root
        fold = manifest["folds"][held_out]
        self.frames = [f for f in manifest["frames"] if fold["assignments"][f["id"]] == split]
        self.supervises = ["scene"]
        self.transform = build_transforms(
            input_size, train=train, letterbox=True, augment=augment
        )
        if not self.frames:
            raise ValueError("empty partition")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        with Image.open(self.root / frame["image"]) as image:
            rgb = image.convert("RGB")
        with Image.open(self.root / frame["mask"]) as image:
            target = np.array(image)
        if target.ndim != 2 or not set(np.unique(target)) <= {*CLASSES.values(), IGNORE}:
            raise ValueError("invalid semantic target IDs")
        sample = self.transform(Sample(image=rgb, masks={"scene": target}))
        return {
            "image": sample["image"],
            "targets": sample["masks"],
            "supervises": self.supervises,
        }
