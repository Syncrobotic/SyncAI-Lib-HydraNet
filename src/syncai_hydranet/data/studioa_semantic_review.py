"""Replayable AI semantic revisions with explicit split scope and immutable test data."""

from __future__ import annotations

import copy
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .studioa_review import digest, write_json
from .studioa_supervision import CLASSES, IGNORE, _relative, check_supervision

REVIEW_SCHEMA = "studioa.semantic-review.v1"
REVISION_SCHEMA = "studioa.semantic-review.v2"


def review_splits(review: dict) -> tuple[str, ...]:
    """Legacy reviews are train-only; evaluation revisions require explicit provenance."""
    if review.get("schema") == REVIEW_SCHEMA:
        if any(f.get("quarantine_regions") for f in review["frames"]):
            raise ValueError("regional quarantine requires a v2 review")
        return ("train",)
    if (
        review.get("schema") == REVISION_SCHEMA
        and review.get("revision_kind") == "source_train_val_correction"
        and review.get("evaluation_exposure") == "previously_inspected_not_blind"
        and review.get("evaluation_revision_reason")
    ):
        return ("train", "val")
    raise ValueError("unknown review schema or missing evaluation revision provenance")


def validate_item_scope(item: dict, split: str | None, review: dict) -> None:
    if split not in review_splits(review):
        raise ValueError("non-train correction or protected evaluation data")
    if review["schema"] == REVISION_SCHEMA and item.get("split") != split:
        raise ValueError("review item split binding mismatch")


def polygon_mask(region: dict, shape: tuple[int, ...]) -> np.ndarray:
    """Rasterize a reasoned, bounded interior using original image coordinates."""
    if region.get("scope") != "semantic_interior" or not region.get("reason"):
        raise ValueError("region must be a reasoned semantic interior")
    h, w = shape
    points = region["polygon_xy"]
    if len(points) < 3 or any(
        len(p) != 2
        or any(type(v) is not int for v in p)
        or not (0 <= p[0] < w and 0 <= p[1] < h)
        for p in points
    ):
        raise ValueError("polygon outside image or noninteger coordinates")
    area = abs(
        sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(points, points[1:] + points[:1], strict=True)
        )
    )
    if not area:
        raise ValueError("degenerate polygon")
    canvas = Image.new("L", (w, h))
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in points], fill=1)
    return np.asarray(canvas, dtype=bool)


def check_review(root: Path, manifest: dict) -> None:
    """Replay corrections; require byte-identical test and unreviewed data."""
    directory = root / "semantic_review"
    meta = manifest["semantic_review"]
    parent = json.loads((directory / "parent_manifest.json").read_text())
    parent_report = json.loads((directory / "parent_report.json").read_text())
    review = json.loads((directory / "review.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    if (
        meta["schema"] != review["schema"]
        or digest(directory / "review.json") != meta["review_sha256"]
        or digest(directory / "parent_manifest.json") != meta["source_manifest_sha256"]
        or review["source_manifest_sha256"] != meta["source_manifest_sha256"]
        or review["reviewer_kind"] != "ai"
        or review["held_out"] != meta["held_out"]
        or set(manifest["folds"]) != {meta["held_out"]}
    ):
        raise ValueError("semantic review provenance mismatch")
    review_splits(review)
    items = {row["frame_id"]: row for row in review["frames"]}
    if len(items) != len(review["frames"]) or set(results) != set(items):
        raise ValueError("semantic review frame mismatch")
    parents = {f["id"]: f for f in parent["frames"]}
    if set(parents) != {f["id"] for f in manifest["frames"]}:
        raise ValueError("semantic review frame set changed")
    fold = manifest["folds"][meta["held_out"]]
    original_fold = parent["folds"][meta["held_out"]]
    for key in ("assignments", "counts"):
        if fold[key] != original_fold[key]:
            raise ValueError("semantic review partitions changed")
    for frame in manifest["frames"]:
        fid = frame["id"]
        previous = parents[fid]
        for field in ("image", "companion"):
            if (
                frame[field] != previous[field]
                or digest(root / frame[field]) != (parent_report["outputs"][previous[field]])
            ):
                raise ValueError("semantic review changed source evidence")
        if fid not in items:
            if (
                frame != previous
                or digest(root / frame["mask"]) != (parent_report["outputs"][previous["mask"]])
            ):
                raise ValueError("unreviewed frame changed")
            continue
        item = items[fid]
        validate_item_scope(item, fold["assignments"][fid], review)
        parent_mask = directory / "parent_masks" / f"{fid}.png"
        if (
            digest(parent_mask) != parent_report["outputs"][previous["mask"]]
            or digest(parent_mask) != item["mask_sha256"]
            or item["image_sha256"] != previous["image_sha256"]
        ):
            raise ValueError("semantic review parent binding changed")
        with Image.open(parent_mask) as im:
            expected, stats = apply_correction(np.array(im), item)
        with Image.open(root / frame["mask"]) as im:
            if not np.array_equal(expected, np.array(im)):
                raise ValueError("semantic review replay mismatch")
        if results[fid] != stats or any(
            frame[key] != stats[key]
            for key in ("pixels_by_class", "valid_pixels", "ignore_pixels")
        ):
            raise ValueError("semantic review statistics changed")
    for split in ("train", "val", "test"):
        counts = Counter(dict.fromkeys(CLASSES, 0))
        for frame in manifest["frames"]:
            if fold["assignments"][frame["id"]] == split:
                counts.update(frame["pixels_by_class"])
        if dict(counts) != fold["pixels_by_split"][split]:
            raise ValueError("semantic review fold statistics changed")


def apply_correction(target: np.ndarray, item: dict) -> tuple[np.ndarray, dict]:
    """Quarantine old positives, then label reviewed interiors of unknown regions.

    Regional quarantines explicitly name the mistaken source classes. Other positive
    classes remain protected. These interiors are not complete object instances.
    """
    if target.ndim != 2 or not np.isin(target, [*CLASSES.values(), IGNORE]).all():
        raise ValueError("invalid semantic target")
    if not item.get("reason"):
        raise ValueError("correction needs a visual review reason")
    result = target.copy()
    for entity in item["quarantine_classes"]:
        if entity not in CLASSES:
            raise ValueError("unknown quarantine class")
        result[result == CLASSES[entity]] = IGNORE
    for region in item.get("quarantine_regions", []):
        classes = region["classes"]
        if not classes or any(entity not in CLASSES for entity in classes):
            raise ValueError("unknown or empty regional quarantine classes")
        mask = polygon_mask(region, target.shape)
        result[mask & np.isin(result, [CLASSES[entity] for entity in classes])] = IGNORE
    quarantined = int(np.count_nonzero(result != target))
    proposals = np.full(target.shape, IGNORE, dtype=np.uint8)
    blocked = np.zeros(target.shape, dtype=bool)
    for region in item["positive_interiors"]:
        if region["entity"] not in CLASSES:
            raise ValueError("unknown positive class")
        mask = polygon_mask(region, target.shape)
        cid = CLASSES[region["entity"]]
        if np.any(mask & (proposals != IGNORE) & (proposals != cid)):
            raise ValueError("contradictory reviewed polygons")
        proposals[mask] = cid
        blocked |= mask & (result != IGNORE) & (result != cid)
    accepted = (proposals != IGNORE) & (result == IGNORE)
    result[accepted] = proposals[accepted]
    counts = np.bincount(result.ravel(), minlength=256)
    changed = target != result
    transitions = Counter(zip(target[changed].tolist(), result[changed].tolist(), strict=True))
    return result, {
        "quarantined_pixels": quarantined,
        "added_positive_pixels": int(accepted.sum()),
        "preserved_other_positive_pixels": int(blocked.sum()),
        "changed_pixels": int(changed.sum()),
        "transitions": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())},
        "pixels_by_class": {name: int(counts[cid]) for name, cid in CLASSES.items()},
        "valid_pixels": int((result != IGNORE).sum()),
        "ignore_pixels": int(counts[IGNORE]),
    }


def export_review(source: Path, reviews: Path, out: Path) -> dict:
    """Keep the parent immutable; bind every edit to image and mask content."""
    manifest = check_supervision(source)
    review = json.loads(reviews.read_text())
    if manifest.get("semantic_review"):
        raise ValueError("review chaining requires a new explicit source audit")
    if review.get("reviewer_kind") != "ai" or review.get("source_manifest_sha256") != digest(
        source / "manifest.json"
    ):
        raise ValueError("review requires source binding and AI provenance")
    review_splits(review)
    held_out = review["held_out"]
    assignments = manifest["folds"][held_out]["assignments"]
    frames = {f["id"]: f for f in manifest["frames"]}
    prepared, seen = {}, set()
    for item in review["frames"]:
        fid = item["frame_id"]
        if fid in seen or fid not in frames:
            raise ValueError("duplicate or unknown correction")
        validate_item_scope(item, assignments[fid], review)
        seen.add(fid)
        frame = frames[fid]
        for field in ("image", "mask"):
            if item.get(f"{field}_sha256") != digest(_relative(source, frame[field])):
                raise ValueError(f"review {field} binding changed")
        with Image.open(_relative(source, frame["mask"])) as im:
            target = np.array(im)
        if list(target.shape[::-1]) != frame["image_size_px"]:
            raise ValueError("mask dimensions disagree")
        prepared[fid] = apply_correction(target, item)
    if not prepared:
        raise ValueError("empty review")
    if out.resolve().is_relative_to(source.resolve()):
        raise ValueError("output must be separate from the immutable source")
    # Copy only bound files, never untracked previews or links to mutable parents.
    parent_report = json.loads((source / "report.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    try:
        for name in parent_report["outputs"]:
            destination = _relative(out, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_relative(source, name), destination)
        provenance = out / "semantic_review"
        provenance.mkdir()
        shutil.copyfile(reviews, provenance / "review.json")
        shutil.copyfile(source / "manifest.json", provenance / "parent_manifest.json")
        shutil.copyfile(source / "report.json", provenance / "parent_report.json")
        shutil.copyfile(Path(__file__), provenance / "producer.py")
        for name in ("studioa_supervision.py", "studioa_review.py", "store_split.py"):
            shutil.copyfile(Path(__file__).with_name(name), provenance / name)
        updated = copy.deepcopy(manifest)
        results = {}
        for frame in updated["frames"]:
            fid = frame["id"]
            if fid not in prepared:
                continue
            target, stats = prepared[fid]
            original = provenance / "parent_masks" / f"{fid}.png"
            original.parent.mkdir(exist_ok=True)
            shutil.copyfile(source / frame["mask"], original)
            Image.fromarray(target).save(out / frame["mask"])
            frame["parent_mask_sha256"] = digest(original)
            frame["semantic_review_frame"] = fid
            frame["parent_uncertainty_statistics"] = {
                name: frame.pop(name)
                for name in ("positive_conflict_pixels", "unresolved_pixels")
                if name in frame
            }
            for name in ("pixels_by_class", "valid_pixels", "ignore_pixels"):
                frame[name] = stats[name]
            results[fid] = stats
        # Revisions never transfer reviewed source frames to another held-out fold.
        updated["folds"] = {held_out: updated["folds"][held_out]}
        fold = updated["folds"][held_out]
        for split in ("train", "val", "test"):
            counts = Counter(dict.fromkeys(CLASSES, 0))
            for frame in updated["frames"]:
                if assignments[frame["id"]] == split:
                    counts.update(frame["pixels_by_class"])
            fold["pixels_by_split"][split] = dict(counts)
        fold["missing_train_classes"] = [
            k for k, n in fold["pixels_by_split"]["train"].items() if not n
        ]
        updated["semantic_review"] = {
            "schema": review["schema"],
            "held_out": held_out,
            "review_sha256": digest(reviews),
            "source_manifest_sha256": digest(source / "manifest.json"),
            "source": str(source.resolve()),
            "scope": (
                "train-only semantic interiors; not instance annotations"
                if review["schema"] == REVIEW_SCHEMA
                else "source train/val revision; test immutable; not instance annotations"
            ),
            "companion_role": "parent evidence only; replay review for corrected targets",
            "uncertainty": "explicit reviewed interiors resolve parent ignore pixels",
            "independent_accuracy": False,
        }
        write_json(provenance / "results.json", results)
        write_json(out / "manifest.json", updated)
        report = {
            "status": "completed",
            "frames": len(frames),
            "corrected_train_frames": sum(assignments[fid] == "train" for fid in prepared),
            "corrected_val_frames": sum(assignments[fid] == "val" for fid in prepared),
            "classes": CLASSES,
            "valid_pixels": sum(f["valid_pixels"] for f in updated["frames"]),
            "ignore_pixels": sum(f["ignore_pixels"] for f in updated["frames"]),
            "detection_training_ready": False,
            "outputs": {
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file() and p.name not in ("report.json", "status.json")
            },
        }
        write_json(out / "report.json", report)
        check_supervision(out)
        write_json(out / "status.json", {"status": "completed"})
        return report
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "error": str(exc)})
        raise
