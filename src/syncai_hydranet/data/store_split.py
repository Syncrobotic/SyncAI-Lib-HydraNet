"""Whole-store folds that preserve the existing camera-level development boundary."""

from __future__ import annotations

import re
from collections import defaultdict

STORES = ("Kaohsiung", "Taichung", "Tao-Hsin")
CAMERA = re.compile(r"(Kaohsiung|Taichung|Tao-Hsin)-cam\d+")


def camera_store(path: str) -> tuple[str, str]:
    match = CAMERA.search(path)
    if match is None:
        raise ValueError(f"cannot identify store/camera: {path}")
    return match[0], match[1]


def fold_split(store: str, original_split: str, held_out: str) -> str | None:
    """The target store is test only; source-store test cameras remain unused."""
    if store not in STORES or held_out not in STORES:
        raise ValueError("unknown store")
    if original_split not in ("train", "val", "test"):
        raise ValueError("unknown original split")
    if store == held_out:
        return "test"
    return original_split if original_split != "test" else None


def validate_fold(rows: list[dict], held_out: str) -> dict[str, int]:
    """Reject shared cameras or identical image contents across partitions."""
    cameras: dict[str, set[str]] = defaultdict(set)
    contents: dict[str, set[str]] = defaultdict(set)
    counts = dict.fromkeys(("train", "val", "test"), 0)
    for row in rows:
        split = fold_split(row["store"], row["original_split"], held_out)
        if split is None:
            continue
        counts[split] += 1
        cameras[row["camera"]].add(split)
        contents[row["image_sha256"]].add(split)
    if any(len(parts) != 1 for parts in cameras.values()):
        raise ValueError("camera leakage across fold partitions")
    if any(len(parts) != 1 for parts in contents.values()):
        raise ValueError("identical image leakage across fold partitions")
    if not all(counts.values()):
        raise ValueError(f"empty fold partition: {counts}")
    return counts
