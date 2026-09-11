"""Stage0 small objects retain instance identity, prompt and source-image provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ObjectInstance:
    category: str
    mask: np.ndarray
    score: float
    prompt: str


# Keep this separate from the training taxonomy: these are static scene assets.
# `desktop computer` is deliberately absent: the fleet prompt audit found price cards.
OBJECT_PROMPTS = {
    "laptop": (("open laptop", "laptop computer"), 0.30),
    "monitor": (("computer monitor", "imac"), 0.50),
    "chair": (("chair", "office chair"), 0.45),
    "stool": (("stool",), 0.45),
    "tablet": (("tablet computer",), 0.40),
    "phone": (("smartphone",), 0.40),
}


def deduplicate(instances: list[ObjectInstance]) -> list[ObjectInstance]:
    """Prompt duplicates and nested screen/laptop claims become one physical asset.

    A monitor prompt often finds only a laptop's lid; prefer the whole laptop when
    its mask contains that screen. Otherwise confidence resolves overlapping claims.
    """
    ordered = sorted(instances, key=lambda i: i.score, reverse=True)
    kept = []
    for candidate in ordered:
        area = int(candidate.mask.sum())
        if not area:
            continue
        duplicate = False
        replace = []
        for index, other in enumerate(kept):
            intersection = int((candidate.mask & other.mask).sum())
            smaller = min(area, int(other.mask.sum()))
            union = area + int(other.mask.sum()) - intersection
            if intersection / union > 0.45 or intersection / smaller > 0.75:
                # A lower-confidence laptop can replace a screen-only detection when
                # its visible base adds substantial area. Equal masks retain the more
                # confident class, so an iMac is not automatically turned into a laptop.
                if (
                    candidate.category == "laptop"
                    and other.category == "monitor"
                    and area > 1.18 * other.mask.sum()
                    and intersection > 0.80 * other.mask.sum()
                ):
                    replace.append(index)
                    continue
                duplicate = True
                break
        if not duplicate:
            kept = [other for index, other in enumerate(kept) if index not in replace]
            kept.append(candidate)
    return kept


def image_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_instances(path: Path, instances, *, shape, source: Path, categories):
    """An empty successful pass is distinct from a camera never processed."""
    h, w = shape
    masks = np.asarray([i.mask for i in instances], dtype=bool).reshape(-1, h, w)
    metadata = {
        "schema_version": 1,
        "source_sha256": image_digest(source),
        "source": str(source),
        "categories": list(categories),
        "instances": [
            {"category": i.category, "score": i.score, "prompt": i.prompt} for i in instances
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        masks=np.packbits(masks, axis=-1),
        shape=np.array(shape),
        metadata=np.array(json.dumps(metadata)),
    )


def load_instances(path: Path, *, source: Path | None = None):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["schema_version"] != 1:
            raise ValueError("unsupported object instance schema")
        if source is not None and image_digest(source) != metadata["source_sha256"]:
            raise ValueError("object masks belong to a different plate; rerun objects_pass")
        h, w = map(int, data["shape"])
        masks = np.unpackbits(data["masks"], axis=-1)[..., :w].astype(bool)
        if masks.shape != (len(metadata["instances"]), h, w):
            raise ValueError("object instance mask/metadata dimensions disagree")
    instances = [
        ObjectInstance(mask=m, **entry)
        for m, entry in zip(masks, metadata["instances"], strict=True)
    ]
    return instances, metadata
