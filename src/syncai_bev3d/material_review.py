"""Mutually exclusive candidate opening materials from source-bound image regions."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from syncai_bev3d.opening_controls import source_identity
from syncai_hydranet.geometry.camera_json import CameraFile


def material_overlap(masks: dict[str, np.ndarray]) -> dict:
    glass = np.asarray(masks["glass"], bool)
    door = np.asarray(masks["glass_door"], bool)
    if glass.shape != door.shape:
        raise ValueError("material masks must share raw image dimensions")
    count = int((glass & door).sum())
    return {
        "glass_pixels": int(glass.sum()),
        "glass_door_pixels": int(door.sum()),
        "overlap_pixels": count,
        "glass_door_overlap_fraction": count / int(door.sum()) if door.any() else None,
    }


def _polygon(points, size):
    p = np.asarray(points, float)
    if (
        p.ndim != 2
        or p.shape[1] != 2
        or len(p) < 3
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p >= size).any()
    ):
        raise ValueError("review polygon needs at least three finite raw in-frame points")
    # A repeated/collinear polygon cannot silently delete a material region.
    area = (
        abs(float(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))))
        / 2
    )
    if area < 4:
        raise ValueError("review polygon is degenerate")
    image = Image.new("1", size)
    ImageDraw.Draw(image).polygon([tuple(point) for point in p], fill=1)
    return np.asarray(image, bool)


def review_materials(root, camera, review):
    """Keep the legacy references and return a separate candidate label set.

    The reviewer supplies image regions, never a projected candidate GLB. Original
    teacher support is retained inside the observed facade. Occluders remain unknown;
    pixels outside the observed facade are not labelled as glass.
    """
    if review.get("schema") != 1 or review.get("camera") != camera:
        raise ValueError("material review schema or camera mismatch")
    if review.get("source_identity") != source_identity(root, camera):
        raise ValueError("stale material review: plate, camera or masks changed")
    if not review.get("reviewer") or not review.get("evidence"):
        raise ValueError("material review needs reviewer and visible evidence")
    cf = CameraFile.load(root / f"runs/commission01/{camera}.camera.json")
    if review.get("image_size_px") != list(cf.image_size_px):
        raise ValueError("material review must use raw image dimensions")
    masks = {}
    for kind in ("glass", "glass_door", "door"):
        relative = cf.mask_files.get(kind, f"{camera}/masks/{kind}.png")
        with Image.open(root / "runs/commission01" / relative) as image:
            if image.size != cf.image_size_px:
                raise ValueError("material mask does not match raw frame")
            masks[kind] = np.asarray(image.convert("L")) > 127
    facade = _polygon(review["facade_region_px"], cf.image_size_px)
    panels = np.zeros_like(facade)
    if not review.get("door_regions_px"):
        raise ValueError("material review needs visible door regions")
    for polygon in review["door_regions_px"]:
        panels |= _polygon(polygon, cf.image_size_px)
    unknown = np.zeros_like(facade)
    for polygon in review.get("occluded_regions_px", []):
        unknown |= _polygon(polygon, cf.image_size_px)
    support = (masks["glass"] | masks["glass_door"] | masks["door"]) & facade
    candidate = {
        "glass_door": support & panels & ~unknown,
        "glass": support & ~panels & ~unknown,
        "door": masks["door"] & ~facade,
        "material_ignore": support & unknown,
    }
    report = {
        "schema": 1,
        "camera": camera,
        "status": "candidate; visual review required",
        "before": material_overlap(masks),
        "after": material_overlap(candidate),
        "occluded_pixels": int(candidate["material_ignore"].sum()),
        "glazing_pixels_outside_reviewed_facade": int(
            ((masks["glass"] | masks["glass_door"]) & ~facade).sum()
        ),
        "source_identity": review["source_identity"],
        "scope": (
            "agent-reviewed regions on existing teacher masks; not independent human labels"
        ),
        "legacy_reference_preserved": True,
    }
    return candidate, report
