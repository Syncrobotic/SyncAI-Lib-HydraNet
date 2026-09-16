"""Separate walkable-area depth checks from explicitly observed visible-floor checks."""

from __future__ import annotations

import numpy as np
from PIL import Image

from syncai_bev3d.material_review import _polygon
from syncai_bev3d.opening_controls import source_identity


def visible_floor_mask(root, cf, review):
    """Image observations only: neither predicted heights nor residuals select pixels."""
    if review.get("schema") != 1 or review.get("camera") != cf.camera_id:
        raise ValueError("visible-floor review schema or camera mismatch")
    if review.get("image_size_px") != list(cf.image_size_px):
        raise ValueError("visible-floor regions must use raw image coordinates")
    if review.get("source_identity") != source_identity(root, cf.camera_id):
        raise ValueError("stale visible-floor review")
    if not review.get("reviewer") or not review.get("evidence"):
        raise ValueError("visible-floor review needs reviewer and image evidence")
    if not review.get("regions_px"):
        raise ValueError("visible-floor review needs observed regions")
    selected = np.zeros(cf.image_size_px[::-1], bool)
    for region in review["regions_px"]:
        selected |= _polygon(region, cf.image_size_px)
    with Image.open(root / "runs/commission01" / cf.mask_files["walkable"]) as image:
        if image.size != cf.image_size_px:
            raise ValueError("walkable mask must use raw image dimensions")
        walkable = np.asarray(image.convert("L")) > 127
    return selected & walkable


def floor_height_consistency(arrays, walkable, *, visible=None):
    """Report both denominators; do not call occupied walkable pixels floor accuracy."""
    walkable = np.asarray(walkable, bool)
    if walkable.ndim != 2:
        raise ValueError("walkable mask must be a 2D image")
    if visible is not None:
        visible = np.asarray(visible, bool)
        if visible.shape != walkable.shape or (visible & ~walkable).any():
            raise ValueError("visible floor must be a subset of the raw walkable mask")
    height, valid = arrays["height"], arrays["geom_ok"]
    if height.ndim != 2 or height.shape != valid.shape:
        raise ValueError("height and geometry validity must share dimensions")

    def score(mask):
        selected = np.asarray(
            Image.fromarray(mask).resize(height.shape[::-1], Image.Resampling.NEAREST)
        )
        usable = selected & valid & np.isfinite(height)
        values = height[usable]
        return {
            "raw_selected_pixels": int(mask.sum()),
            "cache_selected_pixels": int(selected.sum()),
            "valid_cache_pixels": int(usable.sum()),
            "median_signed_height_m": float(np.median(values)) if len(values) else None,
            "median_abs_height_m": float(np.median(abs(values))) if len(values) else None,
            "p95_abs_height_m": float(np.percentile(abs(values), 95)) if len(values) else None,
        }

    return {
        "walkable_area": score(walkable),
        "walkable_scope": (
            "may include people, chairs and inferred floor; not visible-floor error"
        ),
        "visible_floor": score(visible) if visible is not None else None,
        "visible_scope": (
            "source-image reviewed regions; conditional on existing camera and teacher"
        ),
        "independent_metric_accuracy": False,
    }
