"""Dimensioned Stage0 assets. Local front is -Z; origin is the support contact plane.

These are category templates, not exact brand/SKU replicas. Fitting scales each axis
to a bounded estimate and rotates the whole template, including its asymmetric parts.
"""

from __future__ import annotations

import numpy as np

from syncai_bev3d.meshes import _merge, box, chair, extrude

# Nominal width, depth, total height (metres); priors, never surveyed measurements.
NOMINAL = {
    "laptop": (0.34, 0.26, 0.23),
    "monitor": (0.54, 0.22, 0.44),
    "chair": (0.48, 0.52, 0.88),
    "stool": (0.38, 0.38, 0.65),
    "computer_tower": (0.20, 0.40, 0.42),
    "tablet": (0.19, 0.13, 0.25),
    "phone": (0.08, 0.09, 0.15),
}
COLORS = {
    "laptop": (77, 153, 225),
    "monitor": (75, 186, 210),
    "chair": (194, 136, 88),
    "stool": (194, 157, 106),
    "computer_tower": (103, 111, 129),
    "tablet": (229, 160, 77),
    "phone": (218, 91, 104),
}
FLOOR_OBJECTS = {"chair", "stool", "computer_tower"}


def _shift(mesh, offset):
    return mesh[0] + offset, mesh[1]


def asset_mesh(category: str, width: float, depth: float, height: float, *, variant="standard"):
    if not np.isfinite([width, depth, height]).all() or min(width, depth, height) <= 0:
        raise ValueError("asset dimensions must be finite and positive")
    w, d, h = width, depth, height
    if variant == "flat" and category in {"tablet", "phone"}:
        return box(w, h, d)
    if category == "chair":
        return chair(w, d, h)
    if category == "stool":
        angle = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        if variant == "pedestal":
            ring = np.c_[np.cos(angle) * w, np.sin(angle) * d] / 2
            foot = extrude(ring, h * 0.035)
            stem = box(w * 0.10, h * 0.70, d * 0.10)
            seat = _shift(extrude(ring, h * 0.06), [0, h * 0.70, 0])
            back = _shift(box(w * 0.90, h * 0.24, d * 0.10), [0, h * 0.76, d * 0.4])
            return _merge(foot, stem, seat, back)
        seat = extrude(np.c_[w * np.cos(angle), d * np.sin(angle)] / 2, h * 0.10)
        parts = [_shift(seat, [0, h * 0.9, 0])]
        for x in (-0.29 * w, 0.29 * w):
            for z in (-0.29 * d, 0.29 * d):
                parts.append(_shift(box(w * 0.07, h * 0.9, d * 0.07), [x, 0, z]))
        return _merge(*parts)
    if category == "laptop":
        base = box(w, h * 0.06, d)
        # Open lid at the back; front/back produce different calibrated silhouettes.
        lid = _shift(box(w, h * 0.94, d * 0.05), [0, h * 0.06, d * 0.475])
        return _merge(base, lid)
    if category in {"monitor", "tablet", "phone"}:
        foot = box(w * 0.48, h * 0.045, d)
        stem = _shift(box(w * 0.10, h * 0.34, d * 0.16), [0, h * 0.045, 0])
        panel = _shift(box(w, h * 0.72, d * 0.14), [0, h * 0.28, d * 0.10])
        return _merge(foot, stem, panel)
    if category == "computer_tower":
        return box(w, h, d)
    raise ValueError(f"unknown Stage0 asset category: {category}")


def templates(category):
    """Distinct physical configurations, each with its own size prior."""
    yield "standard", NOMINAL[category]
    if category in {"tablet", "phone"}:
        yield "flat", (0.19, 0.25, 0.012) if category == "tablet" else (0.078, 0.155, 0.009)
    if category == "stool":
        yield "pedestal", (0.44, 0.44, 0.85)


def plausible_aspect(category, dimensions, nominal, variant):
    """Prevent silhouette fitting from turning a landscape laptop into a narrow slab.

    These are deliberately broad template priors, not SKU measurements. Independent
    axis bounds alone allow width/depth ratios to vary by a factor of two.
    """
    pairs = []
    if category == "laptop":
        pairs = [(0, 1), (0, 2)]
    elif category in {"monitor", "tablet", "phone"}:
        pairs = [(0, 1)] if variant == "flat" else [(0, 2)]
    for a, b in pairs:
        relative = (dimensions[a] / dimensions[b]) / (nominal[a] / nominal[b])
        if not 0.8 <= relative <= 1.25:
            return False
    return True


def door_mesh(points, height: float, thickness: float = 0.08):
    """Closed door leaf plus jambs, in its fitted plane. Swing/handle side is unknown.

    A handle is deliberately not invented: neither handedness nor open angle follows
    from the doorway mask. The frame stays within the observed opening's dimensions.
    """
    from syncai_bev3d.meshes import Placement, place

    points = np.asarray(points, float)
    delta = points[1] - points[0]
    width = float(np.linalg.norm(delta))
    if not np.isfinite(points).all() or min(width, height, thickness) <= 0:
        raise ValueError("invalid door dimensions")
    jamb = min(0.045, width / 8, height / 8)
    pieces = [box(width - 2 * jamb, height - jamb, thickness * 0.5)]
    for x in (-width / 2 + jamb / 2, width / 2 - jamb / 2):
        pieces.append(_shift(box(jamb, height, thickness), [x, 0, 0]))
    pieces.append(_shift(box(width, jamb, thickness), [0, height - jamb, 0]))
    centre = points.mean(0)
    return place(
        _merge(*pieces), Placement(*centre, heading_rad=-float(np.arctan2(delta[1], delta[0])))
    )
