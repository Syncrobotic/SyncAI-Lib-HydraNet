"""Turn one frame's masks, boxes and depth into a metric scene a renderer can draw.

The **depth** half of this package. `bev.scene` answers the same question from a ground
plane -- assumed, or fitted -- and is what runs on footage with no depth at all; this one
back-projects registered metric depth and needs no plane. Both emit a payload called a
scene, both carry `grid` and `objects`, and they are not interchangeable, so each declares
its `source` and they share key names only where the quantity really is the same.

The output is the Tesla-style picture's input: a floor tiled into cells with a label
each, and a list of objects with a position, an extent and a heading -- all in metres,
in the camera's own frame (x right, y down, z forward).

Two decisions are worth stating, because both are places where this could have looked
better and meant less.

**No ground plane is fitted.** With registered metric depth, back-projecting through K is
exact; fitting a plane and projecting onto it would add an assumption (flat floor) to
data that does not need one. The plane matters when you have no depth -- see
docs/RETAIL.md -- and the costmap side of the project fits one for that reason.

**Pixels the depth sensor could not see become `unknown`, not floor.** A polished floor
returns nothing to a projector-based sensor: 10.6% of walkable pixels in a real lobby
frame. Drawing those as floor because the segmentation head is confident would produce
exactly the failure this whole project is trying to avoid -- a picture that looks certain
where the robot is blind. They are rendered as a hole, and the hole is the point.

pytest tests/test_bev_scene.py -v
"""

from __future__ import annotations

import numpy as np

from .scene_types import DepthGrid, DepthObject, DepthObjectMetrics, DepthScene

# Traversability class ids, as in data/label_maps_indoor.py.
BLOCKED, CAUTION, GO = 0, 1, 2

# Cell states the renderer knows how to draw.
CELL_EMPTY, CELL_GO, CELL_CAUTION, CELL_BLOCKED, CELL_UNKNOWN = 0, 1, 2, 3, 4


def backproject(depth_m: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Depth image -> (H, W, 3) points in camera coordinates, in metres.

    ``k`` is the 3x3 intrinsic matrix for *this* image's resolution. Passing the full-res
    K for a subsampled image is off by exactly the stride and produces a scene that is
    self-consistent and wrong, which is why the recorder scales K when it writes it.
    """
    h, w = depth_m.shape
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_m.astype(np.float32)
    return np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=-1)


def bev_grid(
    points: np.ndarray,
    trav: np.ndarray,
    *,
    extent: tuple[float, float, float] = (-4.0, 4.0, 8.0),
    cell: float = 0.2,
    max_height: float = 0.35,
) -> DepthGrid:
    """Bin back-projected points into a top-down grid of labelled cells.

    ``extent`` is (x_min, x_max, z_max); the camera sits at z=0. Cells take the label of
    the majority of points that land in them, with one exception: a cell holding any
    `blocked` point is blocked. Under-reporting an obstacle is the expensive direction of
    that error.

    ``max_height`` keeps a walkable cell from being claimed by whatever hangs above it.
    Height here is -y, and the camera's own height is not known to this function, so the
    cut is relative to the lowest point seen in that cell rather than to the floor.
    """
    x_min, x_max, z_max = extent
    nx = round((x_max - x_min) / cell)
    nz = round(z_max / cell)
    grid = np.zeros((nz, nx), dtype=np.uint8)

    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    valid = (z > 0) & (z < z_max) & (x >= x_min) & (x < x_max)
    ix = np.full(z.shape, -1, dtype=np.int32)
    iz = np.full(z.shape, -1, dtype=np.int32)
    ix[valid] = ((x[valid] - x_min) / cell).astype(np.int32)
    iz[valid] = (z[valid] / cell).astype(np.int32)

    flat = iz * nx + ix
    for state, cls in ((CELL_GO, GO), (CELL_CAUTION, CAUTION), (CELL_BLOCKED, BLOCKED)):
        sel = valid & (trav == cls)
        if not sel.any():
            continue
        if state == CELL_GO:
            # Drop points floating above the local surface: a table top over free floor
            # must not turn that floor into a walkable cell at table height.
            low = _per_cell_min(flat[sel], y[sel], nx * nz)
            sel_idx = flat[sel]
            keep = y[sel] <= low[sel_idx] + max_height
            counts = np.bincount(sel_idx[keep], minlength=nx * nz)
        else:
            counts = np.bincount(flat[sel], minlength=nx * nz)
        hit = counts.reshape(nz, nx) > 0
        grid[hit & (grid == CELL_EMPTY)] = state
        if state == CELL_BLOCKED:
            grid[hit] = CELL_BLOCKED  # blocked always wins

    # Walkable by the model, invisible to the sensor: no 3D position exists for these
    # pixels, so they cannot become cells at all. They leave a hole in the floor, and the
    # hole is the honest rendering -- reported as a number so it can be watched.
    walkable = trav == GO
    blind = walkable & (z <= 0)
    return {
        "nx": nx,
        "nz": nz,
        "cell_m": cell,
        "x_min": x_min,
        "cells": grid.reshape(-1).tolist(),
        "blind_fraction": float(blind.sum() / max(walkable.sum(), 1)),
    }


def _per_cell_min(flat_idx: np.ndarray, values: np.ndarray, n_cells: int) -> np.ndarray:
    """Minimum value per cell, as a dense array. np.minimum.at is the honest way to do
    this without sorting; it is slow enough to matter only at full resolution."""
    out = np.full(n_cells, np.inf, dtype=np.float32)
    np.minimum.at(out, flat_idx, values)
    return out


def _centre_stats(points: np.ndarray, box, frac: float = 0.5) -> tuple[float, float] | None:
    """Depth and spread of the middle of a box: the part a detector is pointing at.

    Both come from the same points on purpose. Taking the anchor from the centre and the
    spread from the whole box puts the background back in through the other door -- with
    a wall behind the object the spread becomes the distance to that wall, the acceptance
    window opens wide enough to swallow it, and the filtering does nothing.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * frac / 2, (y1 - y0) * frac / 2
    sub = points[
        max(int(cy - hh), 0) : int(cy + hh) + 1, max(int(cx - hw), 0) : int(cx + hw) + 1
    ].reshape(-1, 3)
    z = sub[sub[:, 2] > 0][:, 2]
    if len(z) < 8:
        return None
    med = float(np.median(z))
    return med, float(np.median(np.abs(z - med)))


def object_from_box(
    box, points: np.ndarray, *, min_points: int = 30
) -> DepthObjectMetrics | None:
    """Position, extent and heading for one detection, from the depth inside its box.

    Returns None when the box holds too few depth returns to say anything -- which is the
    common case for glass, for anything past about 4 m on a D435-class sensor, and for
    thin structures. A missing object is honest; an object placed from six pixels is not.

    The heading comes from the principal axis of the footprint, and is **ambiguous by
    180 degrees**: a chair's point cloud does not say which way it faces. That is
    tolerable for a picture and not for a planner, so it is reported alongside the
    evidence rather than presented as a measurement.
    """
    x0, y0, x1, y1 = (round(v) for v in box)
    h, w = points.shape[:2]
    x0, x1 = max(x0, 0), min(x1, w)
    y0, y1 = max(y0, 0), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = points[y0:y1, x0:x1].reshape(-1, 3)
    patch = patch[patch[:, 2] > 0]
    if len(patch) < min_points:
        return None

    # A detection box is a rectangle around a thing that is not rectangular, so it always
    # contains background: floor in front, wall behind, corridor either side. Measured
    # naively that background sets the extent -- a chair at 3.4 m came back 9.65 m wide
    # in the demo scene, because the box clipped the walls.
    #
    # The anchor comes from the middle of the box rather than from the majority of it. A
    # chair is mostly gaps: legs, the space under the seat, the gap above the backrest,
    # so background can be the majority of a perfectly tight box and the median depth is
    # then the wall behind it. What a detector does guarantee is that the *centre* is the
    # thing it detected.
    stats = _centre_stats(points, (x0, y0, x1, y1))
    if stats is None:
        return None
    z_anchor, spread = stats
    # Scaled by the object's own spread so a deep object is not trimmed to its front
    # face, floored so a flat one is not trimmed to nothing.
    window = max(3.0 * spread, 0.25)
    near = patch[np.abs(patch[:, 2] - z_anchor) <= window]
    if len(near) < min_points:
        return None
    patch = near

    # Percentiles, not min/max: a single stray return at the far wall would otherwise set
    # the object's depth. The 5-95 span deliberately under-reports extent by about 10%,
    # which is the right direction for a drawn box -- an object rendered slightly small
    # is a smaller lie than one rendered slightly large.
    lo = np.percentile(patch, 5, axis=0)
    hi = np.percentile(patch, 95, axis=0)
    centre = np.median(patch, axis=0)

    footprint = patch[:, [0, 2]] - centre[[0, 2]]
    cov = np.cov(footprint.T)
    if not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    yaw = float(np.arctan2(axis[1], axis[0]))
    # How elongated the footprint is. A near-circular one makes the heading meaningless,
    # and the renderer uses this to stop drawing a confident orientation.
    ratio = float(np.sqrt(max(eigvals[1], 1e-9) / max(eigvals[0], 1e-9)))

    return {
        "x_m": float(centre[0]),
        "z_m": float(centre[2]),
        "width_m": float(hi[0] - lo[0]),
        "depth_m": float(hi[2] - lo[2]),
        "height_m": float(hi[1] - lo[1]),
        "yaw_rad": yaw,
        "yaw_confidence": min(1.0, (ratio - 1.0) / 2.0) if ratio > 1 else 0.0,
        "range_m": float(np.linalg.norm(centre[[0, 2]])),
        "n_points": len(patch),
    }


def build_scene(
    trav: np.ndarray, depth_m: np.ndarray, k: np.ndarray, detections=None
) -> DepthScene:
    """The whole per-frame payload: floor cells plus objects, in metres."""
    points = backproject(depth_m, k)
    objects: list[DepthObject] = []
    for det in detections or []:
        metrics = object_from_box(det["box"], points)
        if metrics is None:
            continue
        # Built in one expression rather than by assigning `name`/`score` onto what
        # `object_from_box` returned. The two stages are genuinely different -- geometry
        # first, then what the detector called it -- and spreading the first into the
        # second says so, where mutating it in place claimed `object_from_box` had known
        # the class all along.
        objects.append(
            {**metrics, "name": det.get("cls", "object"), "score": float(det.get("score", 0.0))}
        )
    # `source` is not decoration. A renderer handed the wrong payload would find `grid`
    # and `objects` where it expected them and draw something plausible out of the other
    # half of this package; this is what lets it refuse instead. `DepthScene` and its
    # `Literal["depth"]` are the same refusal moved to where a checker can make it.
    return {"source": "depth", "grid": bev_grid(points, trav), "objects": objects}
