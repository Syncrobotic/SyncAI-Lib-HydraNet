"""Turn a traversability mask into a metric top-down map.

A mask answers "is this pixel walkable". A navigation stack asks "is there a path three
metres ahead". Same information, different coordinates, and the conversion is geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ground import Camera, GroundPlane, pixel_to_ground

IGNORE = 255


@dataclass(frozen=True)
class BevGrid:
    """The window of floor to map, in metres. Origin is under the camera."""

    x_min: float = -4.0
    x_max: float = 4.0
    z_min: float = 0.5
    z_max: float = 9.0
    cell: float = 0.025

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int((self.z_max - self.z_min) / self.cell),
            int((self.x_max - self.x_min) / self.cell),
        )

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.shape
        xs = self.x_min + (np.arange(cols) + 0.5) * self.cell
        zs = self.z_min + (np.arange(rows) + 0.5) * self.cell
        return np.meshgrid(xs, zs)

    def to_cell(self, x: float, z: float) -> tuple[int, int] | None:
        """Metres -> (row, col) with row 0 at the far edge, as a map is read."""
        if not (self.x_min <= x < self.x_max and self.z_min <= z < self.z_max):
            return None
        rows, _ = self.shape
        return rows - 1 - int((z - self.z_min) / self.cell), int((x - self.x_min) / self.cell)


def project_mask(
    mask: np.ndarray, cam: Camera, plane: GroundPlane, grid: BevGrid = BevGrid()
) -> np.ndarray:
    """Sample a per-pixel mask onto the floor grid. Cells no pixel sees are ``IGNORE``.

    Maps backwards, from each cell to the pixel that sees it. Projecting forwards leaves
    holes in the far field, where one pixel covers many cells.
    """
    from .ground import ground_to_pixel

    h, w = mask.shape
    xx, zz = grid.centres()
    u, v, depth = ground_to_pixel(xx, zz, cam, plane)
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    ok = (depth > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.full(grid.shape, IGNORE, dtype=mask.dtype)
    out[ok] = mask[vi[ok], ui[ok]]
    return out[::-1]  # far field at the top


def place_boxes(boxes: np.ndarray, cam: Camera, plane: GroundPlane) -> np.ndarray:
    """Detections -> floor positions, one (x, z) per box, NaN where it cannot be placed.

    A box carries no range. Where its bottom edge meets the floor does, because that point
    is on the plane. It is the only ground position a single camera recovers, and it is
    wrong for anything not standing on the floor -- a wall-mounted screen, a mug on a table.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0, 2))
    u = (boxes[:, 0] + boxes[:, 2]) / 2
    v = boxes[:, 3]
    x, z = pixel_to_ground(u, v, cam, plane)
    return np.stack([x, z], axis=-1)


def scene(
    mask: np.ndarray,
    cam: Camera,
    plane: GroundPlane,
    *,
    grid: BevGrid = BevGrid(),
    boxes=None,
    labels=None,
    scores=None,
    names=None,
) -> dict:
    """One frame as plain data: the floor, and what is standing on it.

    This is the handoff format for anything that renders -- a 3D view, an RViz overlay, a
    costmap publisher. It carries metres and class ids, and deliberately no colours.
    """
    bev = project_mask(mask, cam, plane, grid)
    objects = []
    if boxes is not None and len(boxes):
        pos = place_boxes(boxes, cam, plane)
        labels = np.asarray(labels).reshape(-1)
        scores = np.asarray(scores).reshape(-1) if scores is not None else np.ones(len(pos))
        for (x, z), lab, sc in zip(pos, labels, scores, strict=True):
            if not (np.isfinite(x) and np.isfinite(z)):
                continue
            objects.append(
                {
                    "x_m": round(float(x), 3),
                    "z_m": round(float(z), 3),
                    "range_m": round(float(np.hypot(x, z)), 3),
                    "label": int(lab),
                    "name": (names or {}).get(int(lab), str(int(lab))),
                    "score": round(float(sc), 4),
                }
            )
    return {
        "camera": {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy},
        "plane": {
            "height_m": round(plane.height, 4),
            "pitch_deg": round(np.degrees(plane.pitch), 3),
            "roll_deg": round(np.degrees(plane.roll), 3),
        },
        "grid": {
            "x_min": grid.x_min,
            "x_max": grid.x_max,
            "z_min": grid.z_min,
            "z_max": grid.z_max,
            "cell_m": grid.cell,
            "shape": list(grid.shape),
        },
        "class_share": {str(c): round(float((bev == c).mean()), 4) for c in np.unique(bev)},
        "objects": objects,
    }, bev
