"""Turn a traversability mask into a metric top-down map.

A mask answers "is this pixel walkable". A navigation stack asks "is there a path three
metres ahead". Same information, different coordinates, and the conversion is geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from syncai_hydranet.geometry.ground import Camera, GroundPlane, pixel_to_ground
from syncai_hydranet.labels import IGNORE

from .scene_types import PlaneObject, PlaneScene


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
    from syncai_hydranet.geometry.ground import ground_to_pixel

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


def box_extents(boxes: np.ndarray, cam: Camera, plane: GroundPlane) -> np.ndarray:
    """Detections -> (width_m, height_m) on the floor, NaN where the box cannot be placed.

    Width comes from the two bottom corners, which are both on the plane, so it is as
    real as the plane is. Height is the top edge's ray evaluated at the base's forward
    distance -- exact for something standing upright and directly seen, wrong for a
    leaning object or one whose top is occluded, in the same way and for the same reason
    as `place_boxes`.

    Sizes are what turns a dot on a map into something a person can recognise, so they
    belong in the payload rather than in a renderer's guesswork.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0, 2))
    xl, yt, xr, yb = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_left, z_left = pixel_to_ground(xl, yb, cam, plane)
    x_right, z_right = pixel_to_ground(xr, yb, cam, plane)
    width = np.abs(x_right - x_left)

    u_mid = (xl + xr) / 2
    rays = np.stack(
        [(u_mid - cam.cx) / cam.fx, (yt - cam.cy) / cam.fy, np.ones_like(u_mid)], axis=-1
    )
    level = rays @ plane.rotation
    z_base = (z_left + z_right) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = z_base / level[..., 2]
    height = plane.height - t * level[..., 1]
    height = np.where(np.isfinite(height) & (height > 0), height, np.nan)
    return np.stack([width, height], axis=-1)


def ray_reach(bev: np.ndarray, grid: BevGrid, n_rays: int = 240):
    """Per bearing: how far the floor reached. Also the per-cell bearing and range.

    The one reduction both the flat map's free-space filter and the perspective renderer
    are built on, so that they cannot disagree about where the floor ends.

    ``bev`` is a map as ``project_mask`` returns it, **far field at row 0**. The grid's
    own ``centres()`` runs the other way, so it is reversed here rather than at each call
    site: a caller that forgets computes every range mirrored and produces a map that is
    smooth, plausible and wrong about the only distance anyone reads off it.
    """
    xx, zz = grid.centres()
    xx, zz = xx[::-1], zz[::-1]
    ang = np.arctan2(xx, zz)
    rng = np.hypot(xx, zz)
    lo, hi = float(ang.min()), float(ang.max())
    ray = np.clip(
        ((ang - lo) / max(hi - lo, 1e-9) * (n_rays - 1)).astype(np.int32), 0, n_rays - 1
    )
    walkable = (bev == 2) | (bev == 1)
    reach = np.zeros(n_rays, dtype=np.float64)  # last walkable range per bearing
    np.maximum.at(reach, ray[walkable], rng[walkable])
    angles = lo + (np.arange(n_rays) + 0.5) / n_rays * (hi - lo)
    return angles, reach, ray, rng


def free_space_map(bev: np.ndarray, grid: BevGrid, n_rays: int = 240) -> np.ndarray:
    """Keep what a camera can actually assert: free space, its boundary, nothing beyond.

    Projecting a mask onto the floor treats every pixel as if it lay on the floor. For
    walkable pixels that is true. For a wall or a shelf it is not -- the ray through a
    wall pixel meets the ground plane somewhere well past the wall -- so the naive map
    paints the whole far field `blocked`, and a map that says "obstacle" where it means
    "I cannot see" will be trusted in the wrong direction.

    What a single camera knows along each ray is: floor out to the last walkable cell,
    something standing at that boundary, and nothing at all behind it. That is what this
    returns. It is also why the far field of an honest camera map is mostly empty, which
    is the correct shape for one camera and the reason a robot carries a second sensor.
    """
    _, reach, ray, rng = ray_reach(bev, grid, n_rays)
    walkable = (bev == 2) | (bev == 1)
    out = np.full_like(bev, IGNORE)
    # The floor is copied whole. This used to be gated on `rng <= reach[ray]`, which reads
    # like a range filter and is not one: a cell's own bearing's reach is the maximum over
    # that bearing's walkable cells, so the test is true for every one of them. What makes
    # the far field empty is that non-walkable cells are never copied at all.
    out[walkable] = bev[walkable]
    # The boundary: one cell thick, where the floor stopped. Rays that never saw floor
    # get no boundary, because nothing was established about them.
    edge = (np.abs(rng - reach[ray]) <= grid.cell * 1.5) & (reach[ray] > 0) & ~walkable
    out[edge] = 0
    return out


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
) -> tuple[PlaneScene, np.ndarray]:
    """One frame as plain data: the floor, and what is standing on it.

    This is the handoff format for anything that renders -- a 3D view, an RViz overlay, a
    costmap publisher. It carries metres and class ids, and deliberately no colours.
    """
    bev = project_mask(mask, cam, plane, grid)
    objects: list[PlaneObject] = []
    if boxes is not None and len(boxes):
        pos = place_boxes(boxes, cam, plane)
        ext = box_extents(boxes, cam, plane)
        labels = np.asarray(labels).reshape(-1)
        scores = np.asarray(scores).reshape(-1) if scores is not None else np.ones(len(pos))
        for (x, z), (bw, bh), lab, sc in zip(pos, ext, labels, scores, strict=True):
            if not (np.isfinite(x) and np.isfinite(z)):
                continue
            objects.append(
                {
                    "x_m": round(float(x), 3),
                    "z_m": round(float(z), 3),
                    "range_m": round(float(np.hypot(x, z)), 3),
                    # None rather than a filler: a renderer that substitutes its own
                    # default should have to say so, not inherit a number that looks
                    # measured.
                    "width_m": round(float(bw), 3) if np.isfinite(bw) else None,
                    "height_m": round(float(bh), 3) if np.isfinite(bh) else None,
                    "label": int(lab),
                    "name": (names or {}).get(int(lab), str(int(lab))),
                    "score": round(float(sc), 4),
                }
            )
    return {
        # Which evidence built this. `depth_scene.build_scene` emits the same word with
        # the value "depth", and the two payloads share key names only where the quantity
        # is the same -- see that module's docstring.
        "source": "plane",
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
