"""Tracks to the two numbers a shop actually buys: where people stood, and for how long.

Both are computed on the **floor in metres**, not in pixels, and that distinction is the
reason this file can exist at all. A pixel heatmap cannot be compared between two
cameras, cannot be laid over a store plan, and cannot answer "how many square metres of
this aisle go unvisited" -- the near field of an angled camera occupies ten times the
pixels of the far field for the same floor area, so a pixel heatmap is mostly a picture
of the camera's perspective.

`geometry/ground.py` already does the projection. It was built for the robot's BEV and
is exactly what analytics needs, which is worth saying plainly: the ground-plane work
that was deprioritised when 3D was dropped is the prerequisite for the retail numbers.

What the projection assumes, and where it fails in a shop:

* the floor is flat and the camera pose is known. `scripts/fit_camera_from_people.py`
  estimates the pose, and docs/journal/2026-08-14 records that fitting lens distortion
  first is not optional -- a pinhole fit absorbed barrel distortion and put 142 people
  at 1.0-1.2 m tall.
* a person's box bottom is on the floor. **This is the one that breaks indoors**: a
  shopper standing behind a counter has their feet occluded, so the box bottom sits on
  the counter edge and the projected position lands metres too far away. Fixtures are
  where shoppers stand, so this is not a rare case, and it biases dwell toward the far
  side of every counter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.ground import Camera, GroundPlane, pixel_to_ground
from .tracker import Track


def track_ground_path(track: Track, cam: Camera, plane: GroundPlane) -> np.ndarray:
    """(N,2) floor positions in metres for one track's observed frames.

    Rows are NaN where the foot point sits at or above the horizon, which
    `pixel_to_ground` refuses to turn into a very large distance. Keep them as NaN:
    dropping them silently would shorten a path without shortening its duration and
    inflate every speed derived from it.
    """
    if not track.boxes:
        return np.zeros((0, 2))
    boxes = np.stack(track.boxes)
    u = (boxes[:, 0] + boxes[:, 2]) / 2
    v = boxes[:, 3]
    x, z = pixel_to_ground(u, v, cam, plane)
    return np.stack([x, z], axis=-1)


@dataclass
class GroundMap:
    """Occupancy on the floor, in metres. ``cells`` counts track-frames per cell."""

    cells: np.ndarray
    x_min: float
    z_min: float
    cell_m: float

    @property
    def visited_m2(self) -> float:
        return float((self.cells > 0).sum()) * self.cell_m**2

    def busiest(self, n: int = 5) -> list[tuple[float, float, int]]:
        """The n most-occupied cells as (x_m, z_m, track-frames)."""
        flat = np.argsort(self.cells, axis=None)[::-1][:n]
        out = []
        for f in flat:
            r, c = np.unravel_index(f, self.cells.shape)
            if self.cells[r, c] == 0:
                continue
            out.append(
                (
                    self.x_min + (c + 0.5) * self.cell_m,
                    self.z_min + (r + 0.5) * self.cell_m,
                    int(self.cells[r, c]),
                )
            )
        return out


def ground_map(
    paths: list[np.ndarray],
    cell_m: float = 0.25,
    bounds: tuple[float, float, float, float] | None = None,
) -> GroundMap:
    """Rasterise floor paths into an occupancy grid.

    ``cell_m`` is 0.25 m by default: about the footprint of a standing adult, and small
    enough that "in front of this fixture" and "in the aisle" are different cells.
    Finer than the projection is accurate, which is deliberate -- the grid should not be
    the thing that limits resolution, so that improving the pose estimate improves the
    map without a re-raster.
    """
    pts = np.concatenate([p for p in paths if len(p)]) if paths else np.zeros((0, 2))
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return GroundMap(np.zeros((1, 1), dtype=int), 0.0, 0.0, cell_m)
    if bounds is None:
        x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        z_min, z_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    else:
        x_min, x_max, z_min, z_max = bounds
    nx = max(int(np.ceil((x_max - x_min) / cell_m)), 1)
    nz = max(int(np.ceil((z_max - z_min) / cell_m)), 1)
    cells = np.zeros((nz, nx), dtype=int)
    cx = np.clip(((pts[:, 0] - x_min) / cell_m).astype(int), 0, nx - 1)
    cz = np.clip(((pts[:, 1] - z_min) / cell_m).astype(int), 0, nz - 1)
    np.add.at(cells, (cz, cx), 1)
    return GroundMap(cells, x_min, z_min, cell_m)


def dwell_table(tracks: list[Track], fps: float, last_frame: int) -> list[dict]:
    """Per-track dwell, with the incomplete ones marked rather than dropped.

    A track still alive on the final frame belongs to a shopper who had not left when
    the recording stopped, so its dwell is a lower bound. Averaging those in with
    completed visits pulls the mean down by an amount that depends on clip length rather
    than on the shop, which is how a five-minute clip and an hour of footage end up
    disagreeing about the same store.
    """
    out = []
    for t in sorted(tracks, key=lambda t: t.frames[0] if t.frames else 0):
        if not t.frames:
            continue
        span = t.frames[-1] - t.frames[0] + 1
        out.append(
            {
                "track_id": t.track_id,
                "first_frame": t.frames[0],
                "last_frame": t.frames[-1],
                "observed_frames": len(t.frames),
                "dwell_s": span / fps,
                # Frames the track existed but was not observed, i.e. coasted through an
                # occlusion. A high share means the dwell is held together by prediction.
                "coasted": span - len(t.frames),
                "truncated": t.frames[-1] >= last_frame,
            }
        )
    return out
