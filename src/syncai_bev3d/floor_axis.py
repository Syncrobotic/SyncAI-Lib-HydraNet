"""The store's axis, read off the floor it was laid on.

Tiles and planks are laid parallel to the walls, and the walls are what the fixtures stand
against. So the joints on the walkable floor are a witness to the store's axis that needs
no fixture at all -- and the fixture-derived `scene_mesh.store_yaw` was measured against it
on 2026-09-10 to be off by 36 deg on Taichung-cam01 (2.5 vs 38.5), 16 on Tao-Hsin-cam04,
7 on Taichung-cam04 and cam10, 3 on cam11: the blob vote answers "which way are the blobs
elongated", and a depth smear is elongated along the camera's ray, not the wall.

The instrument: strong gradients on the walkable floor of the static plate, each edge's
image tangent carried onto the ground by a finite difference of the ground projection, the
ground directions histogrammed mod 90 deg and weighted by gradient strength. The peak is the
axis. Its sharpness -- the second peak's height, at least 20 deg away, over the first --
says whether the floor has lines at all: 0.10-0.31 on the five cameras where it does,
0.61-0.85 on dark Kaohsiung tiles and on Taichung-cam07 / Tao-Hsin-cam03 / cam15, where the
answer is noise and the caller must fall back.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Gradient percentile on the floor above which a pixel is a joint, not texture.
EDGE_PCT = 92
# Pixels eroded off the walkable outline: the floor's own boundary is the strongest
# edge in the mask and it runs along the fixtures, which is the wrong witness here.
EDGE_ERODE_PX = 6
# Floor further than this reads as a smear of a few pixels per metre.
EDGE_MAX_Z_M = 10.0
# Fewer joint pixels than this and no histogram is a measurement.
EDGE_MIN_PX = 500
# Second peak (>= 20 deg from the first) over the first: above this the floor has no
# dominant line direction. Measured 2026-09-10: 0.10-0.31 where tiles are visible,
# 0.61-0.85 where they are not. The gap is wide; 0.35 sits in it on the safe side.
SHARPNESS_MAX = 0.35
# Half-width of the window around the peak bin the final angle is averaged over.
REFINE_DEG = 6.0
_BINS = 90


def floor_line_axis(plate, walk, gz, geom_ok, ground_points) -> tuple[float | None, float]:
    """(axis in radians mod pi/2, second-peak share), or (None, share) when not sharp.

    `plate` is the RGB plate at the cache's resolution, `walk` the walkable mask, `gz` and
    `geom_ok` the cache's forward metres and validity per pixel, and `ground_points` maps
    pixels *in the plate's own resolution* to floor metres (`CameraFile.ground_points`
    wrapped for the resolution difference by the caller).
    """
    lum = ndimage.gaussian_filter(np.asarray(plate, float).mean(axis=2), 1.2)
    gy, gx = np.gradient(lum)
    mag = np.hypot(gx, gy)
    floor = (
        ndimage.binary_erosion(walk, iterations=EDGE_ERODE_PX) & geom_ok & (gz < EDGE_MAX_Z_M)
    )
    if floor.sum() < EDGE_MIN_PX:
        return None, 1.0
    thr = np.percentile(mag[floor], EDGE_PCT)
    ys, xs = np.nonzero(floor & (mag > thr))
    if len(ys) < EDGE_MIN_PX:
        return None, 1.0
    # the edge's tangent in the image, carried to the ground by a half-pixel step
    t = np.stack([-gy[ys, xs], gx[ys, xs]], axis=1)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    p0 = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(float)
    g0 = ground_points(p0)
    g1 = ground_points(p0 + 0.5 * t)
    ok = np.isfinite(g0).all(axis=1) & np.isfinite(g1).all(axis=1)
    d = g1[ok] - g0[ok]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90
    hist, edges = np.histogram(ang, bins=_BINS, range=(0, 90), weights=mag[ys, xs][ok])
    hist = ndimage.gaussian_filter1d(hist, 1.5, mode="wrap")
    k = int(np.argmax(hist))
    if hist[k] <= 0:
        return None, 1.0
    idx = np.arange(_BINS)
    near = np.minimum((idx - k) % _BINS, (k - idx) % _BINS) <= 20
    second = float(hist[~near].max() / hist[k]) if (~near).any() else 1.0
    if second > SHARPNESS_MAX:
        return None, second
    # The peak bin is a degree wide; the answer is the weighted circular mean of the
    # edges within a few degrees of it, which is where the sub-degree precision is.
    centre = (edges[k] + edges[k + 1]) / 2
    off = (ang - centre + 45) % 90 - 45
    near_peak = np.abs(off) <= REFINE_DEG
    wts = mag[ys, xs][ok][near_peak]
    refined = (
        centre + float(np.average(off[near_peak], weights=wts)) if wts.sum() > 0 else centre
    )
    return float(np.radians(refined % 90)), second
