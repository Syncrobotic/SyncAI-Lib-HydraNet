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
    wts = mag[ys, xs][ok]
    ang180 = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
    # Over 180 deg, not 90: a floor has two line families, and folding them mod 90 only
    # works if the projection keeps them 90 deg apart. On Tao-Hsin-cam15 (2026-09-10)
    # they came out 100 deg apart -- the assumed vfov skews the ground -- and folded
    # they were two peaks 10 deg apart that failed the sharpness gate. The families are
    # found separately; the axis is the stronger one; their departure from 90 deg is a
    # calibration reading the caller may want (`floor_line_axes`).
    a1, _a2, second = _two_families(ang180, wts)
    if a1 is None or second > SHARPNESS_MAX:
        return None, second
    return float(np.radians(a1 % 90)), second


def floor_line_axes(plate, walk, gz, geom_ok, ground_points):
    """(stronger family deg, other family deg or None, second-peak share) over 180 deg.
    The two should be 90 deg apart on a calibrated floor; how far they are not is how
    far the calibration is off."""
    lum = ndimage.gaussian_filter(np.asarray(plate, float).mean(axis=2), 1.2)
    gy, gx = np.gradient(lum)
    mag = np.hypot(gx, gy)
    floor = (
        ndimage.binary_erosion(walk, iterations=EDGE_ERODE_PX) & geom_ok & (gz < EDGE_MAX_Z_M)
    )
    if floor.sum() < EDGE_MIN_PX:
        return None, None, 1.0
    thr = np.percentile(mag[floor], EDGE_PCT)
    ys, xs = np.nonzero(floor & (mag > thr))
    if len(ys) < EDGE_MIN_PX:
        return None, None, 1.0
    t = np.stack([-gy[ys, xs], gx[ys, xs]], axis=1)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    p0 = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(float)
    g0 = ground_points(p0)
    g1 = ground_points(p0 + 0.5 * t)
    ok = np.isfinite(g0).all(axis=1) & np.isfinite(g1).all(axis=1)
    d = g1[ok] - g0[ok]
    return _two_families(np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180, mag[ys, xs][ok])


def _two_families(ang180, wts):
    bins = 2 * _BINS
    hist, edges = np.histogram(ang180, bins=bins, range=(0, 180), weights=wts)
    hist = ndimage.gaussian_filter1d(hist, 1.5, mode="wrap")
    k1 = int(np.argmax(hist))
    if hist[k1] <= 0:
        return None, None, 1.0
    idx = np.arange(bins)
    dist1 = np.minimum((idx - k1) % bins, (k1 - idx) % bins)
    # the other family: the best bin 70-110 deg from the first
    window = (dist1 >= 70) & (dist1 <= 110)
    k2 = int(np.argmax(np.where(window, hist, -1)))
    dist2 = np.minimum((idx - k2) % bins, (k2 - idx) % bins)
    rest = (dist1 > 20) & (dist2 > 20)
    second = float(hist[rest].max() / hist[k1]) if rest.any() else 1.0

    def refine(k):
        centre = (edges[k] + edges[k + 1]) / 2
        off = (ang180 - centre + 90) % 180 - 90
        near = np.abs(off) <= REFINE_DEG
        return (
            centre + float(np.average(off[near], weights=wts[near]))
            if wts[near].sum()
            else centre
        )

    a1 = refine(k1) % 180
    a2 = refine(k2) % 180 if hist[k2] > 0.2 * hist[k1] else None
    return a1, a2, second


# Joint periods are searched between these; below is texture, above is not a tile.
PERIOD_MIN_M, PERIOD_MAX_M = 0.25, 1.30
_PERIOD_BIN_M = 0.01
# Autocorrelation at twice the candidate pitch, below which the candidate is not periodic.
HARMONIC_MIN = 0.15
# Edges within this many degrees of an axis vote for the period across that axis.
_ALONG_DEG = 12.0


def floor_period(plate, walk, gz, geom_ok, ground_points, axis) -> tuple[float | None, float]:
    """(joint period in metres across the floor's lines, autocorrelation at it), or (None, 0).

    The same edges `floor_line_axis` reads, rotated into the store frame at `axis`: the
    edges running along one axis are histogrammed by their coordinate across it at 1 cm,
    and the histogram's autocorrelation has its first local maximum past PERIOD_MIN_M at
    the joint pitch. Both axes are tried and the stronger answers. In the camera's own
    metres -- which is the point: cameras over one floor must agree, and against a known
    tile they give the scale outright, with no person in the frame.
    """
    lum = ndimage.gaussian_filter(np.asarray(plate, float).mean(axis=2), 1.2)
    gy, gx = np.gradient(lum)
    mag = np.hypot(gx, gy)
    floor = (
        ndimage.binary_erosion(walk, iterations=EDGE_ERODE_PX) & geom_ok & (gz < EDGE_MAX_Z_M)
    )
    if floor.sum() < EDGE_MIN_PX:
        return None, 0.0
    thr = np.percentile(mag[floor], EDGE_PCT)
    ys, xs = np.nonzero(floor & (mag > thr))
    if len(ys) < EDGE_MIN_PX:
        return None, 0.0
    t = np.stack([-gy[ys, xs], gx[ys, xs]], axis=1)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    p0 = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(float)
    g0 = ground_points(p0)
    g1 = ground_points(p0 + 0.5 * t)
    ok = np.isfinite(g0).all(axis=1) & np.isfinite(g1).all(axis=1)
    g0, d = g0[ok], g1[ok] - g0[ok]
    c, s = np.cos(axis), np.sin(axis)
    u, v = g0[:, 0] * c + g0[:, 1] * s, -g0[:, 0] * s + g0[:, 1] * c
    ang = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) - np.degrees(axis)) % 90
    along_u = np.minimum(ang, 90 - ang) <= _ALONG_DEG  # runs along u: votes across, on v
    best: tuple[float | None, float] = (None, 0.0)
    for coords in (v[along_u], u[~along_u]):
        got = _period_of(coords)
        if got is not None and got[1] > best[1]:
            best = got
    return best


def _period_of(coords) -> tuple[float, float] | None:
    if len(coords) < EDGE_MIN_PX:
        return None
    lo, hi = np.percentile(coords, [2, 98])
    c = coords[(coords >= lo) & (coords <= hi)]
    nbins = int((hi - lo) / _PERIOD_BIN_M) + 1
    if nbins < 3 * int(PERIOD_MIN_M / _PERIOD_BIN_M):
        return None
    h, _ = np.histogram(c, bins=nbins, range=(lo, hi))
    # Detrended: the edge density ramps with distance from the camera, and that ramp's
    # autocorrelation is positive at every lag -- a noise plate then "repeats" at 0.3 m
    # with a harmonic at 0.6. What is left after the ramp is the joints.
    h = h - ndimage.gaussian_filter1d(h.astype(float), PERIOD_MAX_M / 2 / _PERIOD_BIN_M)
    ac = np.correlate(h, h, "full")[len(h) - 1 :]
    ac = ac / (ac[0] + 1e-9)
    a = int(PERIOD_MIN_M / _PERIOD_BIN_M)
    b = min(int(PERIOD_MAX_M / _PERIOD_BIN_M), len(ac) - 2)
    seg = ac[a:b]
    if len(seg) < 3:
        return None
    # A joint pitch repeats: the autocorrelation peaks at p AND at 2p. Random edges
    # (the cache's aliasing, scuffs, a plank's grain) put a peak near PERIOD_MIN_M and
    # nothing at its double -- measured on noise plates, 0.29-0.37 m at 0.43-0.52, with
    # no second harmonic; a drawn 0.60 m grid gives 0.60 with its double at 1.20.
    peaks = [
        i
        for i in range(1, len(seg) - 1)
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.05
    ]
    best = None
    for i in peaks:
        k = i + a
        if 2 * k + 2 >= len(ac):
            continue
        harmonic = float(ac[2 * k - 2 : 2 * k + 3].max())
        if harmonic < HARMONIC_MIN:
            continue
        score = float(seg[i])
        if best is None or score > best[1]:
            best = (float(k * _PERIOD_BIN_M), score)
    return best
