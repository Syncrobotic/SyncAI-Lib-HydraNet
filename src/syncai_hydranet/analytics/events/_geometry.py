"""Plane geometry the detectors share, vectorised over numpy and nothing else.

Four helpers, all private, all pure. They were interleaved with their callers in the
single-file version, which is why `_segments_cross` sat 180 lines below the section
header that describes it -- next to `line_events`, its only caller at the time. It has
two now.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- geometry


def _point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd ray casting, vectorised over points. No matplotlib, no shapely.

    Both are heavier dependencies than 12 lines, and this project already refuses a
    dependency for a small numeric job (`tracker.py` on scipy and the Hungarian
    assignment). The boundary case is left as-is rather than tuned: a track sitting
    exactly on a zone edge for several frames flips, and that is visible in the frame
    span rather than hidden by an epsilon whose value nobody measured.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    poly = np.asarray(polygon, dtype=float)
    if len(poly) < 3:
        raise ValueError(f"a zone needs at least 3 corners, got {len(poly)}")
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1, strict=True):
        straddles = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at_y = (bx - ax) * (y - ay) / (by - ay) + ax
        inside ^= straddles & (x < x_at_y)
    return inside & np.isfinite(x) & np.isfinite(y)


def _cross2(d: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The z component of a 2-D cross product, which is all the side tests need.

    `np.cross` on 2-D vectors is deprecated in NumPy 2.0 and scheduled for removal;
    writing the one component out is smaller than a deprecation shim and says what is
    actually being computed.
    """
    p = np.atleast_2d(p)
    return d[0] * p[..., 1] - d[1] * p[..., 0]


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Index spans where ``flags`` is True, as inclusive (start, end) pairs."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _segments_cross(p0, p1, q0, q1) -> bool:
    """Do segments p0-p1 and q0-q1 properly intersect? Collinear counts as no."""

    def orient(a, b, c):
        return np.sign(_cross2(b - a, c - a))

    o1, o2 = orient(p0, p1, q0), orient(p0, p1, q1)
    o3, o4 = orient(q0, q1, p0), orient(q0, q1, p1)
    return bool(o1 != o2 and o3 != o4 and o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0)
