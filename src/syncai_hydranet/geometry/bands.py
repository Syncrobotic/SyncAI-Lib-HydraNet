"""The floor beside a fixture, rather than the floor nearest to it.

**The failure this exists for, measured 2026-09-08.** The commissioned fixture zones TILE
the walkable floor: on Taichung-cam04 twelve zones sum to 38.9 m2 against a 26.2 m2
walkable polygon, and 95.6% of that polygon is claimed by some fixture, with only 0.2%
claimed twice. Every point on the floor therefore belongs to whichever display is nearest,
with no upper bound on the distance -- so "dwell at fixture F07" has meant "standing in the
right-hand quarter of the room", not "engaging with a display". Merging the fragments makes
the labels coarser but does not touch that; only a distance does.

A band is a **predicate, not a polygon**, and that is a design choice rather than a
limitation. "Within a metre of a display" is a distance query; turning it into a polygon
would need boolean geometry this project does not depend on, and would quantise the answer
on the way. Callers compose it with the zone they already have:

    band = Band(contact_line(mask, cf), width_m=1.0)
    engaged = zone.contains(path) & band.contains(path)

WHAT IT CANNOT DO, AND WHY THE NUMBER IT PRODUCES IS AN UPPER BOUND
`contact_line` reads a *furniture* mask and cannot tell a display from a wall: a band drawn
along a blank wall counts exactly like a band along a shelf. The tool that would separate
them is `tools/commissioning/footprints_from_masks.py`, whose own header records that it
does not yet produce usable footprints. Until it does, a caller quoting "time spent at a
fixture" is quoting time spent beside furniture, and should say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def contact_line(mask: np.ndarray, cam_file, *, min_component_px: int = 200) -> np.ndarray:
    """Ground metres of where a furniture mask meets the floor.

    The lowest mask pixel in each column of each connected component: for an object
    standing on the floor that pixel is its contact with it, which is the only part of a
    mask whose ground projection means anything. Projecting the whole mask instead puts a
    shelf's top shelf several metres behind the shop.

    ``min_component_px`` drops speckle. Components below it are mask noise, and a stray
    pixel projected to the floor is a fixture that is not there.

    Returns (N, 2) floor metres, with rays at or above the horizon dropped -- a mask
    legitimately reaches past it, which is `ground_points`' documented "drop" case.
    """
    from scipy import ndimage

    m = np.asarray(mask)
    m = (m if m.ndim == 2 else m[..., 0]) > 0
    if not m.any():
        return np.zeros((0, 2))
    lab, n = ndimage.label(m)
    px = []
    for c in range(1, n + 1):
        comp = lab == c
        if comp.sum() < min_component_px:
            continue
        for x in np.where(comp.any(axis=0))[0]:
            px.append((x + 0.5, np.where(comp[:, x])[0].max() + 0.5))
    if not px:
        return np.zeros((0, 2))
    g = cam_file.ground_points(
        np.asarray(px, dtype=float), above_horizon="drop", what="fixture contact line"
    )
    return g[np.isfinite(g).all(axis=1)]


@dataclass(frozen=True)
class Band:
    """Floor within ``width_m`` of any contact point.

    ``width_m`` is the caller's, deliberately: it is a claim about what counts as being at
    a display, and this module has no way to measure the right answer. On Taichung-cam04
    the peak sample put 42% of floor-time inside 1.0 m and 55% inside 1.5 m, against 45%
    inside some commissioned zone -- so the choice moves the headline and has to be stated
    beside it.
    """

    points_m: np.ndarray
    width_m: float

    def __post_init__(self) -> None:
        p = np.asarray(self.points_m, dtype=float).reshape(-1, 2)
        object.__setattr__(self, "points_m", p)
        if not self.width_m > 0:
            raise ValueError(f"width_m must be positive, got {self.width_m!r}")

    def contains(self, xy: np.ndarray) -> np.ndarray:
        """(N,) bool for (N,2) floor metres. All False when the band has no points.

        A band with no contact points is a fixture whose mask projected to nothing. It
        answers False rather than True so a missing mask reads as "no engagement measured"
        instead of silently promoting the whole floor to engaged.
        """
        q = np.asarray(xy, dtype=float).reshape(-1, 2)
        if not len(self.points_m) or not len(q):
            return np.zeros(len(q), dtype=bool)
        from scipy.spatial import cKDTree

        d, _ = cKDTree(self.points_m).query(q)
        return np.asarray(d <= self.width_m)
