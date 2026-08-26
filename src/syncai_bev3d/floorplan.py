"""Floor geometry in metres: the BEV raster, and the two shape helpers that go with it.

Three tools grew their own copy of this and the copies had begun to disagree.
`scripts/propose_zones.py` has `BevGrid` and a Douglas-Peucker over a closed ring;
`tools/commissioning/footprints_from_masks.py` has `_simplify` and `_area`;
`tools/commissioning/service_zones.py` has `FloorGrid`, and a `_simplify` and an `_area`
byte-identical to the other tool's except for one guard (`len < 4` against `len < 3`).

`tests/test_scripts_are_not_libraries.py` says why that is worth a module rather than
worth leaving alone, and it is not about neatness: code shared between two entry points
but living beside them sits **outside the wheel, outside the type ratchet and outside the
coverage floor**, so the thing every caller depends on is the thing nothing checks. That
is exactly how four copies of one tracking loop came to disagree about whether to correct
the lens, under a mining run that then concluded "none of the 48 spans is a posture".

**What is deliberately NOT unified here: the contour step.** `propose_zones.outer_contour`
keeps the largest closed contour and `service_zones.polygons` keeps every connected piece,
and that is not a copy that drifted -- it is a fix. Keeping only the largest piece is what
discarded an island counter's staff aisle on every run (`2620e29`), leaving a tracked
shopper standing in a part of the room no zone covered. Merging the two behind one function
would put the two callers back one flag away from that bug. They stay separate until
somebody measures which one `propose_zones` should have.

`scipy` and `contourpy` are imported inside functions, not at module scope, and that is
load-bearing: neither is a dependency of this project (`scipy` arrives via the `bench`
extra, `contourpy` via matplotlib), and nothing under `src/` imports either today. A
top-level import here would make a base `pip install syncai-hydranet` fail on a module the
serving path never calls.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BevGrid", "polygon_area", "shoelace", "simplify_chain", "simplify_ring"]


class BevGrid:
    """A binary occupancy raster over ground coordinates (x lateral, z forward).

    One grid per camera rather than one per fixture, because zones on a floor have to
    **partition** it: a shopper standing between two display tables is at one of them, and
    a set of overlapping per-fixture bands makes `journeys` report a route through three
    zones without anyone moving. That was the first version's real defect, visible on
    Taichung-cam01 as three zones over one aisle.

    `pad` widens the extent past the observed points so a contour taken on the raster has
    a background cell to close against on every side.
    """

    def __init__(self, x: np.ndarray, z: np.ndarray, cell: float, pad: float = 1.0):
        self.cell = float(cell)
        self.x0 = float(np.floor((x.min() - pad) / cell) * cell)
        self.z0 = float(np.floor(max(z.min() - pad, 0.0) / cell) * cell)
        self.nx = int(np.ceil((x.max() + pad - self.x0) / cell))
        self.nz = int(np.ceil((z.max() + pad - self.z0) / cell))

    @classmethod
    def over(cls, points_m: np.ndarray, cell: float, pad: float = 1.0) -> BevGrid:
        """The (N, 2) form, which is what every caller that projects a mask actually holds.

        `z0` is **not** floored at zero here, unlike the two-array constructor: a floor
        mask's nearest point can sit behind the camera origin on a wide mount, and clamping
        it silently drops that row of cells.
        """
        grid = cls.__new__(cls)
        grid.cell = float(cell)
        lo = points_m.min(axis=0) - pad
        hi = points_m.max(axis=0) + pad
        grid.x0, grid.z0 = float(lo[0]), float(lo[1])
        grid.nx = int(np.ceil((hi[0] - grid.x0) / cell))
        grid.nz = int(np.ceil((hi[1] - grid.z0) / cell))
        return grid

    def raster(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Points in metres -> a (nz, nx) bool grid. Points outside the extent are dropped."""
        cols = ((x - self.x0) / self.cell).astype(int)
        rows = ((z - self.z0) / self.cell).astype(int)
        ok = (cols >= 0) & (cols < self.nx) & (rows >= 0) & (rows < self.nz)
        grid = np.zeros((self.nz, self.nx), dtype=bool)
        grid[rows[ok], cols[ok]] = True
        return grid

    def raster_points(self, points_m: np.ndarray) -> np.ndarray:
        return self.raster(points_m[:, 0], points_m[:, 1])

    def to_metres(self, cells: np.ndarray) -> np.ndarray:
        """contourpy index coordinates (x=col, y=row, cell centres at integers) -> metres."""
        out = np.empty_like(np.asarray(cells, dtype=float))
        out[:, 0] = self.x0 + (cells[:, 0] + 0.5) * self.cell
        out[:, 1] = self.z0 + (cells[:, 1] + 0.5) * self.cell
        return out


def shoelace(poly: np.ndarray) -> float:
    """Signed area; the sign is the winding. `scripts/propose_zones.py`'s formula.

    See `polygon_area` for why there are two of these and why they must not be collapsed
    into one.
    """
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - z * np.roll(x, -1)))


def polygon_area(poly: np.ndarray) -> float:
    """Unsigned area -- square metres, for a zone. `tools/commissioning/`'s formula.

    **This is not `abs(shoelace(poly))`, and the difference is deliberate.** The two
    expressions are the same identity and agree to about 1e-16; they do not agree
    bit-for-bit, because one sums `x*roll(z,-1) - z*roll(x,-1)` elementwise and the other
    takes two dot products, and floating-point addition is not associative.

    That is normally beneath notice. It is not here, because **both callers sort by area
    and both name the result by rank**: `service_zones` writes `fixture_01..NN` in
    descending area order, and a shop floor produces many one-cell triangles of exactly
    equal area. `list.sort` is stable, so a tie is broken by insertion order -- until one
    of the tied keys moves in its last bit, at which point two zones swap names. Measured
    while preparing this change: 110 polygons from one synthetic floor came back as the
    same 110 shapes in a different order, and the only difference between the two runs was
    which of these two expressions computed the sort key.

    A renamed zone is not cosmetic. `camera.json` stores the name, a manager's rule keys
    on it, and every dwell already reported against `fixture_06` silently becomes a
    different fixture's. So each caller keeps the arithmetic it was measured with, and the
    two functions stay side by side with this note between them.
    """
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))))


def simplify_chain(chain: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker over an **open** chain, keeping both endpoints.

    Iterative rather than recursive, so a base line one point per image column wide cannot
    blow the stack.

    A chain shorter than three points is returned untouched: there is nothing between the
    endpoints to drop.

    **The two tools this came from used different guards and it is not drift.**
    `footprints_from_masks` guards at `< 3`, which is right for an open chain.
    `service_zones` guards at `< 4` because what it hands in is a **closed ring** written
    as a chain, and there the middle vertex of a 3-point ring is the ring: dropping it
    leaves two points, zero area, and a zone `Zone.contains` answers False for forever
    and silently. So this keeps the open-chain rule, `< 3`, and the ring caller keeps its
    own guard at the call site with that reason written next to it. Verified on a
    near-collinear triangle: `[[0,0],[1,0.001],[2,0]]` at tol 0.08 comes back whole under
    the `< 4` guard and as `[[0,0],[2,0]]` without it.
    """
    if len(chain) < 3:
        return chain
    keep = np.zeros(len(chain), dtype=bool)
    keep[0] = keep[-1] = True
    _douglas_peucker(chain, 0, len(chain) - 1, tol, keep)
    return chain[keep]


def simplify_ring(poly: np.ndarray, tol: float, max_vertices: int) -> np.ndarray:
    """Douglas-Peucker on a **closed** ring, tightening until under the vertex budget.

    A ring has no endpoints to anchor, so it is split at two far-apart vertices into two
    open chains and each is simplified. Splitting at an arbitrary pair instead pins a
    vertex that may sit mid-edge, which reads afterwards as a kink nobody put there.
    """
    n = len(poly)
    if n <= 3:
        return poly
    a = 0
    b = int(np.hypot(*(poly - poly[a]).T).argmax())
    if b == a:
        b = n // 2
    lo, hi = min(a, b), max(a, b)
    while True:
        keep = np.zeros(n, dtype=bool)
        keep[[lo, hi]] = True
        _douglas_peucker(poly, lo, hi, tol, keep)
        rolled = np.vstack([poly[hi:], poly[: lo + 1]])
        keep_r = np.zeros(len(rolled), dtype=bool)
        keep_r[[0, len(rolled) - 1]] = True
        _douglas_peucker(rolled, 0, len(rolled) - 1, tol, keep_r)
        idx = set(np.flatnonzero(keep).tolist())
        idx |= {(hi + i) % n for i in np.flatnonzero(keep_r)}
        out = poly[sorted(idx)]
        if len(out) <= max_vertices:
            return out
        tol *= 1.5


def _douglas_peucker(
    points: np.ndarray, first: int, last: int, tol: float, keep: np.ndarray
) -> None:
    """Mark the indices worth keeping between `first` and `last`. Iterative, in place."""
    stack = [(first, last)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = points[b] - points[a]
        norm = float(np.hypot(*seg))
        rel = points[a + 1 : b] - points[a]
        # 2-D scalar cross product written out: np.cross on 2-D inputs is deprecated.
        dist = (
            np.hypot(rel[:, 0], rel[:, 1])
            if norm < 1e-12
            else np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm
        )
        i = int(dist.argmax())
        if dist[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
