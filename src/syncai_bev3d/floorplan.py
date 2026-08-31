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
        idx |= {(hi + i) % n for i in np.flatnonzero(keep_r).tolist()}
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


# Wall extraction, in the store frame. The numbers are metres and each is a claim.
WALL_BIN = 0.20  # across-axis bin: a wall's cells scatter this much through mask noise
WALL_GAP_TOL = 0.60  # a gap this small is occlusion; a door is 0.8-1.0 m and must survive
WALL_MIN_RUN_M = 1.20  # shorter than this is a patch of white surface, not a wall
WALL_MIN_CELLS = 40  # per candidate line, before it is allowed to be a wall at all
WALL_CORNER_TOL = 0.80  # extend two perpendicular runs this far to make them meet
# How far either side of an accepted run its evidence is consumed. A wall seen obliquely
# is a band of cells, not a line, so this is the band's half-width rather than a tolerance.
WALL_SUPPRESS = 0.55


def wall_runs(cells_u, cells_v):
    """Wall runs fitted to the whole `wall` point set, not to its connected components.

    **The components are not walls, and that is why the render showed eight to fourteen
    disconnected panes.** Measured on 2026-08-28 in the store frame: Taichung-cam11's five
    `wall` components are 1.12 x 1.07, 1.44 x 0.85, 1.22 x 0.27, 1.23 x 0.86 and
    1.11 x 0.63 m, and Taichung-cam04's eight are 0.6-1.5 m long and 0.3-0.8 m thick. A
    real shop wall is four to eight metres. What the mask holds is not walls but the
    patches of white surface still visible between the fixtures standing in front of them,
    and drawing each patch as its own 2.4 m slab is what put a row of floating panes in the
    scene. Merging the boxes afterwards barely helped -- 5 -> 5, 8 -> 7, 6 -> 6 -- because
    boxes fitted to fragments are not collinear enough to merge.

    So this does the step the scan-to-BIM sequence actually specifies, and does it in the
    order that sequence gives: **extract the wall axes from the point set first**, then
    split each axis into runs at real gaps, then intersect perpendicular runs at corners.
    Fitting a line through five patches that share a wall recovers the wall; fitting a box
    to each patch never can, however carefully the boxes are merged afterwards.

    Closing the runs into a room polygon -- the last step of that sequence -- is
    deliberately not done. A fixed camera sees part of one store, so closure would draw
    walls along the edge of the field of view, where the shop continues.

    Takes the `wall` cells' `(u, v)` coordinates. Returns `(axis, perp, lo, hi)` tuples.
    """
    used = np.zeros(len(cells_u), bool)
    cands = []
    for axis, along, across in (("u", cells_u, cells_v), ("v", cells_v, cells_u)):
        if not len(across):
            continue
        lo_edge = float(across.min())
        idx = ((across - lo_edge) / WALL_BIN).astype(int)
        for b in np.unique(idx):
            # two adjacent bins together, so a wall sitting on a bin boundary is not split
            sel = (idx >= b) & (idx <= b + 1)
            if sel.sum() < WALL_MIN_CELLS:
                continue
            order = np.argsort(along[sel])
            pos = along[sel][order]
            members = np.nonzero(sel)[0][order]
            breaks = np.nonzero(np.diff(pos) > WALL_GAP_TOL)[0] + 1
            for piece, mem in zip(
                np.split(pos, breaks), np.split(members, breaks), strict=True
            ):
                if len(piece) < WALL_MIN_CELLS or piece[-1] - piece[0] < WALL_MIN_RUN_M:
                    continue
                lo_p, hi_p = np.percentile(across[mem], [3, 97])
                cands.append(
                    (piece[-1] - piece[0], axis, float(np.median(across[mem])),
                     float(piece[0]), float(piece[-1]), float(hi_p - lo_p), mem)
                )  # fmt: skip

    # Longest first, then suppress the neighbourhood -- not just the cells that voted.
    #
    # **Marking only the member cells is not enough, and the first version did exactly
    # that.** Every wall votes in several adjacent bins, so one wall produced a stack of
    # parallel lines a few centimetres apart, each with its own untouched cells: the fleet
    # went to 26 and 28 runs per camera, longer than before and more numerous, which is
    # worse. Suppressing a band either side of an accepted run is the non-maximum
    # suppression any Hough-style extraction needs and this one was missing.
    out: list[list] = []
    for _len_m, axis, perp, lo, hi, thick, mem in sorted(cands, key=lambda t: -t[0]):
        if used[mem].mean() > 0.5:
            continue
        along, across = (cells_u, cells_v) if axis == "u" else (cells_v, cells_u)
        used |= (
            (along >= lo - WALL_GAP_TOL)
            & (along <= hi + WALL_GAP_TOL)
            & (np.abs(across - perp) <= WALL_SUPPRESS)
        )
        out.append([axis, perp, lo, hi, thick])

    for a in out:
        for b in out:
            if a[0] == b[0]:
                continue
            if not (b[2] - WALL_CORNER_TOL <= a[1] <= b[3] + WALL_CORNER_TOL):
                continue
            if 0 < a[2] - b[1] <= WALL_CORNER_TOL:
                a[2] = b[1]
            elif 0 < b[1] - a[3] <= WALL_CORNER_TOL:
                a[3] = b[1]
    return [tuple(v) for v in out]


# A run with this much floor on its thinner side is not a boundary of the room.
FLOOR_BOTH_SIDES = 0.25
FLOOR_BAND_M = 1.20  # how far either side of a run its floor evidence is counted


def floor_both_sides(run, floor_u, floor_v) -> float:
    """How much floor lies on the *thinner* side of a run, as a fraction of the thicker.

    **The one relation that separates a wall from a counter, and neither a shape nor a
    size can do it.** A 7.9 m run 15 cm thick is a perfectly plausible wall by every
    per-object check this file has; the thing that makes it not a wall is that a shopper
    can stand on both sides of it. A room boundary has floor on one side and the outside
    world on the other.

    Measured 2026-08-28 across four cameras, roughly half of the fitted `wall` runs fail
    it -- including Tao-Hsin-cam04's two longest at 7.9 m (724 floor cells against 550) and
    7.2 m (626 against 430). Those are the counter runs that PLAN 7.21 recorded as being
    classified `wall` by both teachers on a white-fixture store. Extracting them as long
    continuous runs made them *more* convincing, not less: before this test the scene put a
    pair of eight-metre walls through the middle of a phone shop.
    """
    axis, perp, lo, hi = run[0], run[1], run[2], run[3]
    along, across = (floor_u, floor_v) if axis == "u" else (floor_v, floor_u)
    near = (along >= lo) & (along <= hi) & (np.abs(across - perp) <= FLOOR_BAND_M)
    left = int((near & (across < perp)).sum())
    right = int((near & (across > perp)).sum())
    return min(left, right) / max(max(left, right), 1)


# Fixture regularisation. Every scan-to-BIM and structured-modelling pipeline has a pass
# like this between fitting and meshing, and this one did not: each component was fitted
# and drawn without ever being compared with its neighbours, so every relationship error
# in the scene came from that gap. The numbers are metres and fractions.
FIXTURE_CONTAINED_FRAC = 0.60  # overlap above this share of the smaller box: one object
FIXTURE_SNAP_M = 0.20  # a fixture this close to a wall is against it, and is moved flush


def _overlap(a, b) -> tuple[float, float]:
    """Overlap of two axis-aligned boxes along each axis, in metres. Negative is a gap."""
    return (
        min(a[1], b[1]) - max(a[0], b[0]),
        min(a[3], b[3]) - max(a[2], b[2]),
    )


def resolve_overlaps(boxes):
    """Fixtures that interpenetrate, resolved by containment or by shrinking to contact.

    `boxes` are `(u0, u1, v0, v1)` in the store frame. Returns a list the same length,
    with `None` where a box was absorbed into another.

    **Two fixtures cannot occupy the same floor, and until this existed they routinely
    did.** Nothing in the pipeline compared one fitted component with another, so a
    counter standing in front of a wall -- which PLAN 7.21 measured arriving as a single
    welded object 263 times out of 503 merges on Taichung-cam01 -- could also arrive as
    two boxes drawn through each other. A reviewer reading the render calls that
    "a jumble of boxes", and no per-object check can see it: each box on its own is a
    perfectly plausible fixture.

    The rule is deliberately asymmetric. A box more than `FIXTURE_CONTAINED_FRAC` inside
    another is not a second fixture, it is the same one fitted twice from different
    evidence -- the class mask and the re-classified wall run can both cover it -- so it
    is absorbed rather than shrunk. Anything else is two real fixtures whose boxes have
    grown into each other, and the smaller gives way along whichever axis it overlaps
    least, which is the direction its own extent was least certain in.
    """
    out = [list(map(float, b)) for b in boxes]
    order = sorted(
        range(len(out)),
        key=lambda i: -((out[i][1] - out[i][0]) * (out[i][3] - out[i][2])),
    )
    dropped = set()
    for rank, i in enumerate(order):
        if i in dropped:
            continue
        for j in order[rank + 1 :]:
            if j in dropped:
                continue
            ou, ov = _overlap(out[i], out[j])
            if ou <= 0 or ov <= 0:
                continue
            area_j = (out[j][1] - out[j][0]) * (out[j][3] - out[j][2])
            if area_j <= 0 or (ou * ov) / area_j > FIXTURE_CONTAINED_FRAC:
                dropped.add(j)
                continue
            # give way along the thinner overlap: the axis this box was least sure of
            if ou <= ov:
                if out[j][0] < out[i][0]:
                    out[j][1] = out[i][0]
                else:
                    out[j][0] = out[i][1]
            elif out[j][2] < out[i][2]:
                out[j][3] = out[i][2]
            else:
                out[j][2] = out[i][3]
    return [None if i in dropped else tuple(out[i]) for i in range(len(out))]


def snap_to_walls(box, walls):
    """A fixture within `FIXTURE_SNAP_M` of a wall line is moved flush against it.

    Shop fixtures stand against walls; a 12 cm gap between a shelving run and the wall
    behind it is a fitting error, and it is the kind a reader sees immediately because
    the daylight through the gap is what the eye follows. Only the near face moves, so
    the fixture keeps its measured depth.
    """
    u0, u1, v0, v1 = box
    for axis, perp, lo, hi, _thick in walls:
        along = (u0, u1) if axis == "u" else (v0, v1)
        if along[1] < lo - FIXTURE_SNAP_M or along[0] > hi + FIXTURE_SNAP_M:
            continue  # the wall does not run past this fixture at all
        near, far = (v0, v1) if axis == "u" else (u0, u1)
        if 0 < near - perp <= FIXTURE_SNAP_M:
            near = perp
        elif 0 < perp - far <= FIXTURE_SNAP_M:
            far = perp
        else:
            continue
        u0, u1, v0, v1 = (u0, u1, near, far) if axis == "u" else (near, far, v0, v1)
    return (u0, u1, v0, v1)
