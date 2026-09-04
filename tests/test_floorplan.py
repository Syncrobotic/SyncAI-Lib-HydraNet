"""The polygons that ship in `camera.json`, and the two arithmetics that must not be merged.

`floorplan.py` was measured at **0%** in `pyproject.toml`'s coverage note on 2026-08-28
and reads 60% only because other tests import it on their way elsewhere. The parts nothing
executes are `simplify_ring` and `_douglas_peucker` -- the whole of the polygon
simplification -- and most of `FloorRaster`. Those produce the walkable outline and all 71
service zones written into the eight shipped `camera.json`, so they are the shape of every
zone event the product will ever raise.

Companion to `tests/test_plate_calibration.py`, for the same reason: the arithmetic under
the product's central claim had nothing loading it.

Three of these tests exist to hold a finding this module's own docstrings already record,
rather than to raise a percentage. Each says so where it sits.
"""

from __future__ import annotations

import numpy as np
import pytest

from syncai_bev3d.floorplan import (
    FloorRaster,
    polygon_area,
    resolve_overlaps,
    shoelace,
    simplify_chain,
    simplify_ring,
    snap_to_walls,
)


def _square(side: float = 2.0) -> np.ndarray:
    return np.array([[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]])


def _circle(n: int = 200, r: float = 3.0) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(t) * r, np.sin(t) * r], axis=-1)


# ---------------------------------------------------------------------------
# the two area functions, and why collapsing them renames zones


def test_the_two_area_functions_agree_as_numbers():
    for poly in (_square(), _square(0.4), _circle(12)):
        assert polygon_area(poly) == pytest.approx(abs(shoelace(poly)), rel=1e-12)


def test_the_two_area_functions_do_not_agree_bit_for_bit_and_that_is_why_both_exist():
    """`polygon_area` is not `abs(shoelace)`, and this test is what stops someone merging them.

    The module docstring records the consequence: both callers sort polygons by area and
    name the result by rank, `service_zones` writing `fixture_01..NN` in descending area
    order. `list.sort` is stable, so equal areas keep insertion order -- until one key
    moves in its last bit, and then two zones swap names. `camera.json` stores the name, a
    manager's rule keys on it, and every dwell reported against `fixture_06` silently
    becomes a different fixture's.

    Measured here over 4,000 random polygons: **2,400 of them differ**, at a relative
    3.6e-16. The divergence is not rare or exotic; it is the common case, and the identity
    holding to twelve digits is exactly what makes it invisible.

    If this test ever fails, the two expressions have become bit-identical and the note
    between them in `floorplan.py` should be revisited -- but check *why* before deleting
    either, because a caller silently changing its sort key is the failure it prevents.
    """
    rng = np.random.default_rng(0)
    differ = 0
    for _ in range(4000):
        poly = rng.uniform(-5.0, 5.0, (int(rng.integers(3, 9)), 2))
        if abs(shoelace(poly)) != polygon_area(poly):
            differ += 1
    assert differ > 1000, (
        f"only {differ}/4000 polygons differ; the two expressions may have converged"
    )


def test_shoelace_keeps_the_sign_that_polygon_area_throws_away():
    """The signed/unsigned split is the other reason there are two: winding is information."""
    ccw = _square()
    cw = ccw[::-1]
    assert shoelace(ccw) > 0 > shoelace(cw)
    assert polygon_area(ccw) == pytest.approx(polygon_area(cw))
    assert polygon_area(ccw) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# open-chain simplification


def test_a_chain_too_short_to_simplify_is_returned_untouched():
    """Below three points there is nothing between the endpoints to drop."""
    two = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert simplify_chain(two, 0.08) is two


def test_the_near_collinear_triangle_the_docstring_names():
    """The case `simplify_chain`'s docstring says it was verified on, executed.

    It is the evidence for the guard being `< 3` here and `< 4` at the ring caller, and
    the reason that difference is not drift: `[[0,0],[1,0.001],[2,0]]` at tol 0.08 loses
    its middle vertex, which is right for an open chain and would leave a *ring* with two
    points, zero area, and a `Zone.contains` that answers False forever and silently.
    """
    tri = np.array([[0.0, 0.0], [1.0, 0.001], [2.0, 0.0]])
    out = simplify_chain(tri, 0.08)
    assert out.tolist() == [[0.0, 0.0], [2.0, 0.0]]


def test_a_corner_further_than_the_tolerance_survives():
    """The other half of the same rule, so the test above cannot pass by dropping everything."""
    chain = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    out = simplify_chain(chain, 0.08)
    assert len(out) == 3


def test_a_degenerate_segment_falls_back_to_point_distance():
    """First and last coincide, so there is no segment to measure against.

    `_douglas_peucker` switches to the distance from the point itself when the segment
    norm underflows. Without that branch the division is by zero and every vertex in a
    closed-on-itself chain comes back NaN -- which compares False against `tol` and
    silently deletes the lot.
    """
    chain = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    out = simplify_chain(chain, 0.5)
    assert len(out) == 3, "the far middle vertex must survive a zero-length base segment"


# ---------------------------------------------------------------------------
# closed-ring simplification -- the walkable outline and every service zone


def test_a_ring_of_three_or_fewer_is_returned_untouched():
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert len(simplify_ring(tri, 0.1, max_vertices=2)) == 3


@pytest.mark.parametrize("budget", [60, 20, 8, 4])
def test_a_ring_is_tightened_until_it_fits_its_vertex_budget(budget):
    """The loop that raises `tol` by 1.5x until the ring fits, which nothing was running.

    A 200-vertex circle is the shape a contour over a raster actually produces, and the
    budget is what keeps `camera.json` from carrying one vertex per cell.
    """
    out = simplify_ring(_circle(), 0.01, max_vertices=budget)
    assert len(out) <= budget
    assert len(out) >= 3, "a ring must survive as a polygon"


def test_simplification_loses_area_and_only_ever_loses_it():
    """A budget is paid for in square metres, and the price is worth stating.

    Measured on a 3 m circle (28.27 m2): 8 vertices reads 25.46 m2 and 4 reads 18.00 m2 --
    a 36% loss at the tightest budget. Zone areas are reported to the store, so a budget
    is not a cosmetic setting, and the direction is one-way: Douglas-Peucker only ever
    removes vertices from a convex ring, so the simplified area cannot exceed the original.
    """
    circle = _circle()
    full = polygon_area(circle)
    loose = polygon_area(simplify_ring(circle, 0.01, max_vertices=60))
    tight = polygon_area(simplify_ring(circle, 0.01, max_vertices=8))
    assert full > loose > tight
    assert tight / full < 0.95, "a tight budget on a circle should cost several percent"
    assert tight / full > 0.5, "and should not collapse the shape"


def test_a_simplified_ring_keeps_its_vertices_in_order():
    """The output is `poly[sorted(idx)]`, so a ring cannot come back self-intersecting.

    A ring whose vertices are returned out of order is still a valid array and still has
    an area; it is `Zone.contains` that stops working, on a polygon that renders fine.
    """
    circle = _circle()
    out = simplify_ring(circle, 0.05, max_vertices=40)
    idx = [int(np.argmin(np.hypot(*(circle - p).T))) for p in out]
    assert idx == sorted(idx)


# ---------------------------------------------------------------------------
# the raster the contours are taken on


def test_the_two_constructors_disagree_about_z0_on_purpose():
    """The asymmetry `FloorRaster.over`'s docstring calls out, measured so it cannot drift.

    `FloorRaster(x, z)` floors `z0` at zero; `FloorRaster.over` does not, because a floor mask's
    nearest point can sit behind the camera origin on a wide mount and clamping silently
    drops that row of cells. On the same input the two differ by **1.5 m** here, which is
    six cells at the 0.25 m commissioning grid -- not a rounding difference, a band of
    floor that either exists or does not.
    """
    x = np.array([-2.0, 3.0])
    z = np.array([-0.5, 6.0])
    clamped = FloorRaster(x, z, cell=0.25, pad=1.0)
    unclamped = FloorRaster.over(np.stack([x, z], axis=-1), cell=0.25, pad=1.0)
    assert clamped.z0 == 0.0
    assert unclamped.z0 == pytest.approx(-1.5)
    assert clamped.cell == unclamped.cell == 0.25


def test_points_outside_the_extent_are_dropped_rather_than_wrapped():
    """An out-of-extent point must not land on a cell; a negative index would wrap silently."""
    grid = FloorRaster(np.array([0.0, 2.0]), np.array([0.0, 2.0]), cell=1.0, pad=0.0)
    raster = grid.raster(np.array([0.5, 99.0, -99.0]), np.array([0.5, 99.0, -99.0]))
    assert raster.shape == (grid.nz, grid.nx)
    assert raster.sum() == 1


def test_metres_come_back_at_cell_centres():
    """`to_metres` is the inverse of `raster`'s binning, offset by half a cell.

    The half-cell is not decoration: contourpy indexes cell *centres* at integers, so
    dropping it shifts every polygon in `camera.json` by half a cell -- 0.125 m at the
    commissioning grid, in the same direction, on every zone at once.
    """
    grid = FloorRaster(np.array([0.0, 2.0]), np.array([0.0, 2.0]), cell=1.0, pad=0.0)
    out = grid.to_metres(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert out.tolist() == [[0.5, 0.5], [1.5, 1.5]]


def test_a_rastered_point_survives_the_round_trip_to_its_own_cell():
    """Bin a point, take the cell back to metres, and require it inside that cell."""
    grid = FloorRaster.over(np.array([[-1.0, 2.0], [4.0, 9.0]]), cell=0.25, pad=1.0)
    pt = np.array([[1.3, 5.7]])
    raster = grid.raster_points(pt)
    rows, cols = np.nonzero(raster)
    assert len(rows) == 1
    centre = grid.to_metres(np.array([[cols[0], rows[0]]], dtype=float))[0]
    assert abs(centre[0] - 1.3) <= grid.cell / 2
    assert abs(centre[1] - 5.7) <= grid.cell / 2


# ---------------------------------------------------------------------------
# two fixtures cannot occupy the same floor


def test_boxes_that_do_not_touch_are_left_alone():
    out = resolve_overlaps([(0, 1, 0, 1), (2, 3, 2, 3)])
    assert out == [(0.0, 1.0, 0.0, 1.0), (2.0, 3.0, 2.0, 3.0)]


def test_a_box_mostly_inside_another_is_absorbed_rather_than_shrunk():
    """The asymmetry the docstring argues for, as a case.

    A box more than `FIXTURE_CONTAINED_FRAC` inside another is the same fixture fitted
    twice from different evidence -- the class mask and the re-classified wall run can
    both cover it -- so shrinking it to contact would leave a sliver of a fixture that
    was never there. PLAN 7.21 measured the welded-object half of this at 263 of 503
    merges on Taichung-cam01.
    """
    out = resolve_overlaps([(0, 4, 0, 4), (1, 2, 1, 2)])
    assert out[0] == (0.0, 4.0, 0.0, 4.0)
    assert out[1] is None, "a contained box is absorbed, and the list keeps its length"


def test_the_smaller_box_gives_way_along_the_axis_it_overlaps_least():
    """Two real fixtures grown into each other: the thinner overlap is the yielding axis.

    That axis is the direction the smaller box's own extent was least certain in, which
    is why the rule picks it rather than splitting the difference.
    """
    lateral = resolve_overlaps([(0, 2, 0, 4), (1.8, 4, 0, 4)])
    assert lateral[0] == (0.0, 1.8, 0.0, 4.0), "the smaller box's u face moved to contact"
    assert lateral[1] == (1.8, 4.0, 0.0, 4.0), "the larger box is untouched"

    forward = resolve_overlaps([(0, 4, 0, 2), (0, 4, 1.8, 4)])
    assert forward[0] == (0.0, 4.0, 0.0, 1.8)
    assert forward[1] == (0.0, 4.0, 1.8, 4.0)


def test_resolving_leaves_no_pair_still_interpenetrating():
    """The property the function exists for, asserted over its own output."""
    boxes = [(0, 2, 0, 4), (1.8, 4, 0, 4), (3.5, 6, 0, 4), (10, 11, 10, 11)]
    out = [b for b in resolve_overlaps(boxes) if b is not None]
    for i, a in enumerate(out):
        for b in out[i + 1 :]:
            ou = min(a[1], b[1]) - max(a[0], b[0])
            ov = min(a[3], b[3]) - max(a[2], b[2])
            assert ou <= 1e-9 or ov <= 1e-9, f"{a} and {b} still overlap"


# ---------------------------------------------------------------------------
# fixtures stand against walls


def test_a_fixture_within_the_snap_distance_is_moved_flush():
    """A 12 cm gap between shelving and the wall behind it is a fitting error.

    It is also the kind a reader spots instantly, because daylight through the gap is
    what the eye follows -- which is why this runs at all rather than being left to a
    reviewer.
    """
    box = (1.0, 3.0, 0.12, 1.0)
    wall = ("u", 0.0, 0.0, 5.0, 0.1)
    assert snap_to_walls(box, [wall]) == (1.0, 3.0, 0.0, 1.0)


def test_snapping_moves_only_the_near_face_so_depth_is_measured_not_invented():
    box = (1.0, 3.0, 0.12, 1.0)
    out = snap_to_walls(box, [("u", 0.0, 0.0, 5.0, 0.1)])
    assert out[3] == box[3], "the far face is the fixture's measured depth and must not move"
    assert out[0] == box[0] and out[1] == box[1], "the along-wall extent is untouched"


def test_a_gap_wider_than_the_snap_distance_is_a_gap():
    """35 cm is a real space between a fixture and a wall, not a fitting error."""
    box = (1.0, 3.0, 0.35, 1.0)
    assert snap_to_walls(box, [("u", 0.0, 0.0, 5.0, 0.1)]) == box


def test_a_wall_that_does_not_run_past_the_fixture_is_not_snapped_to():
    """Perpendicular distance alone would pull a fixture onto a wall at the other end."""
    box = (1.0, 3.0, 0.12, 1.0)
    assert snap_to_walls(box, [("u", 0.0, 8.0, 9.0, 0.1)]) == box
