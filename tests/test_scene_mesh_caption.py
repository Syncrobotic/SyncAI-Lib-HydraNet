"""The commissioning render's height caption, which used to assert what the render denied.

`scene_mesh.py` draws `wall` and `column` at constants, and says why above `CLASS_NAMES`:
DA-V2 collapses on white surfaces, so a 2.4 m wall measures 1.07-1.65 m and a
floor-to-ceiling column measures 1.16 m. The geometry has always been honest about that.
**The caption was not** -- it printed those collapsed numbers as "measured p85" next to a
picture drawn at the constants, so the one line a reader takes a number from was the one
line that had no consumer checking it.

This is the checker. It is here rather than left to the eye because the failure is
invisible in the picture: a wrong caption over a right render looks like a right render.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/commissioning"))
from scene_mesh import (
    CLASS_NAMES,
    COLUMN_MIN_H,
    DRAWN_H,
    WALL_INLIER_M,
    height_caption,
    implausible,
    wall_segments,
)

WALL, COLUMN, TABLE, SHELF = 2, 3, 4, 5
COLLAPSED = {WALL: 1.54, COLUMN: 1.16, TABLE: 0.79, SHELF: 1.69}


def test_wall_and_column_are_never_reported_as_measured_heights():
    """The defect itself: `1.54m` may appear, but not on the measured side of the bar."""
    measured, _, drawn = height_caption(COLLAPSED).partition("|")
    assert "wall" not in measured and "column" not in measured
    assert "table 0.79m" in measured and "shelf 1.69m" in measured
    assert "1.54" not in measured and "1.16" not in measured
    assert "1.54" in drawn and "1.16" in drawn


def test_the_caption_states_the_heights_the_render_actually_draws():
    """A constant named in prose and a constant used in the mesh are two things until
    something reads both. `wall()` is built with `DRAWN_H["wall"]` and `column()` with
    `max(h, COLUMN_MIN_H)`, so those are the numbers the caption has to carry."""
    line = height_caption(COLLAPSED)
    assert f"{DRAWN_H['wall']:.1f} m" in line
    assert f"{COLUMN_MIN_H:.1f} m" in line


def test_the_collapse_is_named_rather_than_deleted():
    """Dropping the numbers would also be wrong. The collapse is a real property of the
    depth model on this fleet, and a reader who sees it named will not re-derive it."""
    assert "white-surface collapse" in height_caption(COLLAPSED)


def test_a_camera_whose_depth_gave_no_wall_or_column_says_nothing_about_them():
    line = height_caption({TABLE: 0.79, SHELF: 1.69})
    assert "depth saw" not in line
    assert "footprint only" in line


def test_every_drawn_constant_class_is_one_the_caption_knows_about():
    """If a fifth structure class is added to `CLASS_NAMES` it lands on the measured side
    by default, which is right for a fixture with a top surface and wrong for another
    vertical one. This is the tripwire for that day."""
    assert set(CLASS_NAMES.values()) == {"wall", "column", "display_table", "display_shelf"}


# --------------------------------------------------------------------------- plausibility

TABLE_NAME = "display_table"


def test_a_table_two_and_a_half_metres_deep_is_reported():
    """The case that prompted this: a mask bridge welds two counters into one component,
    the p3-p97 box spans both, and the render draws one enormous table. Taichung-cam01
    built a 2.60 x 2.45 m `display_table` at 0.79 m -- a plausible height on a footprint
    no table has. Drawn silently it reads as furniture and every metre taken off it
    afterwards is wrong in a way the picture does not show."""
    out = implausible([(TABLE_NAME, 2.60, 2.45, 0.79)])
    assert len(out) == 1
    assert "short 2.45" in out[0] and TABLE_NAME in out[0]


def test_an_ordinary_counter_is_not_reported():
    assert implausible([(TABLE_NAME, 2.60, 0.90, 0.79)]) == []
    assert implausible([("display_shelf", 6.30, 0.45, 2.08)]) == []
    assert implausible([("wall", 4.00, 0.15, 2.40)]) == []


def test_it_reads_the_built_dimensions_and_not_the_placed_mesh():
    """A 0.15 m wall is 0.15 m thick however the store frame is rotated.

    The first version of this check measured the world axis-aligned bound of the placed
    mesh. At Taichung-cam01's 37-degree store yaw the AABB of a 2.59 x 2.46 m counter is
    3.55 x 3.53, so every rotated fixture in the shop reported as implausible and a
    0.15 m wall came back 1.60 m thick. `implausible` takes the built dimensions for
    exactly this reason, and the argument type is the guard against it coming back.
    """
    assert implausible([("wall", 4.00, 0.15, 2.40)]) == []
    # the AABB of that same wall at 37 degrees would be ~3.3 x 2.5, which does trip
    assert implausible([("wall", 3.30, 2.50, 2.40)])


def test_a_class_with_no_interval_is_not_judged():
    """`product_ipad`, `door`, `floor`: nothing here knows what shape they should be, and
    a silent pass is honest where an invented interval would not be."""
    assert implausible([("product_ipad", 0.30, 0.30, 0.01)]) == []
    assert implausible([("door", 1.00, 0.12, 2.05)]) == []


# --------------------------------------------------------------------------- walls


def _grids(walk, wall=None, fixture=None):
    """Cell grids in `wall_segments`'s own layout: 233 x 400 at CELL = 0.06 m."""
    shape = (233, 400)
    g = {1: np.zeros(shape, bool), 2: np.zeros(shape, bool)}
    g[1][walk] = True
    if wall is not None:
        g[2][wall] = True
    if fixture is not None:
        g[4] = np.zeros(shape, bool)
        g[4][fixture] = True
    return g


def test_one_straight_wall_comes_back_as_one_segment_not_many():
    """The defect this replaced: a single wall arrived as 8-14 mask components, each drawn
    as its own pane, and the render read as slabs scattered over the floor. The components
    were never in the wrong *place* -- 15 of 17 sat within 0.37 m of the floor boundary."""
    walk = (slice(40, 120), slice(60, 200))
    wall = (slice(36, 41), slice(60, 200))  # a strip along the walkable region's top edge
    segs = wall_segments(_grids(walk, wall))
    assert len(segs) == 1
    x0, z0, x1, z1 = segs[0]
    assert np.hypot(x1 - x0, z1 - z0) > 7.0  # 140 cells at 0.06 m
    # straight to within the inlier band, not to zero: the boundary strip is a few cells
    # thick, so a line fitted inside WALL_INLIER_M may tilt by a fraction of a degree
    assert abs(z0 - z1) < 2 * WALL_INLIER_M


def test_a_counter_edge_is_not_a_wall():
    """The walkable region also ends where a fixture stands, and a fixture is not a wall.
    Without the exclusion every display table would grow a wall along its front."""
    walk = (slice(40, 120), slice(60, 200))
    fixture = (slice(36, 41), slice(60, 200))
    assert wall_segments(_grids(walk, wall=fixture, fixture=fixture)) == []


def test_it_is_seeded_so_one_camera_renders_one_room():
    """RANSAC with a fresh entropy source would give a different set of walls per run, and
    a commissioning render is a thing a human approves once and compares against later."""
    walk = (slice(40, 120), slice(60, 200))
    wall = (slice(36, 41), slice(60, 200))
    assert wall_segments(_grids(walk, wall)) == wall_segments(_grids(walk, wall))
