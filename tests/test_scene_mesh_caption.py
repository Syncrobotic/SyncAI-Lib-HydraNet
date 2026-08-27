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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/commissioning"))
from scene_mesh import CLASS_NAMES, COLUMN_MIN_H, DRAWN_H, height_caption

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
