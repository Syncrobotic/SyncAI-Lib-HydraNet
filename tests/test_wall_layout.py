"""Walls, as the relation that separates one from a counter -- and the two extraction
defects that produced a room full of floating panes.

The user's requirement is the frame for all of it: precision is negotiable, **relative
relationships are not**. A cabinet is not at 45 degrees and a wall is not a row of
disconnected slabs. Neither of those is a per-object property, so no per-object check can
hold them, and `PLAUSIBLE_M` -- which asks whether one box's dimensions are reasonable --
passed every single failure below.

Three things are pinned here, each written against a defect that had already happened:

* **A wall is fitted to the point set, not to a connected component.** Measured
  2026-08-28 in the store frame, Taichung-cam11's five `wall` components are 1.12 x 1.07,
  1.44 x 0.85, 1.22 x 0.27, 1.23 x 0.86 and 1.11 x 0.63 m. A real shop wall is four to
  eight metres. Those are the patches of white surface still visible *between* the
  fixtures standing in front of the wall, and a box fitted to each patch can never be
  merged back into the wall however carefully the boxes are compared -- the first attempt
  did exactly that and moved the fleet 5 -> 5, 8 -> 7, 6 -> 6.
* **Non-maximum suppression across the run.** One wall votes in several adjacent bins, so
  without it a single wall returns a stack of near-parallel lines and the count went *up*,
  to 26 and 28 per camera.
* **Floor on both sides means it is not a wall.** The relation, and the only thing that
  separates a 7.9 m wall from a 7.9 m counter run. Half the fitted runs across four
  cameras failed it, including Tao-Hsin-cam04's two longest.
"""

from __future__ import annotations

import numpy as np

from syncai_bev3d.floorplan import (
    FLOOR_BOTH_SIDES,
    WALL_GAP_TOL,
    WALL_MIN_RUN_M,
    floor_both_sides,
    resolve_overlaps,
    snap_to_walls,
    wall_runs,
)
from syncai_bev3d.shading import FADE_FLOOR, View, occlusion_alpha


def _cells(u0, u1, v0, v1, step=0.06):
    """A filled rectangle of cell centres in the store frame."""
    u = np.arange(u0, u1, step)
    v = np.arange(v0, v1, step)
    uu, vv = np.meshgrid(u, v)
    return uu.ravel(), vv.ravel()


def _join(*parts):
    return (
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
    )


def test_one_straight_wall_is_one_run():
    u, v = _cells(0.0, 5.0, 0.0, 0.15)
    runs = wall_runs(u, v)
    assert len(runs) == 1
    axis, _perp, lo, hi, _thick = runs[0]
    assert axis == "u"
    assert hi - lo > 4.5


def test_a_wall_seen_in_pieces_is_still_one_wall():
    """The defect this whole module exists for: fixtures standing in front of a wall leave
    it visible in patches, and each patch used to become its own 2.4 m slab."""
    u, v = _join(
        _cells(0.0, 1.2, 0.0, 0.15),
        _cells(1.6, 2.9, 0.0, 0.15),  # 0.4 m occlusion gap
        _cells(3.3, 5.0, 0.0, 0.15),  # another
    )
    runs = wall_runs(u, v)
    assert len(runs) == 1, f"three patches of one wall came back as {len(runs)} runs"
    assert runs[0][3] - runs[0][2] > 4.5


def test_a_doorway_is_not_bridged():
    """A gap wider than a door is an opening in the wall and has to survive. 0.9 m is a
    single door leaf; bridging it would draw a wall across the way in."""
    u, v = _join(_cells(0.0, 2.0, 0.0, 0.15), _cells(2.9, 5.0, 0.0, 0.15))
    runs = wall_runs(u, v)
    assert len(runs) == 2
    assert all(r[3] - r[2] < 2.5 for r in runs)


def test_the_gap_tolerance_sits_below_a_door_leaf():
    """Pinned because the two tests above are only meaningful either side of it."""
    assert WALL_GAP_TOL < 0.8


def test_one_wall_does_not_return_a_stack_of_parallel_lines():
    """Non-maximum suppression. A wall seen obliquely is a band of cells 30-50 cm across,
    and every bin through that band is a candidate line for the same wall."""
    u, v = _cells(0.0, 6.0, 0.0, 0.45)
    runs = wall_runs(u, v)
    assert len(runs) == 1, f"one thick wall returned {len(runs)} parallel runs"


def test_a_scattered_patch_is_not_a_wall():
    """Shorter than `WALL_MIN_RUN_M` is a patch of white surface. Every camera has
    several and each used to be drawn 2.4 m tall."""
    u, v = _join(
        _cells(0.0, 0.9, 0.0, 0.6),
        _cells(3.0, 3.8, 2.0, 2.7),
    )
    assert wall_runs(u, v) == []
    assert WALL_MIN_RUN_M >= 1.0


def test_a_corner_is_closed():
    """Two perpendicular runs that stop short of each other read as two walls that miss,
    not as a room. The scan-to-BIM sequence intersects them; so does this.

    **The horizontal arm stops 45 cm short of the vertical one, on purpose.** Two earlier
    versions of this test had the arms meeting or overlapping at the corner, and at a
    corner both arms fall in the same bin -- so each run reached the other through *shared*
    cells and the test passed with the corner-closing branch deleted. Both were found by
    deleting it. Here the two arms share no cell at all, so the intersection step is the
    only thing that can bring them together.
    """
    u, v = _join(_cells(0.0, 2.50, 0.0, 0.15), _cells(2.95, 3.10, 0.65, 4.0))
    runs = wall_runs(u, v)
    assert len(runs) == 2
    horiz = next(r for r in runs if r[0] == "u")
    vert = next(r for r in runs if r[0] == "v")
    assert horiz[3] >= 2.9, "the horizontal run stops short of the vertical one"
    assert vert[2] <= 0.10, "the vertical run stops short of the horizontal one"


def test_a_boundary_wall_has_floor_on_one_side():
    floor_u, floor_v = _cells(0.0, 5.0, 0.2, 4.0)
    assert floor_both_sides(("u", 0.0, 0.0, 5.0), floor_u, floor_v) < FLOOR_BOTH_SIDES


def test_a_counter_run_has_floor_on_both_sides():
    """The measurement this encodes: Tao-Hsin-cam04's longest `wall` was 7.9 m with 724
    floor cells one side and 550 the other. A shopper can stand on both sides of it, so
    it is a merchandise run, and drawing it 2.4 m tall put an eight-metre wall through the
    middle of a phone shop."""
    floor_u, floor_v = _join(_cells(0.0, 5.0, -2.0, -0.2), _cells(0.0, 5.0, 0.2, 2.0))
    assert floor_both_sides(("u", 0.0, 0.0, 5.0), floor_u, floor_v) > FLOOR_BOTH_SIDES


def test_the_relation_is_not_fooled_by_floor_beyond_the_run():
    """Only floor beside the run counts. Floor further along the same line is the rest of
    the room and says nothing about which side of this segment it is on."""
    floor_u, floor_v = _join(
        _cells(0.0, 5.0, 0.2, 2.0),  # one side, beside the run
        _cells(8.0, 12.0, -2.0, -0.2),  # other side, but far past the run's end
    )
    assert floor_both_sides(("u", 0.0, 0.0, 5.0), floor_u, floor_v) < FLOOR_BOTH_SIDES


# ------------------------------------------------------- fixtures, and each other
#
# The other half of "relative relationships must be correct": nothing in this pipeline
# compared one fitted fixture with another until 2026-08-28, so two boxes drawn through
# each other was a normal output. A reader calls that a jumble; no per-object check can
# see it, because each box on its own is a plausible fixture.


def test_two_fixtures_that_grew_into_each_other_are_pushed_apart():
    """The larger box holds its ground -- its extent rests on more evidence. An earlier
    version of this test had the two areas the wrong way round and asserted the wrong box
    stayed put; the code was right and the test was not."""
    a = (0.0, 4.0, 0.0, 2.0)  # 8.0 m2
    b = (3.8, 5.0, 0.0, 1.0)  # 1.2 m2, overlapping by 20 cm
    out = resolve_overlaps([a, b])
    assert out[0] == a, "the larger box should not move"
    assert out[1][0] >= a[1] - 1e-9, "the smaller box still overlaps the larger"


def test_the_same_fixture_fitted_twice_is_absorbed_not_shrunk():
    """The class mask and the re-classified wall run can both cover one counter. Shrinking
    the second to contact would leave a sliver box beside the real one; it is not a second
    fixture at all."""
    big = (0.0, 4.0, 0.0, 1.2)
    inside = (0.5, 2.0, 0.1, 1.0)
    out = resolve_overlaps([big, inside])
    assert out[0] == big
    assert out[1] is None


def test_boxes_that_do_not_touch_are_left_alone():
    a, b = (0.0, 1.0, 0.0, 1.0), (2.0, 3.0, 0.0, 1.0)
    assert resolve_overlaps([a, b]) == [a, b]


def test_a_fixture_gives_way_along_the_axis_it_overlaps_least():
    """Which axis moves is not arbitrary: the thinner overlap is the direction the box's
    own extent was least certain in, and moving the other one would delete most of it."""
    a = (0.0, 6.0, 0.0, 4.0)  # 24 m2
    b = (5.9, 8.0, 1.0, 5.0)  # 8.4 m2, overlapping 0.1 m on u and 3.0 m on v
    out = resolve_overlaps([a, b])
    assert out[1][0] == 6.0, "it should have given way on u, where it overlapped 0.1 m"
    assert out[1][2] == 1.0, "v moved, deleting three metres of a fixture to save ten cm"
    ou = min(out[0][1], out[1][1]) - max(out[0][0], out[1][0])
    ov = min(out[0][3], out[1][3]) - max(out[0][2], out[1][2])
    assert ou <= 0 or ov <= 0, "the boxes still interpenetrate"


def test_a_fixture_near_a_wall_is_moved_flush_to_it():
    wall = ("u", 0.0, 0.0, 6.0, 0.15)
    assert snap_to_walls((1.0, 3.0, 0.12, 0.72), [wall]) == (1.0, 3.0, 0.0, 0.72)


def test_a_fixture_keeps_its_depth_when_it_snaps():
    wall = ("u", 0.0, 0.0, 6.0, 0.15)
    u0, u1, _v0, v1 = snap_to_walls((1.0, 3.0, 0.12, 0.72), [wall])
    assert (u1 - u0) == 2.0
    assert v1 == 0.72, "the far face moved as well, so the fixture changed depth"


def test_a_fixture_beside_a_wall_but_past_its_end_does_not_snap():
    """The wall has to actually run past the fixture. A shelving unit two rooms along is
    not against this wall however close its coordinate happens to be."""
    wall = ("u", 0.0, 0.0, 2.0, 0.15)
    box = (8.0, 10.0, 0.12, 0.72)
    assert snap_to_walls(box, [wall]) == box


def test_a_fixture_far_from_every_wall_is_untouched():
    wall = ("u", 0.0, 0.0, 6.0, 0.15)
    box = (1.0, 3.0, 2.0, 2.6)
    assert snap_to_walls(box, [wall]) == box


# ------------------------------------------------- what stands between the eye and the room


def _slab(x0, x1, y1, z0, z1):
    """A box as (verts, faces) — only the vertices matter to `occlusion_alpha`."""
    verts = np.array(
        [[x, y, z] for x in (x0, x1) for y in (0.0, y1) for z in (z0, z1)],
        float,
    )
    return (verts, np.zeros((0, 3), int))


def _view(eye=(0.0, 2.0, -8.0), target=(0.0, 0.6, 0.0)):
    return View(list(eye), list(target), 620.0, 640.0, 360.0)


def test_a_near_wall_across_the_view_is_faded():
    """PLAN 7.22's open item: the eye is a fixed diagonal and nothing asked what stands
    along it. Taichung-cam10's 6.3 m accessory wall is on that line and the render was one
    blank panel with the shop behind it."""
    blocker = (_slab(-4.0, 4.0, 2.4, -2.0, -1.8), (200, 200, 200), 255)
    room = [(_slab(-1.0 + i, 0.0 + i, 1.0, 3.0, 4.0), (180, 180, 180), 255) for i in range(4)]
    out = occlusion_alpha(_view(), [blocker, *room])
    assert out[0] < 255, "the wall across the view kept its full alpha"
    assert all(a == 255 for a in out[1:]), "the room behind it was faded instead"


def test_a_small_object_close_to_the_eye_is_not_faded():
    """The measure is what an object hides, not how close it is. A stool by the lens hides
    nothing, and fading it would say the opposite of what the viewer sees."""
    stool = (_slab(-0.2, 0.2, 0.5, -2.0, -1.8), (200, 200, 200), 255)
    room = [(_slab(-1.0 + i, 0.0 + i, 1.0, 3.0, 4.0), (180, 180, 180), 255) for i in range(4)]
    out = occlusion_alpha(_view(), [stool, *room])
    assert out[0] == 255


def test_two_things_at_the_same_depth_do_not_hide_each_other():
    """ "Behind" means wholly behind. Two fixtures at one depth overlap on screen and hide
    nothing of each other; treating that as occlusion fades whichever is larger, which on
    a real camera is most of the room."""
    tall = (_slab(-3.0, 3.0, 3.0, -2.0, -1.8), (200, 200, 200), 255)
    short = (_slab(-1.0, 1.0, 1.0, -2.0, -1.8), (200, 200, 200), 255)
    assert occlusion_alpha(_view(), [tall, short]) == [255, 255]


def test_nothing_fades_when_nothing_is_behind_anything():
    items = [(_slab(-3.0, -2.0, 1.0, 0.0, 1.0), (200, 200, 200), 255),
             (_slab(2.0, 3.0, 1.0, 0.0, 1.0), (200, 200, 200), 255)]  # fmt: skip
    assert occlusion_alpha(_view(), items) == [255, 255]


def test_a_faded_object_is_still_visible():
    """It is being seen through, not deleted: a wall that vanishes is a different lie from
    a wall that hides the room."""
    blocker = (_slab(-6.0, 6.0, 3.0, -2.0, -1.8), (200, 200, 200), 255)
    room = [(_slab(-1.0 + i, 0.0 + i, 1.0, 3.0, 4.0), (180, 180, 180), 255) for i in range(6)]
    out = occlusion_alpha(_view(), [blocker, *room])
    assert 0 < out[0] <= round(255 * FADE_FLOOR) + 1


def test_an_item_named_in_keep_is_never_faded():
    """Exercised on something that would otherwise fade, which the floor would not.

    A first version named the floor -- a slab spanning the whole scene -- and passed with
    the `keep` check deleted, because a floor's far edge lies beyond everything, so nothing
    is ever *wholly behind* it and it cannot fade in the first place. The test was green
    about a mechanism it never reached.
    """
    blocker = (_slab(-4.0, 4.0, 2.4, -2.0, -1.8), (200, 200, 200), 255)
    room = [(_slab(-1.0 + i, 0.0 + i, 1.0, 3.0, 4.0), (180, 180, 180), 255) for i in range(4)]
    assert occlusion_alpha(_view(), [blocker, *room])[0] < 255  # it does fade
    assert occlusion_alpha(_view(), [blocker, *room], keep=((200, 200, 200),))[0] == 255
