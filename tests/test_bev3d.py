"""The perspective floor panel: the geometry it draws with, not the pixels it produces.

Rendering tests that assert on colours are brittle and prove little. These pin the three
things that would make the picture *wrong* rather than ugly, and one of them has already
been wrong once: the vertical sign of the projection centre, which only showed after the
fit was tight enough to clip the scene off the top of the panel.

pytest tests/test_bev3d.py -v
"""

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.geometry.bev import IGNORE, BevGrid
from syncai_hydranet.geometry.bev3d import (
    CLASS_HEIGHT_M,
    VirtualCam,
    _perspective_coeffs,
    _smooth_classes,
    _smooth_reach,
    boundary_rays,
    render,
)
from syncai_hydranet.utils.visualize import TRAV_COLORS, terrain_palette

GRID = BevGrid(x_min=-2.0, x_max=2.0, z_min=0.5, z_max=4.5, cell=0.1)
PALETTE = terrain_palette(
    [
        "void",
        "floor_hard",
        "floor_soft",
        "floor_metal",
        "wet_slippery",
        "stairs",
        "threshold_ramp",
        "wall",
        "glass",
        "door",
        "obstacle_furniture",
        "person",
        "display_fixture",
    ]
)


def _floor(walkable_rows):
    """A flat map with `go` in the near rows. Row 0 is the far field."""
    rows, cols = GRID.shape
    bev = np.full((rows, cols), IGNORE, np.uint8)
    bev[rows - walkable_rows :, :] = 2
    return bev


def test_ground_recedes_up_the_panel():
    """Further away must draw higher. If this inverts, the map is mirrored in depth."""
    cam = VirtualCam()
    _, v_near, _ = cam.project(0.0, 0.0, 1.0)
    _, v_far, _ = cam.project(0.0, 0.0, 8.0)
    assert float(v_far) < float(v_near)


def test_height_above_the_floor_draws_higher_than_the_floor():
    cam = VirtualCam()
    _, v_ground, _ = cam.project(0.0, 0.0, 4.0)
    _, v_top, _ = cam.project(0.0, 2.0, 4.0)
    assert float(v_top) < float(v_ground)


def test_lateral_sign_is_not_mirrored():
    cam = VirtualCam()
    u_left, _, _ = cam.project(-1.0, 0.0, 4.0)
    u_right, _, _ = cam.project(1.0, 0.0, 4.0)
    assert float(u_left) < float(u_right)


# --- the camera, reparameterised onto `shading.View` ------------------------


def test_the_head_on_camera_still_projects_exactly_as_it_did():
    """`VirtualCam` was reparameterised onto `View`. At ``orbit_deg = 0`` it must reduce to
    the closed form it had before, or every panel committed against the old one moved."""
    cam = VirtualCam(pitch_deg=60.0, orbit_deg=0.0, focal_px=620.0, cx=17.0, cy=-4.0)
    t = np.radians(60.0)
    for x, y, z in ((0.0, 0.0, 4.0), (1.5, 1.7, 7.0), (-2.0, 0.4, 1.0)):
        px_, py_, pz_ = x, y - cam.height_m, z + cam.setback_m
        yc = py_ * np.cos(t) + pz_ * np.sin(t)
        zc = -py_ * np.sin(t) + pz_ * np.cos(t)
        u, v, depth = cam.project(x, y, z)
        assert float(u) == pytest.approx(cam.cx + cam.focal_px * px_ / zc, rel=1e-9)
        assert float(v) == pytest.approx(cam.cy - cam.focal_px * yc / zc, rel=1e-9)
        assert float(depth) == pytest.approx(zc, rel=1e-9)


def test_the_orbit_swings_the_eye_without_tilting_it_or_moving_the_aim():
    """It rotates about the floor point the camera already looks at, so raising it turns
    the scene in place instead of sliding it out of frame."""
    head_on, swung = VirtualCam(orbit_deg=0.0).view(), VirtualCam(orbit_deg=30.0).view()
    assert swung.eye[0] > 0.5, "an orbit that leaves the eye on the centreline is not one"
    assert float(swung.eye[1]) == pytest.approx(float(head_on.eye[1]))
    # same aim point on the floor: follow each eye along its own forward axis to y = 0
    aims = [float((e + f * (-e[1] / f[1]))[2]) for e, f in
            ((head_on.eye, head_on.fwd), (swung.eye, swung.fwd))]  # fmt: skip
    assert aims[0] == pytest.approx(aims[1])


def test_pitch_outside_the_open_range_is_refused():
    for bad in (0.0, 90.0, 120.0, -10.0):
        with pytest.raises(ValueError, match="pitch_deg"):
            VirtualCam(pitch_deg=bad).view()


def test_perspective_coeffs_round_trip():
    """The homography must actually map the panel corners onto the raster corners."""
    dst = [(0.0, 0.0), (10.0, 1.0), (9.0, 8.0), (1.0, 7.0)]
    src = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    a, b, c, d, e, f, g, h = _perspective_coeffs(dst, src)
    for (dx, dy), (sx, sy) in zip(dst, src, strict=True):
        w = g * dx + h * dy + 1
        assert (a * dx + b * dy + c) / w == pytest.approx(sx, abs=1e-9)
        assert (d * dx + e * dy + f) / w == pytest.approx(sy, abs=1e-9)


def test_boundary_rays_report_the_last_walkable_range():
    angles, reach = boundary_rays(_floor(20), GRID, n_rays=32)
    assert len(angles) == len(reach) == 32
    assert reach.max() > 0
    # a map with no floor asserts nothing about any bearing
    empty = np.full(GRID.shape, IGNORE, np.uint8)
    assert boundary_rays(empty, GRID, n_rays=32)[1].max() == 0


def test_render_returns_the_requested_size_and_draws_something():
    panel = render(
        _floor(20), None, GRID, [], (320, 320), trav_colors=TRAV_COLORS, bg=(0, 0, 0)
    )
    assert isinstance(panel, Image.Image)
    assert panel.size == (320, 320)
    assert np.asarray(panel).any(), "an all-background panel means nothing was projected"


def test_the_scene_stays_inside_the_panel():
    """The regression this file exists for: a sign error centred the scene off-frame."""
    panel = np.asarray(
        render(_floor(30), None, GRID, [], (256, 256), trav_colors=TRAV_COLORS, bg=(0, 0, 0))
    )
    drawn = panel.any(-1)
    assert drawn[0].sum() == 0, "content touching the top edge means it is being clipped"
    rows = np.flatnonzero(drawn.any(1))
    assert rows.size and rows.max() < panel.shape[0] - 1


def test_terrain_colours_the_floor_when_it_is_given_one():
    terrain = np.full(GRID.shape, IGNORE, np.uint8)
    terrain[-20:, :] = 1  # floor_hard
    with_terrain = np.asarray(
        render(
            _floor(20),
            terrain,
            GRID,
            [],
            (256, 256),
            trav_colors=TRAV_COLORS,
            terrain_colors=PALETTE,
            bg=(0, 0, 0),
        )
    )
    without = np.asarray(
        render(_floor(20), None, GRID, [], (256, 256), trav_colors=TRAV_COLORS, bg=(0, 0, 0))
    )
    assert not np.array_equal(with_terrain, without)


def test_an_object_is_drawn_where_it_stands():
    obj = {
        "x_m": 0.0,
        "z_m": 2.0,
        "range_m": 2.0,
        "name": "person",
        "width_m": 0.5,
        "height_m": 1.7,
    }
    empty = np.asarray(
        render(_floor(20), None, GRID, [], (256, 256), trav_colors=TRAV_COLORS, bg=(0, 0, 0))
    )
    with_obj = np.asarray(
        render(_floor(20), None, GRID, [obj], (256, 256), trav_colors=TRAV_COLORS, bg=(0, 0, 0))
    )
    assert not np.array_equal(empty, with_obj)
    # one outside the mapped window must not be drawn at all
    far = dict(obj, z_m=GRID.z_max + 5.0)
    assert np.array_equal(
        empty,
        np.asarray(
            render(
                _floor(20), None, GRID, [far], (256, 256), trav_colors=TRAV_COLORS, bg=(0, 0, 0)
            )
        ),
    )


# --- smoothing the boundary: a drawing decision, with a limit --------------


def test_smoothing_never_fills_a_gap():
    """The one artefact here a viewer would read as real geometry. ``reach == 0`` means "no
    floor on this bearing", not "floor at zero metres", and averaging one in drags the wall
    on either side of a doorway across the doorway."""
    reach = np.array([3.0, 3.0, 3.0, 0.0, 0.0, 3.0, 3.0, 3.0])
    out = _smooth_reach(reach)
    assert (out[reach == 0] == 0).all()
    assert (out[reach > 0] > 0).all()


def test_smoothing_removes_a_single_ray_spike():
    """One stray cell can vote a bearing metres past its neighbours. That lands on the cap
    line, which is the edge the eye follows along a wall."""
    reach = np.full(40, 3.0)
    reach[20] = 9.0
    out = _smooth_reach(reach)
    assert out[20] == pytest.approx(3.0), "a mean would have smeared the spike, not removed it"
    assert out.max() < 3.5


def test_smoothing_moves_a_clean_boundary_by_centimetres_at_most():
    """It does move the drawn boundary -- `boundary_rays` is deliberately the same reduction
    the flat map filters on, and this reintroduces a disagreement. The bound is what makes
    that acceptable for a drawing, so it is worth pinning rather than asserting away."""
    angles = np.linspace(-0.8, 0.8, 120)
    reach = 4.0 + 0.35 * np.sin(angles * 2.5)  # a smoothly curving wall
    out = _smooth_reach(reach)
    assert np.abs(out - reach).max() < 0.02


def test_smoothing_an_empty_bearing_set_is_a_no_op():
    assert (_smooth_reach(np.zeros(30)) == 0).all()


def test_class_smoothing_makes_one_wall_one_run():
    """The class is read three cells past where the floor ended, so a boundary that wanders
    by centimetres reads wall, floor, wall along a single flat surface. Each flip splits the
    run and every run under two rays is dropped, so the wall comes out as a dashed line of
    panels -- which looks like measured structure and is not."""
    classes = [7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 1, 7]
    assert set(_smooth_classes(classes)) == {7}


def test_class_smoothing_never_votes_a_gap_onto_a_bearing_or_off_one():
    classes = [7, 7, "gap", "gap", "gap", 7, 7]
    out = _smooth_classes(classes)
    assert [i for i, c in enumerate(out) if c == "gap"] == [2, 3, 4]

    # a lone bearing surrounded by gap keeps its own class: there is no one else to vote
    lone = ["gap", "gap", 9, "gap", "gap"]
    assert _smooth_classes(lone) == lone


def test_the_label_carries_the_score_because_the_shape_cannot():
    """A chair mesh at 0.31 is the same silhouette as a chair at 0.95, so without the score
    on the label the panel is more confident than the run behind it."""
    from syncai_hydranet.geometry.bev3d import _draw_annotations

    obj = {"x_m": 0.0, "z_m": 2.0, "range_m": 2.0, "name": "chair", "width_m": 0.5,
           "height_m": 0.9, "score": 0.31}  # fmt: skip
    with_score = np.asarray(
        render(_floor(20), None, GRID, [obj], (256, 256), trav_colors=TRAV_COLORS, bg=(0, 0, 0))
    )
    without = np.asarray(
        render(
            _floor(20),
            None,
            GRID,
            [{k: v for k, v in obj.items() if k != "score"}],
            (256, 256),
            trav_colors=TRAV_COLORS,
            bg=(0, 0, 0),
        )
    )
    assert not np.array_equal(with_score, without), "the score never reached the label"
    assert callable(_draw_annotations)


def test_flat_terrain_classes_still_get_a_visible_boundary():
    """Something ended the floor even where the class list says the class is flat."""
    assert CLASS_HEIGHT_M[1] == 0.0  # floor_hard
    terrain = np.full(GRID.shape, 1, np.uint8)
    panel = render(
        _floor(20),
        terrain,
        GRID,
        [],
        (256, 256),
        trav_colors=TRAV_COLORS,
        terrain_colors=PALETTE,
        bg=(0, 0, 0),
    )
    assert np.asarray(panel).any()
