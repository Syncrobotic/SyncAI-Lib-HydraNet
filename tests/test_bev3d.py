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
