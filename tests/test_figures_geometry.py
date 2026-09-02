"""The geometry every commissioning figure is drawn with.

These functions moved out of `tools/commissioning/demo_video.py` into
`syncai_bev3d.figures` on 2026-09-02, and the move is what exposed that nothing tested
them: `tools/` sits outside the coverage floor, so ninety statements of floor geometry had
been carrying the metres printed on two published figures without a single assertion on
them. `stature_m`'s own docstring states a property -- "round-trips exactly on synthetic
people at nine positions and three heights" -- that nothing checked.

Everything here builds its own `CameraFile`. A test that read one out of `runs/` would
pass on the machine that made it and skip in CI, which is where it is needed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from syncai_bev3d.figures import (
    PLACE_MARGIN_M,
    facing_wedge,
    in_fp_zone,
    point_in_walkable,
    stature_m,
    velocity_arrow,
    walkable_bounds,
)
from syncai_hydranet.geometry.camera_json import CameraFile, Zone
from syncai_hydranet.geometry.ground import Camera, GroundPlane


def _camera(height_m: float = 2.8, pitch_deg: float = 35.0, **kw) -> CameraFile:
    """A plausible ceiling camera: 960x540, looking down at a floor 2.8 m below."""
    return CameraFile(
        camera_id="synthetic",
        image_size_px=(960, 540),
        camera=Camera(fx=420.0, fy=420.0, cx=480.0, cy=270.0),
        plane=GroundPlane(height=height_m, pitch=math.radians(pitch_deg)),
        **kw,
    )


def _head_top_px(cf: CameraFile, x_m: float, z_m: float, stature: float) -> float:
    """Project the crown of a person standing at (x, z) and return its image row.

    The inverse of what `stature_m` does, written from the plane model directly rather
    than from that function, so the round-trip below is a check and not a tautology.
    """
    crown_level = np.array([x_m, cf.plane.height - stature, z_m])
    cam = cf.plane.rotation @ crown_level
    return float(cf.camera.cy + cf.camera.fy * cam[1] / cam[2])


@pytest.mark.parametrize("stature", [1.55, 1.70, 1.96])
@pytest.mark.parametrize("x_m,z_m", [(-2.0, 3.0), (0.0, 2.5), (0.0, 6.0), (1.8, 4.5)])
def test_stature_round_trips_from_a_projected_head_top(stature, x_m, z_m):
    """The docstring's own claim, and the one number in the panel that is a measurement."""
    cf = _camera()
    v = _head_top_px(cf, x_m, z_m, stature)
    assert stature_m(x_m, z_m, v, cf) == pytest.approx(stature, abs=1e-6)


def test_stature_is_nan_rather_than_a_number_when_the_ray_is_degenerate():
    """A caller drops an implausible sample; it cannot drop one that never said so.

    The degenerate row is the horizon of the plane's up direction, and it is constructed
    here rather than approached: a ray along it moves in the image without moving in
    height, so no stature puts a head top on it.
    """
    cf = _camera()
    b = cf.plane.rotation @ np.array([0.0, 1.0, 0.0])
    v = cf.camera.cy + cf.camera.fy * (b[1] / b[2])
    assert math.isnan(stature_m(0.0, 4.0, v, cf))


def test_walkable_bounds_come_from_the_zone_plus_a_margin():
    cf = _camera(zones=(Zone(name="w", kind="walkable", points_m=((-5.8, 0.55), (3.7, 10.3))),))
    x_lo, x_hi, z_lo, z_hi = walkable_bounds(cf)
    assert (x_lo, x_hi) == pytest.approx((-5.8 - PLACE_MARGIN_M, 3.7 + PLACE_MARGIN_M))
    # z is clamped at 0 by design and the next test says why; 0.55 - 2.0 would be behind
    # the camera.
    assert (z_lo, z_hi) == pytest.approx((0.0, 10.3 + PLACE_MARGIN_M))


def test_walkable_bounds_never_reach_behind_the_camera():
    """z is forward, so a margin applied to a zone that starts near the lens must clamp:
    a floor position behind the camera is not a place a shopper can be standing."""
    cf = _camera(
        zones=(
            Zone(
                name="w",
                kind="walkable",
                points_m=((0.0, 0.2), (2.0, 3.0)),
            ),
        )
    )
    _x_lo, _x_hi, z_lo, _z_hi = walkable_bounds(cf)
    assert z_lo == 0.0


def test_walkable_bounds_fall_back_when_the_camera_has_no_zone():
    """An uncommissioned camera gets the old hard-coded box rather than an empty one --
    stated here so that widening the fallback is a visible decision."""
    assert walkable_bounds(_camera()) == (-12.0, 12.0, 0.0, 14.0)


def test_a_zone_test_follows_the_polygon_and_not_its_bounding_box():
    """The reason `in_fp_zone` ray-casts: today's zones are grid cells, for which a bbox
    test agrees, and the zone tool will draw arbitrary shapes for which it does not."""
    ell = ((0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0), (0.0, 4.0))
    cf = _camera(false_positive_polygons_px=(ell,))
    assert in_fp_zone(cf, 0.5, 0.5)  # inside the L
    assert in_fp_zone(cf, 3.0, 0.5)  # inside the foot
    assert not in_fp_zone(cf, 3.0, 3.0)  # in the bounding box, outside the shape
    assert not in_fp_zone(cf, 5.0, 5.0)  # outside both


def test_a_camera_with_no_false_positive_zones_vetoes_nothing():
    assert not in_fp_zone(_camera(), 100.0, 100.0)


def test_point_in_walkable_uses_the_walkable_zone():
    square = ((-1.0, 1.0), (1.0, 1.0), (1.0, 3.0), (-1.0, 3.0))
    cf = _camera(zones=(Zone(name="w", kind="walkable", points_m=square),))
    assert point_in_walkable(cf, 0.0, 2.0)
    assert not point_in_walkable(cf, 0.0, 5.0)


def test_the_facing_wedge_points_forward_and_is_closed():
    """`human()` faces +z, and the wedge on the disc has to agree with it or a measured
    heading is drawn pointing at the wrong wall."""
    poly = np.asarray(facing_wedge(radius_m=0.42), float)
    assert len(poly) >= 3
    # The arc is sampled, so its furthest point sits just inside the radius rather than
    # on it: an even number of intervals never lands on the centre line.
    assert 0.40 < poly[:, 1].max() <= 0.42
    assert poly[:, 1].min() == pytest.approx(0.0, abs=1e-9), "the apex is not at the centre"
    assert abs(float(poly[:, 0].mean())) < 0.05, "the wedge is not centred on +z"


def test_the_velocity_arrow_is_as_long_as_the_travel_it_stands_for():
    """The arrow's length IS one second of travel, which is what lets it be read in
    metres off the floor grid; a scale factor here would make the panel lie quietly."""
    poly = np.asarray(velocity_arrow(1.5), float)
    span = float(poly[:, 1].max() - poly[:, 1].min())
    assert span == pytest.approx(1.5, abs=1e-6), "the arrow is not one second long"
    assert poly[:, 1].min() > 0.0, "the arrow starts clear of the figure"
