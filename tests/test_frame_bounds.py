"""The frame-edge refusal, and why the space its points are in has to be declared.

`FrameBounds` exists because of one alert. On 2026-09-09 the step 6 runner raised a 12.2 s
`loitering` alert on Taichung-cam01 against a shopper the frame had cut off: their box
bottom was near the frame bottom, so the foot point measured the frame rather than the
feet and projected to z = 0.08 m -- 8 cm from the camera -- landing in the nearest service
zone. No NaN, no refusal, an alert an operator would have been shown.

docs/PLAN.md section 7.13 had already measured the general form against WILDTRACK ground
truth -- edge-touching boxes are 9% of the population and carry double the error, 13.4 cm
against 6.9 cm -- and recorded it as a rule "nothing in the serving path does yet".

**The half that is easy to get wrong is not the test, it is the frame the test runs in.**
`syncai_bev3d.plate_calibration` pays for that lesson in its own docstring ("the edge gate
runs before the undistortion, and that ordering is the whole point", 269 boxes kept before
and 332 after). These tests hold the same property from the other side: the same points
judged in the wrong space produce a different, wrong answer, and the fixture below is the
fleet's real k1 rather than a number chosen to make the difference visible.

pytest tests/test_frame_bounds.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from syncai_hydranet.analytics.dwell import track_ground_path
from syncai_hydranet.analytics.tracker import Track
from syncai_hydranet.geometry.ground import (
    Camera,
    FrameBounds,
    GroundPlane,
    undistort_points,
)

# Taichung-cam01's real shape: intrinsics on 960x540, the fleet's k1, centre and radius
# as `camera.json` carries them.
W, H = 960, 540
K1 = -0.225
CENTRE = (480.0, 270.0)
RADIUS = float(np.hypot(H, W) / 2.0)

RAW = FrameBounds(W, H)
LENSED = FrameBounds(W, H, k1=K1, centre_px=CENTRE, radius_px=RADIUS)


def test_the_margin_is_the_commissioning_gate_s_own():
    """6 px at 1080 rows, scaled with the frame -- `plate_calibration.load_person_boxes`
    spells the same number as `m = 3.0` on a half-resolution plate."""
    assert RAW.margin == pytest.approx(3.0)
    assert FrameBounds(1920, 1080).margin == pytest.approx(6.0)
    assert FrameBounds(W, H, margin_px=12.0).margin == pytest.approx(12.0)


def test_raw_points_take_the_flat_test():
    feet = np.array([[480.0, 100.0], [480.0, 536.0], [480.0, 540.0], [480.0, 600.0]])
    assert list(RAW.foot_truncated(feet)) == [False, False, True, True]


def test_only_the_bottom_edge_counts():
    """A person cut off at the left still has their feet on the floor; the foot point is
    biased laterally by at most half the visible width, which is the 13.4 cm population
    section 7.13 measured. Refusing those too would discard 9% of all observations to
    avoid a centimetre-scale bias."""
    assert not RAW.foot_truncated(np.array([[0.0, 300.0], [float(W), 300.0]])).any()
    assert not RAW.foot_truncated(np.array([[480.0, 0.0]])).any()


def test_a_point_the_lens_cannot_map_from_is_refused():
    """`distort_points` returns NaN beyond the model's turning point, and a point the lens
    cannot map from was never on this frame at all."""
    assert LENSED.foot_truncated(np.array([[1e6, 1e6]])).all()


def test_a_k1_without_its_centre_is_refused_rather_than_assumed():
    with pytest.raises(ValueError, match="centre_px/radius_px"):
        FrameBounds(W, H, k1=K1).foot_truncated(np.array([[480.0, 300.0]]))


# ------------------------------------------------- the space, which is the whole point


def test_the_undistorted_image_of_the_bottom_row_is_a_curve_not_a_line():
    """Measured on this k1: the raw bottom row lands between y=555 at the centre column
    and y=618 at the corners. That 63 px spread is why a flat test cannot be applied to
    undistorted coordinates, and it is the fact the two tests below rest on."""
    cols = np.linspace(0.0, float(W), 9)
    edge = undistort_points(np.stack([cols, np.full(9, float(H))], -1), K1, CENTRE, RADIUS)
    assert edge[:, 1].min() == pytest.approx(555.4, abs=0.5)
    assert edge[:, 1].max() == pytest.approx(618.4, abs=0.5)


def test_the_raw_bottom_row_is_refused_however_it_is_expressed():
    """The same physical row, in both spaces, must come back truncated in both."""
    cols = np.linspace(0.0, float(W), 9)
    raw = np.stack([cols, np.full(9, float(H))], -1)
    assert RAW.foot_truncated(raw).all()
    assert LENSED.foot_truncated(undistort_points(raw, K1, CENTRE, RADIUS)).all()


def test_judging_undistorted_points_in_the_raw_space_refuses_real_people():
    """The failure this type exists to make impossible, in the direction that costs.

    A barrel model pushes points outward, so an undistorted foot point sits lower than its
    raw pre-image -- most in the corners, which on a downward-looking camera is the near
    foreground. Sampling the lower half of this frame: the flat test wrongly refuses
    thousands of positions that are comfortably inside the raw frame, and every one of
    them is a real shopper's feet.
    """
    cols, rows = np.meshgrid(np.arange(0.0, W, 5.0), np.arange(400.0, H, 5.0))
    raw = np.stack([cols.ravel(), rows.ravel()], -1)
    und = undistort_points(raw, K1, CENTRE, RADIUS)

    correct = LENSED.foot_truncated(und)
    wrong_space = RAW.foot_truncated(und)
    spurious = int((wrong_space & ~correct).sum())
    assert spurious > 1000, "the wrong space has stopped being wrong; check the fixture"
    # And the correct answer agrees with judging the pre-image directly, which is the
    # definition of what `k1` on this type means.
    assert (correct == RAW.foot_truncated(raw)).all()


# ------------------------------------------------------------- reaching the event layer

CAM = Camera(fx=382.7, fy=382.7, cx=480.0, cy=270.0)
PLANE = GroundPlane(height=2.49, pitch=np.deg2rad(49.5))


def a_track(*feet: tuple[float, float]) -> Track:
    boxes = [np.array([u - 15.0, v - 100.0, u + 15.0, v]) for u, v in feet]
    return Track(track_id=1, box=boxes[-1].copy(), hits=len(boxes), age=0,
                 frames=list(range(len(boxes))), boxes=boxes, confirmed=True)  # fmt: skip


def test_track_ground_path_without_bounds_is_exactly_what_it_was():
    """Off by default, because switching it on moves every dwell, path and heatmap figure
    already reported -- the re-baseline `dwell.py` states belongs beside a measurement."""
    t = a_track((480.0, 300.0), (480.0, 539.0))
    path = track_ground_path(t, CAM, PLANE)
    assert np.isfinite(path).all()


def test_track_ground_path_with_bounds_refuses_the_truncated_row_only():
    t = a_track((480.0, 300.0), (480.0, 539.0))
    path = track_ground_path(t, CAM, PLANE, RAW)
    assert np.isfinite(path[0]).all()
    assert np.isnan(path[1]).all()


def test_a_refused_row_is_outside_every_zone_by_construction():
    """`Zone.contains` already reads NaN as outside, deliberately -- so the refusal
    reaches every rule in the event layer without any of them learning a second thing."""
    from syncai_hydranet.analytics.events import Zone

    t = a_track((480.0, 539.0))
    square = Zone(
        name="near", polygon=np.array([[-9.0, -9.0], [9.0, -9.0], [9.0, 9.0], [-9.0, 9.0]])
    )
    assert square.contains(track_ground_path(t, CAM, PLANE)).all()
    assert not square.contains(track_ground_path(t, CAM, PLANE, RAW)).any()
