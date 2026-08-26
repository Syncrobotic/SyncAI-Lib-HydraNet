"""One track's route across the floor, built out of the real vector space.

`world.py` shipped a payload with "one producer, no consumer yet". This is the consumer,
so these tests run the whole chain rather than a stub of it: floor metres -> a box whose
bottom-centre projects there -> `world_frame` through the real `camera.json` geometry ->
`journeys`. A test that hand-built `WorldFrame` dicts would pass while the producer and
the consumer disagreed, which is the failure `test_stage_contract.py` records.

pytest tests/test_journey.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from syncai_hydranet.analytics.events._types import Zone
from syncai_hydranet.analytics.journey import journeys
from syncai_hydranet.analytics.tracker import Track
from syncai_hydranet.analytics.world import world_frame
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, GroundPlane, ground_to_pixel

CAM = Camera(fx=1490.0, fy=1490.0, cx=960.0, cy=540.0)
PLANE = GroundPlane(height=2.38, pitch=math.radians(50.2))
FPS = 5.0


def a_camera_file(camera_id="Tao-Hsin-cam03") -> CameraFile:
    return CameraFile(camera_id=camera_id, image_size_px=(1920, 1080), camera=CAM, plane=PLANE)


def _square(name: str, x0: float, z0: float, side: float = 1.0, **kw) -> Zone:
    poly = np.array([[x0, z0], [x0 + side, z0], [x0 + side, z0 + side], [x0, z0 + side]])
    return Zone(name=name, polygon=poly, **kw)


# A (aisle), B (till) and C (display) are a metre square each, laid out along x at z ~ 3 m.
A = _square("A_aisle", -2.5, 2.5)
B = _square("B_till", -0.5, 2.5)
C = _square("C_display", 1.5, 2.5)
ZONES = (A, B, C)


def _box_with_foot_at(x_m: float, z_m: float, height_px: float = 178.0) -> np.ndarray:
    u, v, _ = ground_to_pixel(np.array([x_m]), np.array([z_m]), CAM, PLANE)
    return np.array([u[0] - 30.0, v[0] - height_px, u[0] + 30.0, v[0]])


def _walk(positions, *, track_id=1, scores=None, camera="Tao-Hsin-cam03", times=None, start=0):
    """Feed one track along ``positions`` and return one WorldFrame per step."""
    cam_file = a_camera_file(camera)
    frames = []
    for i in range(len(positions)):
        boxes = [_box_with_foot_at(px, pz) for px, pz in positions[: i + 1]]
        track = Track(
            track_id=track_id,
            box=boxes[-1].copy(),
            hits=len(boxes),
            age=0,
            frames=list(range(start, start + i + 1)),
            boxes=boxes,
            confirmed=True,
            scores=list(scores[: i + 1]) if scores is not None else [],
        )
        f = world_frame([track], cam_file, start + i, name="person")
        if times is not None:
            f["time_s"] = times[i]
        frames.append(f)
    return frames


# ------------------------------------------------------------------ the question asked


def test_a_track_that_walks_a_to_b_to_c_reports_that_route():
    """The literal question: walked from A to B, then to C, and stayed how long."""
    path = [(-2.0, 3.0)] * 5 + [(0.0, 3.0)] * 30 + [(2.0, 3.0)] * 10
    (j,) = journeys(_walk(path), ZONES, fps=FPS)
    assert j.route == ("A_aisle", "B_till", "C_display")
    assert j.transitions == (("A_aisle", "B_till"), ("B_till", "C_display"))
    assert j.time_in("B_till") == pytest.approx(30 / FPS)
    assert j.time_in("A_aisle") == pytest.approx(5 / FPS)


def test_the_floor_distance_is_metres_walked_not_pixels():
    """-2 m to 0 m to 2 m along x is four metres, whatever the perspective did."""
    (j,) = journeys(_walk([(-2.0, 3.0), (0.0, 3.0), (2.0, 3.0)]), ZONES, fps=FPS)
    assert j.path_m == pytest.approx(4.0, abs=1e-3)


def test_a_re_entry_is_a_second_visit_and_the_total_is_their_sum():
    """Away long enough to be away -- ten frames at the display, not a boundary wobble."""
    path = [(0.0, 3.0)] * 10 + [(2.0, 3.0)] * 10 + [(0.0, 3.0)] * 10
    (j,) = journeys(_walk(path), ZONES, fps=FPS)
    tills = [v for v in j.visits if v.zone == "B_till"]
    assert len(tills) == 2, "leaving and coming back is two things, not one longer one"
    assert j.time_in("B_till") == pytest.approx(20 / FPS)
    assert j.route == ("B_till", "C_display", "B_till")


def test_a_boundary_wobble_is_not_a_departure():
    """The first real clip's failure, in a test.

    Taichung-cam01, 300 frames: a shopper standing on a 1.5 m cell edge produced eleven
    visits to the same cell -- 0.2 s, then 1.8 s, then more, each break one frame on the
    far side of the line. One stay, eleven rows, and the route unreadable.
    """
    path = [(0.0, 3.0)] * 8 + [(2.0, 3.0)] + [(0.0, 3.0)] * 8  # one frame over the line
    (j,) = journeys(_walk(path), ZONES, fps=FPS)
    tills = [v for v in j.visits if v.zone == "B_till"]
    assert len(tills) == 1, "a one-frame excursion is jitter, not leaving"
    assert j.route == ("B_till",), "and the display is not a visit either"


def test_the_hysteresis_can_be_switched_off_to_see_every_crossing():
    path = [(0.0, 3.0)] * 8 + [(2.0, 3.0)] + [(0.0, 3.0)] * 8
    (j,) = journeys(_walk(path), ZONES, fps=FPS, min_seconds=0.0)
    assert j.route == ("B_till", "C_display", "B_till")


def test_a_missed_observation_does_not_split_a_dwell():
    """The tracker missing one frame is not the shopper leaving the till."""
    cam_file = a_camera_file()
    frames = []
    for i in (0, 1, 2, 4, 5):  # frame 3 has no observation at all
        boxes = [_box_with_foot_at(0.0, 3.0)] * 3
        t = Track(1, boxes[-1], hits=3, age=0, frames=[i], boxes=boxes[:1], confirmed=True)
        frames.append(world_frame([t], cam_file, i, name="person"))
    (j,) = journeys(frames, ZONES, fps=FPS)
    tills = [v for v in j.visits if v.zone == "B_till"]
    assert len(tills) == 1
    assert (tills[0].frame_start, tills[0].frame_end) == (0, 5)
    assert (tills[0].span, tills[0].observed) == (6, 5)
    assert tills[0].seen_fraction == pytest.approx(5 / 6)


# ------------------------------------------------------------- what it refuses to do


def test_two_cameras_floors_are_not_one_route():
    a = _walk([(0.0, 3.0)], camera="Taichung-cam01")
    b = _walk([(0.0, 3.0)], camera="Kaohsiung-cam04")
    with pytest.raises(ValueError, match="unrelated axes"):
        journeys(a + b, ZONES, fps=FPS)


def test_a_position_the_geometry_refused_is_not_a_place_in_a_zone():
    """A foot point above the horizon comes back NaN; it must not land in a polygon nor
    count as a jump in the path length.

    The mount is deliberately shallow (10 deg, horizon at v ~= 277) because on the
    measured 50.2 deg store mount the horizon is off the top of the frame and no box in
    the image can produce this refusal -- the same fixture `test_world_frame.py` uses.
    """
    shallow = CameraFile(
        camera_id="shallow-cam",
        image_size_px=(1920, 1080),
        camera=CAM,
        plane=GroundPlane(height=2.38, pitch=math.radians(10.0)),
    )
    # A floor point under the shallow mount, so the measured frame lands inside B_till.
    u, v, _ = ground_to_pixel(np.array([0.0]), np.array([3.0]), CAM, shallow.plane)
    on_floor = np.array([u[0] - 30.0, v[0] - 178.0, u[0] + 30.0, v[0]])
    good = Track(1, on_floor, hits=2, age=0, frames=[0], boxes=[on_floor], confirmed=True)
    sky = np.array([900.0, 60.0, 1020.0, 200.0])
    bad = Track(1, sky, hits=2, age=0, frames=[1], boxes=[sky.copy()], confirmed=True)
    frames = [
        world_frame([good], shallow, 0, name="person"),
        world_frame([bad], shallow, 1, name="person"),
    ]
    assert frames[0]["objects"][0]["basis"] == "foot_point"
    assert frames[1]["objects"][0]["basis"] == "above_horizon"
    # min_seconds=0: this test is about the refusal, not about how long anyone stood
    (j,) = journeys(frames, ZONES, fps=FPS, min_seconds=0.0)
    assert j.path_m == pytest.approx(0.0), "a refusal is not a distance walked"
    # One measured position, inside the till, and the refusal added no second place.
    assert [v.zone for v in j.visits] == ["B_till"]
    assert [(v.frame_start, v.frame_end) for v in j.visits] == [(0, 0)]


def test_without_a_clock_a_duration_is_none_rather_than_a_nominal_rate():
    (j,) = journeys(_walk([(0.0, 3.0)] * 4), ZONES)
    assert j.seconds is None
    assert j.time_in("B_till") is None
    assert j.path_m == pytest.approx(0.0)


def test_pts_seconds_beat_a_nominal_fps_when_both_are_available():
    """The site corpus writes 30/1 into every clip regardless of its true rate."""
    times = [0.0, 0.5, 1.0, 1.5]
    (j,) = journeys(_walk([(0.0, 3.0)] * 4, times=times), ZONES, fps=30.0)
    assert j.seconds == pytest.approx(2.0)  # 1.5 s elapsed plus one 0.5 s frame


def test_a_bag_is_not_a_shopper():
    frames = _walk([(0.0, 3.0)] * 3)
    assert journeys(frames, ZONES, name="bag", fps=FPS) == []


# --------------------------------------------------------- confidence travels with it


def test_a_journey_reports_the_confidence_its_positions_were_built_from():
    low = journeys(_walk([(0.0, 3.0)] * 3, scores=[0.16, 0.18, 0.17]), ZONES, fps=FPS)
    high = journeys(_walk([(0.0, 3.0)] * 3, scores=[0.7, 0.8, 0.9]), ZONES, fps=FPS)
    assert low[0].score_p50 == pytest.approx(0.17)
    assert high[0].score_p50 == pytest.approx(0.8)


def test_a_journey_without_scores_says_none_rather_than_zero():
    (j,) = journeys(_walk([(0.0, 3.0)] * 3), ZONES, fps=FPS)
    assert j.score_p50 is None


def test_the_row_is_flat_and_json_safe():
    import json

    path = [(-2.0, 3.0)] * 6 + [(0.0, 3.0)] * 8
    (j,) = journeys(_walk(path, scores=[0.5] * 14), ZONES, fps=FPS)
    row = j.as_row()
    assert row["route"] == ["A_aisle", "B_till"]
    assert row["space"] == "camera_floor(Tao-Hsin-cam03)"
    json.dumps(row)  # would raise on numpy or NaN


def test_no_frames_is_no_journeys_rather_than_an_error():
    assert journeys([], ZONES, fps=FPS) == []
