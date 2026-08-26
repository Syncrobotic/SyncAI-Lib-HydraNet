"""The vector space, checked against the geometry that produces it rather than a stub.

`analytics/world.py` is a contract plus one producer, and `tests/test_stage_contract.py`
explains why those two have to ship together: the last contract written on its own was
found to be *wrong* when it was finally checked, not merely unadopted. So none of the
tests here assert the key list by hand -- they push real floor points through the real
`camera.json` geometry and check what comes back.

Three properties carry the design:

* a foot point round-trips to the metres it was projected from, **with the lens undone**,
  which is the one behaviour that differs from `dwell.track_ground_path` today;
* a ray that misses the floor comes back as a named refusal rather than a distance, and
  survives serialisation as `None` rather than as a JSON token nobody agrees on;
* a velocity without a time base is `None`, because the site corpus's declared frame rate
  is measured to be a lie and a speed over it is how a walk becomes a run alert.

pytest tests/test_world_frame.py -v
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from syncai_hydranet.analytics.tracker import Track
from syncai_hydranet.analytics.world import (
    BASES,
    WorldFrame,
    WorldObject,
    as_rows,
    world_frame,
    world_frames,
)
from syncai_hydranet.geometry.camera_json import CameraFile, Lens
from syncai_hydranet.geometry.ground import (
    Camera,
    GroundPlane,
    ground_to_pixel,
    pixel_to_ground,
    undistort_points,
)

# The measured store mount: height 2.38 m, pitch 50.2 deg -- the pose `hm3d_cctv` renders
# at and the one docs/PLAN.md quotes, so a metre here is the same metre as everywhere else.
CAM = Camera(fx=1490.0, fy=1490.0, cx=960.0, cy=540.0)
PLANE = GroundPlane(height=2.38, pitch=math.radians(50.2))


def a_camera_file(camera_id: str = "Tao-Hsin-cam03", lens: Lens | None = None) -> CameraFile:
    return CameraFile(
        camera_id=camera_id,
        image_size_px=(1920, 1080),
        camera=CAM,
        plane=PLANE,
        lens=lens,
    )


def _box_with_foot_at(x_m: float, z_m: float, height_px: float = 178.0) -> np.ndarray:
    """A box whose bottom-centre is the pixel that floor point projects to.

    178 px is the measured median person height at network scale (docs/PLAN.md section 2.2); the
    value does not matter to any assertion here, only that the bottom edge is the foot.
    """
    u, v, _ = ground_to_pixel(np.array([x_m]), np.array([z_m]), CAM, PLANE)
    return np.array([u[0] - 30.0, v[0] - height_px, u[0] + 30.0, v[0]])


def a_track(*positions: tuple[float, float], track_id: int = 1, age: int = 0) -> Track:
    boxes = [_box_with_foot_at(x, z) for x, z in positions]
    return Track(
        track_id=track_id,
        box=boxes[-1].copy(),
        hits=len(boxes),
        age=age,
        frames=list(range(len(boxes))),
        boxes=boxes,
        confirmed=True,
    )


def test_every_key_is_required_and_none_carries_the_refusal():
    """The `total=False` decision, pinned. See the module docstring in `world.py`: an
    optional key would make "this producer had nothing" and "this was not measured" the
    same fact, and every other refusal in this project is explicit."""
    assert WorldObject.__optional_keys__ == frozenset()
    assert WorldFrame.__optional_keys__ == frozenset()
    assert {"yaw_rad", "height_m", "vx_ms", "vz_ms"} <= WorldObject.__required_keys__


def test_a_foot_point_round_trips_to_the_metres_it_came_from():
    frame = world_frame([a_track((0.5, 3.0))], a_camera_file(), 7, name="person")
    (obj,) = frame["objects"]
    assert obj["x_m"] == pytest.approx(0.5, abs=1e-6)
    assert obj["z_m"] == pytest.approx(3.0, abs=1e-6)
    assert obj["basis"] == "foot_point"
    assert obj["basis"] in BASES
    assert frame["frame_index"] == 7


def test_the_lens_is_undone_on_the_way_to_the_floor():
    """The behaviour that differs from `dwell.track_ground_path` today, and the reason
    `camera_json.py` says the lens applies to points on their way to the floor."""
    lens = Lens(k1=-0.18, centre_px=(960.0, 540.0), radius_px=960.0)
    track = a_track((1.6, 4.0))
    frame = world_frame([track], a_camera_file(lens=lens), 0, name="person")
    (obj,) = frame["objects"]

    foot = track.foot.reshape(1, 2)
    ux, uz = pixel_to_ground(
        *undistort_points(foot, -0.18, (960.0, 540.0), 960.0).T, CAM, PLANE
    )
    assert obj["x_m"] == pytest.approx(float(ux[0]), abs=1e-9)
    assert obj["z_m"] == pytest.approx(float(uz[0]), abs=1e-9)

    _naive_x, naive_z = pixel_to_ground(foot[:, 0], foot[:, 1], CAM, PLANE)
    assert abs(obj["z_m"] - float(naive_z[0])) > 0.01, (
        "with a real k1 the corrected and uncorrected floor points must differ, or this "
        "test cannot tell whether the lens was applied at all"
    )


def test_a_ray_above_the_horizon_is_a_named_refusal_not_a_distance():
    """A shallow mount, because the measured 50.2 deg one cannot produce this case: its
    horizon sits ~1,250 px above the top of the frame, so every pixel meets the floor.
    Shallow mounts are in the fleet -- the corpus census counted 48 cameras including one
    mounted sideways -- and this is the geometry that makes a foot point unanswerable."""
    shallow = CameraFile(
        camera_id="shallow-cam",
        image_size_px=(1920, 1080),
        camera=CAM,
        plane=GroundPlane(height=2.38, pitch=math.radians(10.0)),
    )
    above = np.array([900.0, 60.0, 1020.0, 200.0])  # horizon is at v ~= 277 here
    track = Track(
        track_id=1, box=above, hits=3, frames=[0], boxes=[above.copy()], confirmed=True
    )

    frame = world_frame([track], shallow, 0, name="person")
    (obj,) = frame["objects"]
    assert obj["basis"] == "above_horizon"
    assert not math.isfinite(obj["x_m"]) and not math.isfinite(obj["z_m"])

    (row,) = as_rows(frame)
    assert row["x_m"] is None and row["z_m"] is None
    json.dumps(row)  # NaN would emit a token most parsers reject


def test_velocity_is_none_without_a_time_base():
    """Not zero, and not a number derived from the `30/1` every site clip declares."""
    track = a_track((0.0, 3.0), (0.0, 2.6))
    (obj,) = world_frame([track], a_camera_file(), 1, name="person")["objects"]
    assert obj["vx_ms"] is None and obj["vz_ms"] is None


def test_pts_seconds_give_metres_per_second_and_beat_a_declared_fps():
    track = a_track((0.0, 3.0), (0.0, 2.6))
    times = {0: 10.0, 1: 10.2}  # 0.2 s between the two observed frames
    frame = world_frame([track], a_camera_file(), 1, name="person", times_s=times, fps=30.0)
    (obj,) = frame["objects"]
    assert obj["vz_ms"] == pytest.approx(-2.0, abs=1e-3)  # 0.4 m closer over 0.2 s
    assert obj["vx_ms"] == pytest.approx(0.0, abs=1e-6)
    assert frame["time_s"] == 10.2


def test_a_declared_fps_is_used_only_when_there_are_no_pts():
    track = a_track((0.0, 3.0), (0.0, 2.6))
    (obj,) = world_frame([track], a_camera_file(), 1, name="person", fps=5.0)["objects"]
    assert obj["vz_ms"] == pytest.approx(-2.0, abs=1e-3)  # 0.4 m over 1/5 s


def test_a_coasting_track_is_reported_and_marked_unobserved():
    """It belongs in the vector space -- the shopper did not stop existing -- but a
    consumer that counts detections must be able to tell it apart from one."""
    track = a_track((0.0, 3.0), (0.0, 2.6), age=2)
    (obj,) = world_frame([track], a_camera_file(), 5, name="person")["objects"]
    assert obj["observed"] is False


def test_unconfirmed_tracks_do_not_reach_the_vector_space():
    """`Tracker.min_hits` defaults high on purpose; showing them here undoes that."""
    track = a_track((0.0, 3.0))
    track.confirmed = False
    assert world_frame([track], a_camera_file(), 0, name="person")["objects"] == []
    kept = world_frame([track], a_camera_file(), 0, name="person", confirmed_only=False)
    assert len(kept["objects"]) == 1


def test_two_cameras_name_two_different_spaces():
    """There is no store frame: nothing in `CameraFile` maps a camera's metres onto a
    store plan, so a consumer must be able to see that two `x_m` are two different x."""
    a = world_frame([a_track((0.0, 3.0))], a_camera_file("Tao-Hsin-cam03"), 0, name="person")
    b = world_frame([a_track((0.0, 3.0))], a_camera_file("Kaohsiung-cam04"), 0, name="person")
    assert a["space"] != b["space"]
    assert "Tao-Hsin-cam03" in a["space"]


def test_rows_carry_no_numpy_and_survive_json():
    track = a_track((0.5, 3.0), (0.4, 2.8))
    frame = world_frame([track], a_camera_file(), 1, name="person", times_s={0: 0.0, 1: 0.2})
    (row,) = as_rows(frame)
    assert not any(isinstance(v, np.generic) for v in row.values())
    assert json.loads(json.dumps(row))["track_id"] == 1


def test_an_empty_frame_is_a_frame_and_not_an_absence():
    frame = world_frame([], a_camera_file(), 12, name="person")
    assert frame["objects"] == [] and frame["frame_index"] == 12
    assert as_rows(frame) == []


# ------------------------------------------- the pixel frame the intrinsics were fitted on


def _half_res_camera_file() -> CameraFile:
    """Taichung-cam01's real shape: intrinsics fitted on 960x540, clips decode at 1920x1080."""
    return CameraFile(
        camera_id="Taichung-cam01",
        image_size_px=(960, 540),
        camera=Camera(fx=382.7, fy=382.7, cx=480.0, cy=270.0),
        plane=GroundPlane(height=2.49, pitch=math.radians(49.5)),
    )


def _foot_box(u: float, v: float) -> np.ndarray:
    return np.array([u - 30.0, v - 178.0, u + 30.0, v])


def test_source_pixels_against_a_smaller_calibration_are_refused_not_projected():
    """The failure found on the first real run, 2026-08-26.

    `clip_tracks.track_clip` returns boxes in the decoded stream's pixels; several
    commissioned cameras are calibrated on half that frame. Projecting anyway returned
    metres -- three shoppers several metres outside the commissioned walkable polygon,
    a 38 m walk in 60 seconds, and no NaN anywhere to say the frame was wrong.
    """
    cam_file = _half_res_camera_file()
    box = _foot_box(1400.0, 1000.0)  # a 1920x1080 pixel
    track = Track(1, box, hits=3, frames=[0], boxes=[box.copy()], confirmed=True)
    with pytest.raises(ValueError, match="calibrated on"):
        world_frame([track], cam_file, 0, name="person")


def test_stating_the_source_frame_scales_the_points_into_the_calibrated_one():
    cam_file = _half_res_camera_file()
    full = _foot_box(1400.0, 1000.0)
    half = _foot_box(700.0, 500.0)
    stated = world_frame(
        [Track(1, full, hits=3, frames=[0], boxes=[full.copy()], confirmed=True)],
        cam_file,
        0,
        name="person",
        source_size_px=(1920, 1080),
    )
    native = world_frame(
        [Track(1, half, hits=3, frames=[0], boxes=[half.copy()], confirmed=True)],
        cam_file,
        0,
        name="person",
    )
    assert stated["objects"][0]["x_m"] == pytest.approx(native["objects"][0]["x_m"], abs=1e-9)
    assert stated["objects"][0]["z_m"] == pytest.approx(native["objects"][0]["z_m"], abs=1e-9)


def test_a_box_clipped_at_the_frame_edge_still_projects():
    """The guard must not fire on the legitimate case it sits next to."""
    cam_file = _half_res_camera_file()
    edge = _foot_box(958.0, 538.0)
    frame = world_frame(
        [Track(1, edge, hits=3, frames=[0], boxes=[edge.copy()], confirmed=True)],
        cam_file,
        0,
        name="person",
    )
    assert frame["objects"][0]["basis"] == "foot_point"


# ------------------------------------------- the offline replay, `world_frames`


def a_finished_track(
    positions: dict[int, tuple[float, float]], *, track_id: int = 1, confirmed: bool = True
) -> Track:
    """A track as `Tracker.finished()` hands one back: observed frames only, in order.

    Keyed by frame index rather than built from `range`, because the property the replay
    exists to get right is what happens on the frames a track was *not* seen in, and a
    contiguous fixture cannot express one.
    """
    frames = sorted(positions)
    boxes = [_box_with_foot_at(*positions[f]) for f in frames]
    return Track(
        track_id=track_id,
        box=boxes[-1].copy(),
        hits=len(boxes),
        age=0,
        frames=frames,
        boxes=boxes,
        confirmed=confirmed,
    )


def test_the_replay_covers_every_frame_any_track_was_seen_in_and_no_others():
    """The union, in frame order -- not one track's range and not a contiguous span."""
    a = a_finished_track({0: (0.0, 3.0), 4: (0.2, 3.0)}, track_id=1)
    b = a_finished_track({2: (1.0, 4.0), 4: (1.1, 4.0), 9: (1.4, 4.0)}, track_id=2)
    frames = world_frames([a, b], a_camera_file(), name="person")
    assert [f["frame_index"] for f in frames] == [0, 2, 4, 9]
    assert [sorted(o["track_id"] for o in f["objects"]) for f in frames] == [
        [1],
        [2],
        [1, 2],
        [2],
    ]


def test_a_frame_a_track_was_not_seen_in_carries_no_row_for_it():
    """The docstring's construction, and the thing a consumer must not read as absence.

    `Track.frames` records matched frames only, so a replay has no way to place the
    shopper on the frames the tracker coasted through. Frame 2 below exists -- another
    track was seen in it -- and track 1 is simply not in it.
    """
    a = a_finished_track({0: (0.0, 3.0), 4: (0.0, 2.0)}, track_id=1)
    b = a_finished_track({0: (2.0, 5.0), 2: (2.0, 5.0), 4: (2.0, 5.0)}, track_id=2)
    (_f0, f2, _f4) = world_frames([a, b], a_camera_file(), name="person")
    assert f2["frame_index"] == 2
    assert [o["track_id"] for o in f2["objects"]] == [2]


def test_a_replay_never_reports_an_unobserved_object():
    """`world.py` states this rather than letting a consumer count `observed=False` and
    find none: coasting is invisible to a replay, and `Visit.observed` against
    `Visit.span` is the instrument that does see the gap."""
    a = a_finished_track({0: (0.0, 3.0), 3: (0.0, 2.4), 7: (0.0, 2.0)})
    frames = world_frames([a], a_camera_file(), name="person")
    assert all(o["observed"] is True for f in frames for o in f["objects"])


def test_a_velocity_is_the_one_that_was_knowable_at_that_frame():
    """Not the whole clip's. The track below walks, then stops; if the replay differenced
    the finished track the first frame would already know about the stop."""
    track = a_finished_track({0: (0.0, 3.0), 1: (0.0, 2.6), 2: (0.0, 2.55)})
    frames = world_frames([track], a_camera_file(), name="person", fps=5.0)
    speeds = [f["objects"][0]["vz_ms"] for f in frames]
    assert speeds[0] is None  # one observation is not a velocity
    assert speeds[1] == pytest.approx(-2.0, abs=1e-3)  # 0.4 m over 1/5 s
    assert speeds[2] == pytest.approx(-0.25, abs=1e-3)  # 0.05 m over 1/5 s


def test_the_last_frame_of_a_replay_is_what_the_live_producer_would_have_emitted():
    """The replay is one shape, not a second implementation: at the final frame the
    truncated track is the whole track, so the two must agree exactly."""
    track = a_finished_track({0: (0.5, 3.0), 1: (0.4, 2.8)})
    replayed = world_frames([track], a_camera_file(), name="person", fps=5.0)[-1]
    live = world_frame([track], a_camera_file(), 1, name="person", fps=5.0)
    assert replayed == live


def test_pts_seconds_reach_every_frame_of_the_replay():
    track = a_finished_track({0: (0.0, 3.0), 1: (0.0, 2.6)})
    times = {0: 10.0, 1: 10.2}
    frames = world_frames([track], a_camera_file(), name="person", times_s=times)
    assert [f["time_s"] for f in frames] == [10.0, 10.2]
    assert frames[1]["objects"][0]["vz_ms"] == pytest.approx(-2.0, abs=1e-3)


def test_the_source_frame_guard_still_fires_through_the_replay():
    """The defect `dfdcb41` was written for reaches offline callers through this function,
    and a guard that only the live shape enforced would be a guard with a way around it."""
    cam_file = _half_res_camera_file()
    box = _foot_box(1400.0, 1000.0)  # a 1920x1080 pixel against a 960x540 calibration
    track = Track(1, box, hits=3, age=0, frames=[0], boxes=[box.copy()], confirmed=True)
    with pytest.raises(ValueError, match="calibrated on"):
        world_frames([track], cam_file, name="person")


def test_no_tracks_is_no_frames_rather_than_one_empty_one():
    """A clip with nothing in it has no frames to replay -- `Track.frames` is the only
    thing that says which frame indices existed, and there are none."""
    assert world_frames([], a_camera_file(), name="person") == []
