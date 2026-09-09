"""The serving path's L1: canvas boxes in, floor positions in metres out.

This is the first test that runs the units the serving path actually uses end to end.
`tests/test_world_frame.py` checks the producer against tracks handed to it directly;
this checks the chain that feeds it -- `CameraState.update` -> the injected tracker's
fragments -> `bytetrack.as_track` -> `CameraState.world_frame` -- because every failure
this area has produced was at a seam between two of those, not inside one of them:

* boxes arrive on the **letterboxed network canvas**, and a canvas is not describable by
  a size (`world._to_calibrated_pixels`);
* the calibrated frame is **half** the decoded stream on several commissioned cameras;
* the tracker's `Fragment` is not the producer's `Track`, and until 2026-09-09 it did not
  record the scores `WorldObject.score` exists to carry.

So the fixtures below start from **metres** and push them all the way out to a canvas box
the way the pipeline would, and the assertions are that the metres come back.

pytest tests/test_serving_world.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from _cameras import HALF_RES_CAM, HALF_RES_PLANE, HALF_RES_SIZE
from syncai_hydranet.analytics.bytetrack import OfflineForward, as_track
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import ground_to_pixel
from syncai_hydranet.preprocessing import letterbox_region
from syncai_hydranet.serving.camera import CameraState

CANVAS_HW = (640, 1120)  # the shipped engine's canvas, exports/pro6000_1120
STREAM = (1920, 1080)  # what the cameras decode at
TERRAIN_CLASSES = 4
DET_CLASSES = ["person", "bag", "device", "boxed_stock"]


def a_camera_file() -> CameraFile:
    """Taichung-cam01's real shape: intrinsics on 960x540, clips decode at 1920x1080."""
    return CameraFile(
        camera_id="Taichung-cam01",
        image_size_px=HALF_RES_SIZE,
        camera=HALF_RES_CAM,
        plane=HALF_RES_PLANE,
    )


def canvas_box_for(x_m: float, z_m: float, height_px: float = 178.0) -> np.ndarray:
    """A canvas box whose bottom-centre is where that floor point actually projects.

    metres -> calibrated px -> stream px -> canvas px, which is the path a frame takes.
    178 px is the measured median person height at network scale (docs/PLAN.md 2.2).
    """
    u, v, _ = ground_to_pixel(np.array([x_m]), np.array([z_m]), HALF_RES_CAM, HALF_RES_PLANE)
    sx, sy = float(u[0]) * 2.0, float(v[0]) * 2.0  # 960x540 intrinsics, 1920x1080 stream
    x0, y0, cw, ch = letterbox_region(*STREAM, CANVAS_HW)
    cu = sx * (cw / STREAM[0]) + x0
    cv = sy * (ch / STREAM[1]) + y0
    return np.array([cu - 30.0, cv - height_px, cu + 30.0, cv])


def a_state(**kw) -> CameraState:
    defaults = {
        "camera": "Taichung-cam01",
        "num_terrain_classes": TERRAIN_CLASSES,
        "canvas_hw": CANVAS_HW,
        "det_classes": DET_CLASSES,
        # min_hits=1 so a single frame confirms; the confirmation delay is
        # tracker.py's measurement and is not what these tests are about.
        "tracker_factory": lambda: OfflineForward(0.35, 0.20, 0.3, 0.4, 5, 1, 25.0 / 5.0),
        "camera_file": a_camera_file(),
        "source_size_px": STREAM,
    }
    defaults.update(kw)
    return CameraState(**defaults)


def feed(state: CameraState, positions, *, seq0: int = 0, dt_s: float | None = None):
    """Drive `update` once per element of `positions`, each a list of (x_m, z_m)."""
    terrain = np.zeros(CANVAS_HW, dtype=np.uint8)
    for i, frame in enumerate(positions):
        boxes = (
            np.stack([canvas_box_for(x, z) for x, z in frame]) if frame else np.zeros((0, 4))
        )
        state.update(
            seq0 + i,
            terrain,
            boxes,
            np.full(len(frame), 0.9),
            np.zeros(len(frame), dtype=np.int64),  # every box is `person`
            time_s=None if dt_s is None else i * dt_s,
        )


def live_world(state: CameraState):
    return state.world_frame([as_track(f) for f in state.tracker.tracks], name="person")


# --------------------------------------------------------------------------- the chain


def test_a_shopper_comes_back_standing_where_they_were_put():
    """The whole point of the chain: canvas pixels in, the right floor position out.

    **The tolerance is 1 cm because the correct path is exact**, and a loose one would
    make this test pass with the letterbox undone wrongly. Measured 2026-09-09 on this
    fixture: the region from `letterbox_region` returns 0.500 / 3.000 to five decimals,
    while ignoring the pad returns z 2.984 and applying the pad without the content
    scale returns 3.050. At `abs=0.05` -- the first tolerance written here -- both of
    those pass, and the test would have been checking that the chain runs rather than
    that it is right.

    It is still not the instrument for a *badly* wrong frame, and does not pretend to
    be: a 16:9 source into this canvas pads by 5 px, so the whole letterbox is worth
    centimetres here. `test_world_frame.py` holds the 2.4-3.4 m cost of naming the wrong
    frame, and `test_letterbox_geometry.py` the 1.37 m the sideways camera pays.
    """
    state = a_state()
    truth = (0.5, 3.0)
    feed(state, [[truth], [truth]])
    frame = live_world(state)

    assert frame is not None
    (obj,) = frame["objects"]
    assert obj["x_m"] == pytest.approx(truth[0], abs=0.01)
    assert obj["z_m"] == pytest.approx(truth[1], abs=0.01)
    assert obj["basis"] == "foot_point" and obj["observed"] is True
    assert frame["camera_id"] == "Taichung-cam01"
    assert frame["space"] == "camera_floor(Taichung-cam01)"


def test_the_frame_index_is_the_one_the_tracker_was_given():
    """`world_frame` takes no index, because a caller's own count would misalign the
    velocity's two observations against the times recorded for them."""
    state = a_state()
    feed(state, [[(0.5, 3.0)]] * 3)
    frame = live_world(state)
    assert frame is not None and frame["frame_index"] == state.frames_seen - 1 == 2


def test_two_shoppers_keep_their_own_positions_and_ids():
    state = a_state()
    a, b = (-1.5, 2.5), (1.2, 4.0)
    feed(state, [[a, b], [a, b]])
    frame = live_world(state)

    assert frame is not None and len(frame["objects"]) == 2
    by_x = sorted(frame["objects"], key=lambda o: o["x_m"])
    assert by_x[0]["x_m"] == pytest.approx(a[0], abs=0.05)
    assert by_x[1]["x_m"] == pytest.approx(b[0], abs=0.05)
    assert by_x[0]["track_id"] != by_x[1]["track_id"]


# ------------------------------------------------------------------- the time base


def test_no_time_base_means_no_velocity_and_never_a_zero():
    state = a_state()
    feed(state, [[(0.0, 3.0)], [(0.0, 2.5)]])
    (obj,) = live_world(state)["objects"]
    assert obj["vx_ms"] is None and obj["vz_ms"] is None


def test_a_supplied_pts_produces_the_speed_that_was_walked():
    """0.5 m of approach in 0.2 s is 2.5 m/s toward the camera, which is -z."""
    state = a_state()
    feed(state, [[(0.0, 3.0)], [(0.0, 2.5)]], dt_s=0.2)
    (obj,) = live_world(state)["objects"]
    assert obj["vz_ms"] == pytest.approx(-2.5, abs=0.1)
    assert obj["vx_ms"] == pytest.approx(0.0, abs=0.1)


def test_the_time_base_stays_bounded():
    from syncai_hydranet.serving.camera import TIME_BASE_FRAMES

    state = a_state()
    feed(state, [[(0.0, 3.0)]] * (TIME_BASE_FRAMES + 20), dt_s=0.2)
    assert len(state._times_s) == TIME_BASE_FRAMES


# ------------------------------------------------------------------- the refusals


def test_a_camera_without_a_commissioned_file_says_so_rather_than_guessing():
    """docs/PLAN.md section 2.1: a camera with no `camera.json` still serves detection and
    segmentation. It must not answer a question about metres."""
    state = a_state(camera_file=None, source_size_px=None)
    feed(state, [[(0.5, 3.0)]])
    assert state.measures_metres is False
    assert live_world(state) is None


def test_a_camera_file_without_a_source_size_is_refused_at_construction():
    """The guard downstream cannot catch this one -- canvas coordinates are only 1.17x
    outside a 960x540 calibration, under `_to_calibrated_pixels`'s 1.5x refusal."""
    with pytest.raises(ValueError, match="without source_size_px"):
        a_state(source_size_px=None)


def test_the_score_a_track_was_built_from_travels_with_its_position():
    """docs/PLAN.md step 4: the event layer has to see the detection confidence a track
    was built from -- a track of 0.15 boxes is not the claim a 0.6 track is."""
    state = a_state()
    terrain = np.zeros(CANVAS_HW, dtype=np.uint8)
    for i, score in enumerate((0.9, 0.42)):
        state.update(
            i,
            terrain,
            np.stack([canvas_box_for(0.5, 3.0)]),
            np.array([score]),
            np.zeros(1, dtype=np.int64),
        )
    (obj,) = live_world(state)["objects"]
    # What this pins is that the score travels and that it is the **last observation's**
    # (0.42, not the 0.9 the track was born on), which is the alignment `Fragment.scores`
    # gained on 2026-09-09 -- one entry per observed frame, beside `frames` and `boxes`.
    #
    # It does **not** exercise `filter_and_scale`'s rescale onto the tracker's band:
    # every class in `DEFAULT_THRESHOLDS` currently has `birth == BIRTH_REF`, so that
    # rescale is the identity today and an assertion written around it would be vacuous.
    # The day a per-class birth moves, this stays true and a rescale test becomes worth
    # writing.
    assert obj["score"] is not None
    assert obj["score"] == pytest.approx(0.42, abs=1e-6)
