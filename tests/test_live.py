"""The live view's per-frame work, now that it is reachable without a robot.

All of this used to be inside `scripts/robot/live_view_ros.py`, where the only way to run it was
to source ROS on a robot that was already holding its camera. Nothing here needs either.

The reach classification is the part worth pinning hardest. It is the one place where the
network's answer and the sensor's answer are allowed to disagree, and the disagreement is
the product: magenta means "the model says you can walk there and the depth sensor sees
nothing at all", which is what glass and polished floor look like.

pytest tests/test_live.py -v
"""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest
import torch

from syncai_hydranet.live import (
    GO,
    REACH_BEYOND,
    REACH_COLORS,
    REACH_NO_DEPTH,
    REACH_WITHIN,
    LiveSettings,
    Recorder,
    classify_reach,
    render_frame,
)

BLOCKED = 0


# ------------------------------------------------------------------ reach


def test_walkable_and_in_range_is_within():
    trav = np.full((4, 4), GO, np.uint8)
    depth = np.full((4, 4), 2.0, np.float32)
    assert (classify_reach(trav, depth, 5.0) == REACH_WITHIN).all()


def test_walkable_but_further_than_the_range_is_its_own_class():
    """Not a failure -- just not this planner's problem yet, which is why it is a slider."""
    trav = np.full((4, 4), GO, np.uint8)
    depth = np.full((4, 4), 9.0, np.float32)
    assert (classify_reach(trav, depth, 5.0) == REACH_BEYOND).all()


def test_zero_depth_is_no_return_not_zero_distance():
    """The failure this whole panel exists to show. Treating 0 as a distance would put
    every glass panel in the nearest bucket -- the most reassuring reading available, for
    the surface most likely to be a wall."""
    trav = np.full((4, 4), GO, np.uint8)
    depth = np.zeros((4, 4), np.float32)
    assert (classify_reach(trav, depth, 5.0) == REACH_NO_DEPTH).all()


def test_pixels_the_model_did_not_call_walkable_are_left_alone():
    """Depth is only consulted where the network claimed floor: a wall at 2 m is not a
    reachable surface, and colouring it in would say it is."""
    trav = np.full((4, 4), BLOCKED, np.uint8)
    depth = np.full((4, 4), 2.0, np.float32)
    assert (classify_reach(trav, depth, 5.0) == 0).all()


def test_the_three_classes_are_split_not_merged():
    trav = np.full((3, 1), GO, np.uint8)
    depth = np.array([[2.0], [9.0], [0.0]], np.float32)
    assert list(classify_reach(trav, depth, 5.0).ravel()) == [
        REACH_WITHIN,
        REACH_BEYOND,
        REACH_NO_DEPTH,
    ]


def test_the_no_depth_colour_is_distinct_from_every_other():
    """Magenta is given a colour nothing else in the project uses, on purpose."""
    assert len({tuple(c) for c in REACH_COLORS}) == len(REACH_COLORS)


# ------------------------------------------------------------------ render


class _FakeModel:
    """Predicts a fixed split: top half walkable, bottom half blocked."""

    def __init__(self, detections=None):
        self.detections = detections or {}

    def predict(self, x, **_kw):
        h, w = x.shape[-2:]
        trav = torch.zeros(1, h, w, dtype=torch.long)
        trav[:, : h // 2] = GO
        return {
            "traversability": trav,
            "terrain": torch.zeros(1, h, w, dtype=torch.long),
            "detection": [self.detections],
        }


def _render(**kw):
    color = np.zeros((48, 64, 3), np.uint8)
    depth = kw.pop("depth", np.full((48, 64), 2.0, np.float32))
    return render_frame(
        color,
        depth,
        kw.pop("model", _FakeModel()),
        "cpu",
        size=(64, 80),
        terrain_colors=np.zeros((256, 3), np.uint8),
        settings=kw.pop("settings", LiveSettings()),
        **kw,
    )


def test_the_panel_is_two_frames_side_by_side():
    """Left is what the model saw, right is where the sensor agreed. Both, always --
    a single panel would make the disagreement invisible."""
    f = _render()
    assert f.panel.width > f.panel.height
    assert f.jpeg.startswith(b"\xff\xd8")  # a real JPEG, not an empty buffer


def test_the_scene_is_skipped_when_the_intrinsics_have_not_arrived():
    """A scene built from a guessed K is self-consistent and the wrong size, which is
    worse than not having one -- nothing downstream can tell."""
    assert _render(k=None).scene is None


def test_the_scene_is_built_once_the_intrinsics_are_known():
    k = np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])
    scene = _render(k=k).scene
    assert scene is not None
    assert scene["source"] == "depth"
    assert scene["range_m"] == LiveSettings().range_m


def test_the_stats_report_no_depth_as_a_share_of_walkable_not_of_the_frame():
    """The question is how much of what the model called floor the sensor could not see.
    A share of the whole frame would shrink as the floor does, which is backwards."""
    f = _render(depth=np.zeros((48, 64), np.float32))
    assert f.stats["go with no depth"].endswith("of go")
    assert f.stats["go with no depth"].startswith("100.0%")


def test_caller_supplied_stats_are_carried_through():
    """fps and frame age belong to the loop, not to one frame."""
    f = _render(extra_stats={"fps": "12.3"})
    assert f.stats["fps"] == "12.3"


def test_the_range_slider_moves_pixels_between_classes_without_a_retrain():
    near = _render(settings=LiveSettings(range_m=5.0), depth=np.full((48, 64), 3.0, np.float32))
    far = _render(settings=LiveSettings(range_m=1.0), depth=np.full((48, 64), 3.0, np.float32))
    assert (near.reach == REACH_WITHIN).sum() > 0
    assert (far.reach == REACH_BEYOND).sum() > 0


# ------------------------------------------------------------------ recorder


class _Info:
    """Stands in for sensor_msgs/CameraInfo."""

    distortion_model = "plumb_bob"
    d: ClassVar[list] = [0.0] * 5
    width = 640
    height = 480
    k: ClassVar[list] = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
    header = type("H", (), {"frame_id": "camera_color_optical_frame"})()


def test_calibration_can_be_written_without_opencv(tmp_path):
    """The import moved out of __init__ for this: a session recording its own intrinsics
    is writing JSON, and needing OpenCV installed to do it made the whole thing
    unreachable on any machine that runs the tests."""
    with Recorder(tmp_path, "s1", keyframe_hz=0) as rec:
        rec.write_calibration(_Info(), _Info(), stride=1)
        assert json.loads(rec.calib_path.read_text())["color"]["K"][0] == 600.0


def test_the_stride_scales_the_intrinsics_that_get_written(tmp_path):
    """What is recorded is the calibration of the frames on disk, not of the topic. Off
    by the stride is off by exactly that factor in every metre measured later, and
    nothing complains."""
    with Recorder(tmp_path, "s1", keyframe_hz=0) as rec:
        rec.write_calibration(_Info(), None, stride=2)
        k = json.loads(rec.calib_path.read_text())["color"]["K"]
        assert [k[0], k[2], k[4], k[5]] == [300.0, 160.0, 300.0, 120.0]
        assert k[8] == 1.0  # the homogeneous row is not a length and must not be scaled


def test_calibration_is_written_once_and_not_overwritten(tmp_path):
    """Intrinsics do not change during a session; a second write would only ever be a
    later, worse guess."""
    with Recorder(tmp_path, "s1", keyframe_hz=0) as rec:
        rec.write_calibration(_Info(), None, stride=1)
        rec.write_calibration(None, None, stride=8)
        assert json.loads(rec.calib_path.read_text())["subsample_stride"] == 1


def test_a_missing_camera_info_is_recorded_as_null_not_omitted(tmp_path):
    """Absent and unwritten are different facts, and the reader cannot tell them apart
    from a missing key."""
    with Recorder(tmp_path, "s1", keyframe_hz=0) as rec:
        rec.write_calibration(_Info(), None, stride=1)
        payload = json.loads(rec.calib_path.read_text())
        assert "depth_aligned_to_color" in payload
        assert payload["depth_aligned_to_color"] is None


@pytest.mark.parametrize("hz", [0, 2])
def test_keyframe_directories_exist_only_when_keyframes_are_wanted(tmp_path, hz):
    with Recorder(tmp_path, "s1", keyframe_hz=hz) as rec:
        assert rec.img_dir.is_dir() == (hz > 0)
        rec.close()
