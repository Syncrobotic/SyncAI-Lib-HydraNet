"""The pose validation metric: PCK against the teacher, and the arrays it zips.

pose01 ran 60 epochs and logged no pose number at all, so `best.pt` was selected on
terrain and the only pose figures that survived the run were the two checkpoints on
disk. These tests cover the metric that replaces that -- and, as much, the invariant it
depends on: keypoints and boxes are parallel arrays, and a transform that drops one
must drop the other.

pytest tests/test_pose_validation.py -v
"""

import numpy as np
import pytest
import torch
from PIL import Image

from syncai_hydranet.data.transforms import Sample, _paste
from syncai_hydranet.engine.evaluator import PoseAccumulator
from syncai_hydranet.models.heads.pose import HEATMAP_STRIDE, NUM_KEYPOINTS


def _heatmaps(cell=(5, 5), h=20, w=20):
    """Logits that are confidently background except one cell, for every joint."""
    hm = torch.full((NUM_KEYPOINTS, h, w), -10.0)
    hm[:, cell[0], cell[1]] = 10.0
    return hm


def _kp_at(cell=(5, 5), conf=1.0):
    """The teacher's keypoints, placed exactly where that cell decodes to."""
    x = (cell[1] + 0.5) * HEATMAP_STRIDE
    y = (cell[0] + 0.5) * HEATMAP_STRIDE
    kp = torch.zeros((1, NUM_KEYPOINTS, 3))
    kp[0, :, 0], kp[0, :, 1], kp[0, :, 2] = x, y, conf
    return kp


def test_a_student_that_agrees_exactly_scores_pck_one():
    acc = PoseAccumulator()
    box = torch.tensor([[0.0, 0.0, 160.0, 160.0]])
    acc.update(_heatmaps()[None], [_kp_at()], [box], [(1.0, 1.0, 0.0, 0.0)])
    m = acc.metrics()
    assert m["PCK@0.2h"] == 1.0
    assert m["L2_p50"] == pytest.approx(0.0, abs=1e-4)
    assert acc.judged == NUM_KEYPOINTS


def test_a_joint_the_teacher_could_not_see_is_not_judged():
    """Below the teacher's confidence floor there is no truth to be measured against."""
    acc = PoseAccumulator(conf_min=0.3)
    box = torch.tensor([[0.0, 0.0, 160.0, 160.0]])
    acc.update(_heatmaps()[None], [_kp_at(conf=0.1)], [box], None)
    assert acc.judged == 0
    assert acc.metrics() == {}


def test_error_is_reported_in_original_pixels_not_network_pixels():
    """A 2x letterbox downscale halves every network-pixel error when mapped back."""
    box = torch.tensor([[0.0, 0.0, 160.0, 160.0]])
    off = _kp_at()
    off[0, :, 0] += 8.0  # one stride cell away, in network pixels
    full = PoseAccumulator()
    full.update(_heatmaps()[None], [off.clone()], [box], [(1.0, 1.0, 0.0, 0.0)])
    half = PoseAccumulator()
    half.update(_heatmaps()[None], [off.clone()], [box], [(0.5, 0.5, 0.0, 0.0)])
    assert full.metrics()["L2_p50"] == pytest.approx(8.0, abs=1e-3)
    assert half.metrics()["L2_p50"] == pytest.approx(16.0, abs=1e-3)


def test_pck_is_scale_free_so_the_letterbox_does_not_move_it():
    box = torch.tensor([[0.0, 0.0, 160.0, 160.0]])
    off = _kp_at()
    off[0, :, 0] += 8.0
    scores = []
    for sx in (1.0, 0.5, 0.25):
        acc = PoseAccumulator()
        acc.update(_heatmaps()[None], [off.clone()], [box], [(sx, sx, 0.0, 0.0)])
        scores.append(acc.metrics()["PCK@0.2h"])
    assert scores[0] == scores[1] == scores[2] == 1.0


def test_desynchronised_arrays_raise_rather_than_score_the_wrong_person():
    acc = PoseAccumulator()
    two_boxes = torch.tensor([[0.0, 0.0, 160.0, 160.0], [0.0, 0.0, 80.0, 80.0]])
    with pytest.raises(ValueError, match="desynchronised"):
        acc.update(_heatmaps()[None], [_kp_at()], [two_boxes], None)


def test_the_cap_stops_the_pass_and_reports_what_it_measured():
    acc = PoseAccumulator(max_persons=1)
    box = torch.tensor([[0.0, 0.0, 160.0, 160.0]])
    for _ in range(5):
        if acc.full:
            break
        acc.update(_heatmaps()[None], [_kp_at()], [box], None)
    assert acc.persons == 1
    assert acc.judged == NUM_KEYPOINTS


# ------------------------------------------------- the parallel-array invariant


def test_a_crop_that_drops_a_person_drops_their_skeleton_too():
    """Person 1 leaves the canvas; person 0's box must not inherit person 1's joints."""
    s = Sample(
        image=Image.new("RGB", (200, 200)),
        boxes=np.array([[10.0, 10.0, 60.0, 60.0], [150.0, 150.0, 190.0, 190.0]], np.float32),
        labels=np.zeros(2, np.int64),
        pose=np.stack(
            [
                np.full((17, 3), 20.0, np.float32),  # person 0, inside
                np.full((17, 3), 170.0, np.float32),  # person 1, cropped away
            ]
        ),
    )
    out = _paste(s, (100, 100), 0, 0)
    assert len(out["boxes"]) == 1
    assert len(out["pose"]) == 1
    assert out["pose"][0, 0, 0] == pytest.approx(20.0)


def test_a_dataset_without_boxes_keeps_every_skeleton():
    """The filter is guarded on the two arrays being parallel; pose-only sources are
    untouched by it."""
    s = Sample(
        image=Image.new("RGB", (200, 200)),
        pose=np.stack([np.full((17, 3), 20.0, np.float32), np.full((17, 3), 30.0, np.float32)]),
    )
    out = _paste(s, (100, 100), 0, 0)
    assert len(out["pose"]) == 2
