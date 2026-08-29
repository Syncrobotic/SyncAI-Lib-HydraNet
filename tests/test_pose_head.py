"""The pose head's contract: pure-conv forward, Gaussian targets, box-window decode.

The decode round-trip is the test that matters: a synthetic peak planted at a known
pixel must come back at that pixel through render -> (pretend the network is perfect) ->
decode. If the two halves disagree about the half-cell offset convention, every
keypoint in the system is 4 px off and nothing else would notice.
"""

from __future__ import annotations

import torch

from syncai_hydranet.models.heads.pose import (
    HEATMAP_STRIDE,
    NUM_KEYPOINTS,
    PoseHeatmapLoss,
    decode_boxes,
)
from syncai_hydranet.models.hydranet import build_model

CFG = {
    "model": {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 32, "num_repeats": 1, "num_levels": 5},
        "heads": {
            "detection": {"type": "fcos", "num_classes": 4, "channels": 32, "num_convs": 1},
            "pose": {"type": "pose_p3", "channels": 32, "num_convs": 1},
        },
        "loss_balancing": "fixed",
        "fixed_weights": {"detection": 1.0, "pose": 1.0},
    },
    "data": {"input_size": [128, 160], "datasets": []},
}


def test_forward_emits_heatmaps_at_stride_8():
    model = build_model(CFG).eval()
    out = model.forward(torch.zeros(2, 3, 128, 160))
    assert out["pose"].shape == (2, NUM_KEYPOINTS, 128 // 8, 160 // 8)


def test_pose_head_is_registered_after_detection():
    model = build_model(CFG)
    names = [h.name for h in model.heads()]
    assert names.index("pose") > names.index("detection")


def test_state_dict_names_the_module_plainly():
    model = build_model(CFG)
    assert any(k.startswith("pose_heads.pose.") for k in model.state_dict())


def test_render_puts_the_peak_in_the_right_cell():
    loss = PoseHeatmapLoss()
    kps = [torch.tensor([[[84.0, 44.0, 0.9]] + [[0.0, 0.0, 0.0]] * 16])]
    target = loss.render(kps, (1, NUM_KEYPOINTS, 16, 20), "cpu")
    peak = target[0, 0].argmax()
    assert (peak // 20, peak % 20) == (44 // 8, 84 // 8)
    assert target[0, 1].sum() == 0  # zero-confidence keypoints render nothing


def test_decode_round_trips_a_planted_peak():
    loss = PoseHeatmapLoss()
    x, y = 84.0, 44.0
    kps = [torch.tensor([[[x, y, 0.9]] * NUM_KEYPOINTS])]
    target = loss.render(kps, (1, NUM_KEYPOINTS, 16, 20), "cpu")
    logits = (target[0] * 8.0) - 4.0  # a confident network's view of that target
    boxes = torch.tensor([[x - 30, y - 30, x + 30, y + 30]])
    decoded = decode_boxes(logits, boxes)
    assert decoded.shape == (1, NUM_KEYPOINTS, 3)
    err = (decoded[0, :, :2] - torch.tensor([x, y])).abs().max()
    assert err <= HEATMAP_STRIDE / 2 + 2.1, f"round-trip error {err} px"
    assert (decoded[0, :, 2] > 0.5).all()


def test_tiny_boxes_refuse_rather_than_guess():
    logits = torch.zeros(NUM_KEYPOINTS, 16, 20)
    decoded = decode_boxes(logits, torch.tensor([[10.0, 10.0, 20.0, 20.0]]))
    assert (decoded == 0).all()


def test_loss_prefers_the_right_answer():
    loss = PoseHeatmapLoss()
    kps = [torch.tensor([[[84.0, 44.0, 0.9]] * NUM_KEYPOINTS])]
    target = loss.render(kps, (1, NUM_KEYPOINTS, 16, 20), "cpu")
    right = (target * 12.0) - 6.0
    wrong = torch.roll(right, shifts=5, dims=-1)
    assert loss(right, kps) < loss(wrong, kps)


def test_predict_decodes_keypoints_per_detection():
    model = build_model(CFG).eval()
    with torch.no_grad():
        result = model.predict(torch.zeros(1, 3, 128, 160), score_thr=-10.0)
    assert "pose" in result
    n_boxes = len(result["detection"][0].get("boxes", ()))
    assert result["pose"][0].shape[0] == n_boxes
