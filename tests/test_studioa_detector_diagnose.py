"""Gate attribution must never combine scores from different candidate points."""

import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[1] / "tools/annotation/studioa_detector_diagnose.py"
spec = importlib.util.spec_from_file_location("studioa_detector_diagnose", path)
assert spec and spec.loader
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.mark.parametrize(
    "iou,prob,ctr,retained,reason",
    [
        (0.49, [0.8, 0.1], 0.9, False, "localization"),
        (0.50, [0.2, 0.1], 1.0, False, "class_probability"),
        (0.50, [0.3, 0.7], 1.0, False, "class_competition"),
        (0.50, [0.8, 0.1], 0.25, False, "centerness_gate"),
        (0.50, [0.8, 0.1], 0.9, False, "nms_or_cap"),
        (0.50, [0.8, 0.1], 0.9, True, "covered_after_decode"),
    ],
)
def test_boundary_and_gate_attribution(iou, prob, ctr, retained, reason):
    result = worker.decompose(
        torch.tensor([iou]), torch.tensor([prob]), torch.tensor([ctr]), 0, retained
    )
    assert result["reason"] == reason


def test_joint_candidates_not_independent_maxima():
    result = worker.decompose(
        torch.tensor([0.8, 0.1, 0.7]),
        torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.05]]),
        torch.tensor([0.1, 1.0, 1.0]),
        0,
        False,
    )
    assert result["combined_pass_points"] == 0
    assert result["oracle_centerness_eligible_points"] == 1
    assert result["reason"] == "centerness_gate"
    pair = result["best_localized_product"]
    assert pair["point_index"] == 2
    assert pair["product"] == pytest.approx(pair["class_probability"] * pair["centerness"])


def test_centerness_oracle_still_requires_class_winner():
    result = worker.decompose(
        torch.tensor([0.8]), torch.tensor([[0.4, 0.9]]), torch.tensor([0.1]), 0, False
    )
    assert result["reason"] == "class_competition"
    assert result["oracle_centerness_eligible_points"] == 0
    with pytest.raises(AssertionError, match="eligible"):
        worker.decompose(
            torch.tensor([0.1]), torch.tensor([[0.9, 0.1]]), torch.tensor([1.0]), 0, True
        )
