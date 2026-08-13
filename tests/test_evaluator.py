"""Validation metrics: the confusion matrix, and which number picks best.pt.

pytest tests/test_evaluator.py -v
"""

from typing import ClassVar

import numpy as np
import pytest
import torch

from syncai_hydranet.engine.evaluator import ConfusionMatrix, select_metric


def _cm(pred, target, n=2):
    cm = ConfusionMatrix(n)
    cm.update(torch.tensor(pred), torch.tensor(target))
    return cm


# ----------------------------------------------------------- confusion matrix


def test_miou_matches_a_hand_computed_example():
    #                     class 0: inter 1, union 2      -> 0.5
    #                     class 1: inter 2, union 3      -> 0.667
    cm = _cm(pred=[0, 1, 1, 1], target=[0, 0, 1, 1])
    miou, per_class = cm.miou()
    assert per_class[0] == pytest.approx(0.5)
    assert per_class[1] == pytest.approx(2 / 3)
    assert miou == pytest.approx((0.5 + 2 / 3) / 2)


def test_perfect_prediction_scores_one():
    miou, per_class = _cm(pred=[0, 1, 1, 0], target=[0, 1, 1, 0]).miou()
    assert miou == pytest.approx(1.0)
    assert list(per_class) == [1.0, 1.0]


def test_ignore_index_pixels_are_excluded():
    """Letterbox padding is labelled 255 and must not count as either a hit or a miss;
    otherwise a portrait video's score is dominated by grey bars."""
    with_pad = _cm(pred=[0, 1, 1, 1], target=[0, 0, 255, 1])
    without_pad = _cm(pred=[0, 1, 1], target=[0, 0, 1])
    assert with_pad.miou()[0] == pytest.approx(without_pad.miou()[0])


def test_absent_class_is_nan_not_zero():
    """A class that appears in neither prediction nor label is unmeasured, not wrong.
    Scoring it 0 would drag mIoU down for every class the split happens not to contain."""
    _, per_class = _cm(pred=[0, 0], target=[0, 0], n=3).miou()
    assert per_class[0] == pytest.approx(1.0)
    assert np.isnan(per_class[1]) and np.isnan(per_class[2])


def test_accumulates_across_batches():
    cm = ConfusionMatrix(2)
    cm.update(torch.tensor([0, 0]), torch.tensor([0, 0]))
    cm.update(torch.tensor([1, 1]), torch.tensor([0, 0]))
    assert cm.miou()[1][0] == pytest.approx(0.5)  # 2 of 4 class-0 pixels recovered


# --------------------------------------------------------------- model choice


METRICS = {
    "traversability_mIoU": 0.42,
    "terrain_mIoU": 0.31,
    "detection_mAP": 0.19,
    "IoU/traversability/00_blocked": 0.55,
}


def test_selects_the_named_metric():
    assert select_metric(METRICS, "traversability_mIoU") == 0.42
    assert select_metric(METRICS, "IoU/traversability/00_blocked") == 0.55


def test_unknown_metric_fails_loudly_and_lists_the_options():
    """Silently falling back would mean best.pt was chosen by something other than what
    the config says -- discoverable only by rerunning the whole training."""
    with pytest.raises(KeyError) as e:
        select_metric(METRICS, "trav_miou")
    assert "traversability_mIoU" in str(e.value)


def test_selection_is_not_an_average_across_heads():
    """The old behaviour averaged mIoU and mAP. Regression guard: an improvement in the
    selected head must win even when the unrelated head gets worse."""
    better_trav = {**METRICS, "traversability_mIoU": 0.50, "detection_mAP": 0.01}
    assert select_metric(better_trav, "traversability_mIoU") > select_metric(
        METRICS, "traversability_mIoU"
    )


# ---------------------------------------------------------------- model mode


class _Recorder(torch.nn.Module):
    """Minimal stand-in: evaluate() only needs seg_heads, det_head and a forward."""

    seg_heads: ClassVar[dict] = {}
    det_head = None
    det_head_name = "detection"

    def forward(self, _x):
        return {}


def test_evaluate_restores_the_callers_model_mode():
    """Training hands in the EMA copy, which lives in eval mode. Forcing train mode on
    the way out lets a later forward pass move its BatchNorm statistics."""
    from syncai_hydranet.engine.evaluator import evaluate

    cfg = {"train": {"batch_size": 2}, "data": {"workers": 0}}
    logger = __import__("logging").getLogger("test")

    model = _Recorder().eval()
    evaluate(model, [], cfg, "cpu", logger)
    assert model.training is False

    model.train()
    evaluate(model, [], cfg, "cpu", logger)
    assert model.training is True
