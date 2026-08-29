"""Three ways a metric can mislead without ever being wrong.

`mIoU` averages over the classes present, so its denominator moves with the dataset --
annotating the classes that are currently missing will lower it even as the model
improves. The traversability head can contradict the terrain head, because the policy
table that ties them together is applied to the *targets* at training time and never to
the outputs at inference. And a per-class IoU is printed identically whether it was
computed over 61% of the labelled pixels or over 0.66% of them -- the third one cost this
project a sixty-epoch run before anything reported it.

None is a bug. All three are invisible unless reported, which is what these cover.

pytest tests/test_metric_honesty.py -v
"""

from types import SimpleNamespace

import numpy as np
import torch

from syncai_hydranet.engine.evaluator import (
    THIN_SUPPORT,
    ConfusionMatrix,
    _seg_metrics,
    head_disagreement,
)

# floor_hard/floor_soft -> go, stairs -> caution, wall/glass -> blocked
TRAV_MAP = {0: 255, 1: 2, 2: 2, 5: 1, 7: 0, 8: 0}


def _cm(pred, target, n=4):
    cm = ConfusionMatrix(n)
    cm.update(torch.tensor(pred), torch.tensor(target))
    return cm


# ------------------------------------------------------- mIoU denominator


def test_absent_classes_are_excluded_not_scored_zero():
    """A class in neither prediction nor truth has no IoU; scoring it 0 would be wrong."""
    miou, per_class = _cm([0, 0, 1, 1], [0, 0, 1, 1]).miou()
    assert np.isnan(per_class[2]) and np.isnan(per_class[3])
    assert miou == 1.0


def test_a_hallucinated_class_is_scored_zero_not_skipped():
    """Predicting a class that is not there must cost something."""
    _, per_class = _cm([2, 2, 1, 1], [0, 0, 1, 1]).miou()
    assert per_class[2] == 0.0, "a class predicted but absent from truth must score 0"


def test_the_denominator_moves_with_the_dataset():
    """The property that makes cross-dataset mIoU comparison invalid."""
    _, few = _cm([0, 1], [0, 1]).miou()
    _, more = _cm([0, 1, 2, 3], [0, 1, 2, 3]).miou()
    assert np.isfinite(few).sum() == 2
    assert np.isfinite(more).sum() == 4


def test_adding_a_hard_class_lowers_miou_despite_no_regression():
    """Concretely: the same predictions on the shared classes, a lower mean."""
    before, _ = _cm([0, 0, 1, 1], [0, 0, 1, 1]).miou()
    after, _ = _cm([0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 2, 2]).miou()
    assert before == 1.0
    assert after < before, "this is the trap the class count exists to expose"


# ------------------------------------------------------- head disagreement


def _t(x):
    return torch.tensor(x)


def test_no_disagreement_when_heads_are_consistent():
    terrain = _t([[1, 5, 7]])  # go, caution, blocked
    trav = _t([[2, 1, 0]])
    d, n = head_disagreement(trav, terrain, TRAV_MAP, torch.ones_like(trav, dtype=torch.bool))
    assert (d, n) == (0, 3)


def test_the_dangerous_case_is_counted():
    """Terrain says glass; traversability says walk. Nothing prevents this."""
    terrain = _t([[8]])
    trav = _t([[2]])
    d, n = head_disagreement(trav, terrain, TRAV_MAP, torch.ones_like(trav, dtype=torch.bool))
    assert (d, n) == (1, 1)


def test_ignore_pixels_are_excluded():
    terrain = _t([[8, 8]])
    trav = _t([[2, 2]])
    valid = _t([[True, False]])
    d, n = head_disagreement(trav, terrain, TRAV_MAP, valid)
    assert (d, n) == (1, 1), "the ignored pixel must not be counted either way"


def test_terrain_classes_mapping_to_ignore_are_excluded():
    """void maps to 255, so it can neither agree nor disagree."""
    terrain = _t([[0, 1]])
    trav = _t([[0, 2]])
    d, n = head_disagreement(trav, terrain, TRAV_MAP, torch.ones_like(trav, dtype=torch.bool))
    assert (d, n) == (0, 1), "only the floor_hard pixel is comparable"


# --------------------------------------------- what a per-class IoU stands on


def test_support_counts_ground_truth_pixels_not_predictions():
    """Row sums, not the IoU denominator.

    `union` includes the model's false positives, so a class it over-predicts would look
    better evidenced than the data makes it. Support has to be a property of the split
    alone, or it cannot be used to decide whether a number is worth reporting.
    """
    # class 3 is predicted everywhere and present nowhere; class 1 is the reverse.
    cm = _cm([3, 3, 3, 3], [1, 1, 1, 0])
    support = cm.support()
    assert support[1] == 3 and support[0] == 1
    assert support[3] == 0, "predicting a class does not create evidence for it"


def test_support_ignores_255():
    cm = _cm([0, 0, 1], [0, 255, 1])
    assert cm.support().sum() == 2


def test_support_of_an_empty_matrix_is_zero_not_a_crash():
    """`miou` returns NaNs for a matrix that never saw an update; support has to survive
    the same case, because the thin-class check reads it on every evaluation."""
    assert ConfusionMatrix(4).support().tolist() == [0, 0, 0, 0]


def test_a_thin_class_and_a_thick_one_report_the_same_iou():
    """The failure this exists to make visible, stated as a test.

    `column` scored 0.40-0.51 on 0.66% of the labelled pixels of the ADE20K val split --
    22 of 285 images -- and then predicted 0.00% of pixels on four store cameras. In
    metrics.jsonl that 0.51 was formatted exactly like `wall`'s, which stands on 61.86%.
    IoU alone cannot tell the two apart; only support can.
    """
    thin = _cm([0] * 999 + [1], [0] * 999 + [1])
    thick = _cm([0, 1, 1, 1], [0, 1, 1, 1])
    assert thin.miou()[1][1] == thick.miou()[1][1] == 1.0
    thin_share = thin.support()[1] / thin.support().sum()
    thick_share = thick.support()[1] / thick.support().sum()
    assert thin_share < THIN_SUPPORT < thick_share


def test_thin_classes_are_named_in_a_warning_and_their_share_recorded():
    """A number that needs a caveat must carry it in the log, not in someone's memory."""
    warned: list[str] = []
    logger = SimpleNamespace(info=lambda *_: None, warning=warned.append)
    cfg = {"data": {"terrain_classes": ["void", "floor", "column", "wall"]}}
    cm = _cm([0] * 999 + [2], [0] * 999 + [2])
    metrics = _seg_metrics({"terrain": cm}, cfg, logger)

    assert metrics["support/terrain/02_column"] < THIN_SUPPORT
    assert metrics["support/terrain/00_void"] > THIN_SUPPORT
    assert len(warned) == 1
    assert "column" in warned[0] and "not measured" in warned[0]
    assert "void" not in warned[0], "a well-evidenced class must not be caveated"


def test_a_well_evidenced_run_warns_about_nothing():
    warned: list[str] = []
    logger = SimpleNamespace(info=lambda *_: None, warning=warned.append)
    cfg = {"data": {"terrain_classes": ["void", "floor"]}}
    _seg_metrics({"terrain": _cm([0, 0, 1, 1], [0, 0, 1, 1], n=2)}, cfg, logger)
    assert warned == []


def test_void_is_not_reported_as_a_thin_measurement():
    """`void` is the ignore class, so "confirm it on site" is not advice anyone can act on.

    It reaches the per-class loop at all because a head sometimes *predicts* class 0 where
    no target has it -- union > 0, so the IoU is a finite 0.000 rather than NaN. Keeping it
    as a metric is right, a head hallucinating the ignore class is real. Warning about it
    is not: on the live batch02 run it was the only warning of the epoch, which is how a
    warning channel stops being read.
    """
    warned: list[str] = []
    logger = SimpleNamespace(info=lambda *_: None, warning=warned.append)
    cfg = {"data": {"terrain_classes": ["void", "floor", "column"]}}
    # class 0 predicted once where the target is class 1: void gets a finite IoU on ~0%.
    cm = _cm([0] + [1] * 999, [1] + [1] * 999, n=3)
    metrics = _seg_metrics({"terrain": cm}, cfg, logger)
    assert metrics["IoU/terrain/00_void"] == 0.0, "the hallucination must still be scored"
    assert metrics["support/terrain/00_void"] == 0.0
    assert warned == [], f"void must not be caveated: {warned}"


def test_a_real_class_0_is_still_checked():
    """Excluded by name, not by index. The traversability head's class 0 is `blocked` --
    a real class, and the one whose failure matters most."""
    warned: list[str] = []
    logger = SimpleNamespace(info=lambda *_: None, warning=warned.append)
    cm = _cm([0] + [2] * 999, [0] + [2] * 999, n=3)
    _seg_metrics({"traversability": cm}, {"data": {}}, logger)
    assert any("blocked" in w for w in warned), f"blocked at 0.1% must warn: {warned}"
