"""Two ways a metric can mislead without ever being wrong.

`mIoU` averages over the classes present, so its denominator moves with the dataset --
annotating the classes that are currently missing will lower it even as the model
improves. And the traversability head can contradict the terrain head, because the
policy table that ties them together is applied to the *targets* at training time and
never to the outputs at inference.

Neither is a bug. Both are invisible unless reported, which is what these cover.

pytest tests/test_metric_honesty.py -v
"""

import numpy as np
import torch

from syncai_hydranet.engine.evaluator import ConfusionMatrix, head_disagreement

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
