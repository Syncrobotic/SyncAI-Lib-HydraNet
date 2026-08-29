"""Moving the confusion matrix onto the device must not move a single digit.

`ConfusionMatrix.update` used to copy both maps to the host and count with
`np.bincount`; it now counts with `torch.bincount` where the predictions already are.
That is a performance change and nothing else, so the burden is to show the numbers are
identical -- not close, identical -- because a metric that shifted by a rounding error
would silently invalidate every comparison against a run made before the change.

Both are integer counting, so bit-identical is the right bar rather than a tolerance.

pytest tests/test_evaluator_equivalence.py -v
"""

import numpy as np
import pytest
import torch

from syncai_hydranet.engine.evaluator import ConfusionMatrix

IGNORE = 255


def reference_matrix(preds, targets, n: int) -> np.ndarray:
    """The previous implementation, kept here as the thing to agree with."""
    mat = np.zeros((n, n), dtype=np.int64)
    for pred, target in zip(preds, targets, strict=True):
        p = pred.flatten().cpu().numpy()
        t = target.flatten().cpu().numpy()
        valid = t != IGNORE
        p, t = p[valid], t[valid]
        idx = t * n + p
        mat += np.bincount(idx, minlength=n**2).reshape(n, n)
    return mat


def reference_miou(mat: np.ndarray) -> tuple[float, np.ndarray]:
    inter = np.diag(mat).astype(np.float64)
    union = mat.sum(1) + mat.sum(0) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou)), iou


def _batches(rng, n_classes, n_batches=3, shape=(2, 24, 32), ignore_frac=0.3):
    preds, targets = [], []
    for _ in range(n_batches):
        pred = torch.from_numpy(rng.integers(0, n_classes, shape).astype(np.int64))
        tgt = rng.integers(0, n_classes, shape).astype(np.int64)
        tgt[rng.random(shape) < ignore_frac] = IGNORE
        preds.append(pred)
        targets.append(torch.from_numpy(tgt))
    return preds, targets


@pytest.mark.parametrize("n_classes", [3, 12, 13])
def test_counts_match_the_numpy_implementation_exactly(n_classes):
    rng = np.random.default_rng(0)
    preds, targets = _batches(rng, n_classes)

    cm = ConfusionMatrix(n_classes)
    for p, t in zip(preds, targets, strict=True):
        cm.update(p, t)

    # `mat` is None until the first update allocates it on that batch's device, so this
    # also pins that the loop above actually ran rather than silently doing nothing.
    assert cm.mat is not None
    got = cm.mat.reshape(n_classes, n_classes).cpu().numpy()
    assert np.array_equal(got, reference_matrix(preds, targets, n_classes))


@pytest.mark.parametrize("n_classes", [3, 12])
def test_miou_is_bit_identical(n_classes):
    rng = np.random.default_rng(7)
    preds, targets = _batches(rng, n_classes, n_batches=4)

    cm = ConfusionMatrix(n_classes)
    for p, t in zip(preds, targets, strict=True):
        cm.update(p, t)
    miou, per_class = cm.miou()

    want_miou, want_per_class = reference_miou(reference_matrix(preds, targets, n_classes))
    assert miou == want_miou
    np.testing.assert_array_equal(per_class, want_per_class)


def test_ignore_pixels_are_excluded_not_counted_as_a_class():
    """255 as a class index would land outside the matrix; as ignore it must vanish."""
    cm = ConfusionMatrix(3)
    pred = torch.zeros(4, 4, dtype=torch.long)
    tgt = torch.full((4, 4), IGNORE, dtype=torch.long)
    tgt[0, 0] = 0
    cm.update(pred, tgt)
    assert cm.mat is not None
    assert int(cm.mat.sum()) == 1


def test_an_all_ignore_batch_leaves_the_matrix_empty():
    cm = ConfusionMatrix(3)
    cm.update(torch.zeros(4, 4, dtype=torch.long), torch.full((4, 4), IGNORE, dtype=torch.long))
    assert cm.mat is not None
    assert int(cm.mat.sum()) == 0
    miou, per_class = cm.miou()
    assert np.isnan(miou) and np.isnan(per_class).all()


def test_miou_before_any_update_is_nan_rather_than_a_crash():
    """A head that no validation batch supervised must not take the run down at the
    point where it reports."""
    miou, per_class = ConfusionMatrix(12).miou()
    assert np.isnan(miou)
    assert per_class.shape == (12,)


def test_absent_classes_are_dropped_from_the_mean_not_scored_zero():
    """The documented contract: a class absent from both prediction and target has no
    IoU, and scoring it 0 would make a richer dataset look like a regression."""
    cm = ConfusionMatrix(4)
    pred = torch.zeros(8, 8, dtype=torch.long)
    tgt = torch.zeros(8, 8, dtype=torch.long)
    pred[4:] = 1
    tgt[4:] = 1
    cm.update(pred, tgt)
    miou, per_class = cm.miou()
    assert miou == 1.0
    assert np.isnan(per_class[2]) and np.isnan(per_class[3])
