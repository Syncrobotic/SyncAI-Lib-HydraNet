"""Per-dataset segmentation metrics, and the pooling that hid a working model.

`evaluate` accumulates one confusion matrix per head across every validation set. For a
class every val set can contain, pooling is right. For a class only one of them can
contain, it is not: the sets that cannot contain it still collect false positives, and
those land in the same denominator as the true positives from the set that can.

Measured on `runs/hydranet_retail_objects_site_balanced`, epoch 2: `product` scored
**0.583 on the store's own val frames** and **0.000 on ADE20K's**, where its ground truth
is necessarily zero and all 62,235 predicted pixels were false positives. Pooled across
285 ADE20K frames and 72 store frames it reported 0.05-0.35 and read as a model that had
not learned the class. It had.

Detection solved this before segmentation did -- `_det_metrics` gives a second dataset its
own suffixed keys rather than letting it redefine the first one's number, and says so.

pytest tests/test_per_dataset_metrics.py -v
"""

import logging

import pytest
import torch

from syncai_hydranet.engine.evaluator import ConfusionMatrix, _seg_metrics_per_dataset

CFG = {"data": {"terrain_classes": ["void", "floor", "wall", "column", "fixture", "product",
                                    "person"]}}  # fmt: skip
PRODUCT = 5
FIXTURE = 4


def _cm(pred, target, n=7):
    cm = ConfusionMatrix(n)
    cm.update(torch.tensor([pred]), torch.tensor([target]))
    return cm


def _site():
    """`product` present and mostly found: 2 of 3 product pixels right."""
    return _cm([PRODUCT, PRODUCT, PRODUCT, FIXTURE], [PRODUCT, PRODUCT, FIXTURE, FIXTURE])


def _ade():
    """No `product` ground truth at all, and the model paints one pixel of it anyway."""
    return _cm([PRODUCT, FIXTURE, FIXTURE, FIXTURE], [FIXTURE] * 4)


def _metrics(cms):
    logger = logging.getLogger("test-per-dataset")
    return _seg_metrics_per_dataset(cms, CFG, logger)


# --- the property the module exists for ------------------------------------


def test_a_class_only_one_dataset_can_contain_is_scored_on_that_dataset():
    """The whole point. Pooled, `product` looks poor; per dataset, the store's number is
    visible and the other is visibly meaningless."""
    m = _metrics({("terrain", "site_sam3"): _site(), ("terrain", "ade20k"): _ade()})
    assert m["IoU/terrain/05_product/site_sam3"] == pytest.approx(2 / 3)
    assert m["IoU/terrain/05_product/ade20k"] == 0.0


def test_the_pooled_key_is_left_alone():
    """Every existing `metrics.jsonl` and `train.primary_metric` refers to the unqualified
    key. This function must add, never redefine -- the same rule `_det_metrics` follows."""
    m = _metrics({("terrain", "site_sam3"): _site(), ("terrain", "ade20k"): _ade()})
    assert not any(k in m for k in ("IoU/terrain/05_product", "terrain_mIoU"))


def test_one_dataset_emits_nothing_extra():
    """With a single val set the pooled number already is the per-dataset one, and
    emitting both would double every key for no information."""
    assert _metrics({("terrain", "site_sam3"): _site()}) == {}


def test_each_dataset_gets_its_own_miou():
    m = _metrics({("terrain", "site_sam3"): _site(), ("terrain", "ade20k"): _ade()})
    assert m["terrain_mIoU/site_sam3"] != m["terrain_mIoU/ade20k"]


# --- the warning, which is what makes the trap findable next time ----------


def test_predicting_a_class_a_dataset_cannot_contain_is_warned_about(caplog):
    with caplog.at_level(logging.WARNING):
        _metrics({("terrain", "site_sam3"): _site(), ("terrain", "ade20k"): _ade()})
    text = caplog.text
    assert "ade20k has no ground truth for product" in text
    assert "false positives" in text
    assert "per-dataset key" in text


def test_no_warning_when_every_dataset_can_contain_the_class(caplog):
    """The warning has to be silent on the normal case or it trains people to skim."""
    with caplog.at_level(logging.WARNING):
        _metrics({("terrain", "a"): _site(), ("terrain", "b"): _site()})
    assert "no ground truth" not in caplog.text


def test_a_class_absent_from_both_the_labels_and_the_predictions_is_not_warned(caplog):
    """Absent everywhere is the ordinary case -- ADE20K contains none of most taxonomies.
    Only absent-in-truth-but-predicted is the failure."""
    quiet = _cm([FIXTURE] * 4, [FIXTURE] * 4)
    with caplog.at_level(logging.WARNING):
        _metrics({("terrain", "a"): _site(), ("terrain", "b"): quiet})
    assert "no ground truth" not in caplog.text


# --- the matrix method underneath ------------------------------------------


def test_predicted_counts_columns_where_support_counts_rows():
    """`support` answers "how much was there to find", `predicted` answers "how much did
    the model claim". They are the same number for a perfect model and the gap between
    them is what the warning reads."""
    cm = _ade()
    assert cm.support()[PRODUCT] == 0  # no ground truth
    assert cm.predicted()[PRODUCT] == 1  # painted anyway
    assert cm.support()[FIXTURE] == 4
    assert cm.predicted()[FIXTURE] == 3


def test_both_are_empty_before_any_update():
    empty = ConfusionMatrix(7)
    assert empty.predicted().sum() == 0
    assert empty.support().sum() == 0
