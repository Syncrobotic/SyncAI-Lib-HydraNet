"""Unknown objects must receive no false-background gradient in partial FCOS batches."""

import pytest
import torch

from syncai_hydranet.models.heads.detection import FCOSHead
from syncai_hydranet.models.losses import FCOSLoss


def test_class_negative_budget_bounds_mass_and_excludes_existing_supervision():
    from syncai_hydranet.models.losses import class_negative_weights

    positive = torch.zeros(2, 20, 3)
    positive[0, :3, 0] = 1
    existing = positive.clone()
    existing[:, -2:, :] = 1  # existing empty negatives must not spend the added budget
    selected = torch.ones_like(positive, dtype=torch.bool)
    weight = class_negative_weights(selected, positive, existing)
    torch.testing.assert_close(weight.sum((0, 1)), torch.tensor([3.0, 1.0, 1.0]))
    assert (weight[existing > 0] == 0).all()
    assert not weight.requires_grad


def test_negative_only_area_replication_keeps_bounded_loss_and_total_gradient():
    results = []
    for area in (8, 16):
        head, cls, reg, ctr, _, _, negative = batch()
        negative.zero_()
        cm = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
        cm[:, 0, :area, :area] = 1
        loss, _ = FCOSLoss(3, class_negative_normalization="positive_budget")(
            head,
            cls,
            reg,
            ctr,
            [torch.zeros(0, 4)],
            [torch.zeros(0, dtype=torch.long)],
            negative_mask=negative,
            class_negative_mask=cm,
        )
        loss.backward()
        results.append((loss.detach(), sum(x.grad.sum() for x in cls)))
        assert all((x.grad[:, 1:] == 0).all() for x in cls)
        assert all((x.grad == 0).all() for x in reg + ctr)
    torch.testing.assert_close(results[0], results[1])


def test_budget_preserves_positive_and_empty_gradients_and_only_scales_added_negatives():
    gradients = []
    for mode in ("sum", "positive_budget"):
        head, cls, reg, ctr, boxes, labels, negative = batch()
        cm = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
        cm[:, 0] = 1
        cm[:, 1, 8:24, 8:24] = 1  # contradictory same-class signal remains protected
        cm[:, 2, 24:] = 255
        FCOSLoss(3, class_negative_normalization=mode)(
            head,
            cls,
            reg,
            ctr,
            boxes,
            labels,
            negative_mask=negative,
            class_negative_mask=cm,
        )[0].backward()
        gradients.append(([x.grad.clone() for x in cls], [x.grad.clone() for x in reg + ctr]))
    a, b = gradients
    torch.testing.assert_close(a[1], b[1])
    for x, y in zip(a[0], b[0], strict=True):
        torch.testing.assert_close(x[:, 1:], y[:, 1:])
    torch.testing.assert_close(a[0][0][:, :, 0, 0], b[0][0][:, :, 0, 0])
    assert 0 < b[0][0][0, 0, 0, 1] < a[0][0][0, 0, 0, 1]


def test_bad_class_negative_normalization_rejected():
    with pytest.raises(ValueError, match="normalization"):
        FCOSLoss(3, class_negative_normalization="invalid")


def batch():
    head = FCOSHead(8, 3, in_levels=[0, 1], channels=8, num_convs=1, strides=[8, 16])
    cls = [
        torch.zeros(1, 3, 4, 4, requires_grad=True),
        torch.zeros(1, 3, 2, 2, requires_grad=True),
    ]
    reg = [
        torch.full((1, 4, 4, 4), 8.0, requires_grad=True),
        torch.full((1, 4, 2, 2), 16.0, requires_grad=True),
    ]
    ctr = [
        torch.zeros(1, 1, 4, 4, requires_grad=True),
        torch.zeros(1, 1, 2, 2, requires_grad=True),
    ]
    boxes = [torch.tensor([[8.0, 8.0, 24.0, 24.0]])]
    labels = [torch.tensor([1])]
    negative = torch.zeros(1, 32, 32, dtype=torch.long)
    negative[:, :8, :8] = 1
    negative[:, 24:, :] = 255  # transform padding is NOT known empty
    return head, cls, reg, ctr, boxes, labels, negative


def test_partial_focal_unknown_channels_padding_and_regression_gradients():
    head, cls, reg, ctr, boxes, labels, negative = batch()
    loss, _ = FCOSLoss(3)(head, cls, reg, ctr, boxes, labels, negative_mask=negative)
    loss.backward()
    g = cls[0].grad[0]
    assert (g[:, 0, 0] > 0).all()  # explicit empty patch pushes all classes down
    assert g[1, 1, 1] < 0  # assigned label pushed up
    assert g[0, 1, 1] == 0 and g[2, 1, 1] == 0  # other classes unknown
    assert (g[:, 0, 1:] == 0).all()  # unreviewed background
    assert (g[:, 3, :] == 0).all()  # padding
    assert (reg[0].grad[:, :, 0, :] == 0).all()
    assert (ctr[0].grad[:, :, 0, :] == 0).all()
    assert (cls[1].grad == 0).all()  # small box out of second level's range


def test_class_negative_only_affects_reviewed_channel_and_protects_positive_all_levels():
    head, cls, reg, ctr, boxes, labels, negative = batch()
    negative.zero_()
    per_class = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
    per_class[:, 0, :8, 8:16] = 1
    per_class[:, 0, 8:16, 8:16] = 1  # explicit wrong-class negative on another positive
    per_class[:, 1, 8:24, 8:24] = 1  # contradictory positive is protected at every level
    per_class[:, 2, 24:] = 255  # padding is not a negative
    FCOSLoss(3)(
        head,
        cls,
        reg,
        ctr,
        boxes,
        labels,
        negative_mask=negative,
        class_negative_mask=per_class,
    )[0].backward()
    g = cls[0].grad[0]
    assert g[0, 0, 1] > 0
    assert (g[1:, 0, 1] == 0).all()
    assert g[1, 1, 1] < 0
    assert g[0, 1, 1] > 0 and g[2, 1, 1] == 0
    assert (g[:, 3, :] == 0).all()
    assert (cls[1].grad[:, 1:] == 0).all()
    assert cls[1].grad[0, 0, 0, 0] > 0
    assert (reg[0].grad[:, :, 0, :] == 0).all()
    assert (ctr[0].grad[:, :, 0, :] == 0).all()


@pytest.mark.parametrize(
    "bad", [torch.zeros(1, 32, 32), torch.zeros(1, 2, 32, 32), torch.full((1, 3, 32, 32), 2)]
)
def test_invalid_class_negatives_rejected(bad):
    head, cls, reg, ctr, boxes, labels, negative = batch()
    with pytest.raises(ValueError, match="class_negative_mask"):
        FCOSLoss(3)(
            head, cls, reg, ctr, boxes, labels, negative_mask=negative, class_negative_mask=bad
        )


def test_class_negative_cannot_turn_exhaustive_training_into_partial():
    head, cls, reg, ctr, boxes, labels, _ = batch()
    with pytest.raises(ValueError, match="requires partial"):
        FCOSLoss(3)(
            head, cls, reg, ctr, boxes, labels, class_negative_mask=torch.zeros(1, 3, 32, 32)
        )


def test_box_never_becomes_negative_on_an_unassigned_level():
    head, cls, reg, ctr, boxes, labels, negative = batch()
    negative.fill_(1)
    FCOSLoss(3)(head, cls, reg, ctr, boxes, labels, negative_mask=negative)[0].backward()
    assert (cls[1].grad == 0).all()  # all four level-2 points on/in box boundary
    assert cls[0].grad[0, 0, 1, 1] == 0


def test_empty_partial_batch_is_finite_zero_but_exhaustive_empty_has_negatives():
    head, cls, reg, ctr, _, _, negative = batch()
    boxes, labels = [torch.zeros(0, 4)], [torch.zeros(0, dtype=torch.long)]
    negative.zero_()
    partial, _ = FCOSLoss(3)(head, cls, reg, ctr, boxes, labels, negative_mask=negative)
    assert partial.item() == 0
    partial.backward()
    for tensor in cls + reg + ctr:
        assert tensor.grad is not None and (tensor.grad == 0).all()
    exhaustive, _ = FCOSLoss(3)(head, cls, reg, ctr, boxes, labels)
    assert exhaustive.item() > 0


def test_partial_channel_mask_and_unknown_logit_invariance():
    head, cls, reg, ctr, boxes, labels, negative = batch()
    channel = torch.tensor([[1.0, 1.0, 0.0]])
    loss_fn = FCOSLoss(3)
    before = loss_fn(head, cls, reg, ctr, boxes, labels, channel, negative)[0]
    with torch.no_grad():
        cls[0][:, :, 0, 1:] += 12  # high-confidence hallucinations in unknown area
        cls[0][:, 2] -= 12  # globally unsupervised channel
    after = loss_fn(head, cls, reg, ctr, boxes, labels, channel, negative)[0]
    torch.testing.assert_close(before, after)


@pytest.mark.parametrize("bad", [torch.zeros(32, 32), torch.full((1, 32, 32), 2)])
def test_malformed_partial_mask_rejected(bad):
    head, cls, reg, ctr, boxes, labels, _ = batch()
    with pytest.raises(ValueError, match="negative_mask"):
        FCOSLoss(3)(head, cls, reg, ctr, boxes, labels, negative_mask=bad)


def test_partial_loss_with_cpu_bfloat16_autocast():
    head, _, _, _, boxes, labels, negative = batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cls, reg, ctr = head([torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)])
        loss, _ = FCOSLoss(3)(head, cls, reg, ctr, boxes, labels, negative_mask=negative)
    loss.backward()
    assert torch.isfinite(loss) and head.cls_pred.weight.grad.abs().sum() > 0


def test_partial_dataset_rejected_before_coco_evaluation_or_model_mutation():
    from types import SimpleNamespace

    from syncai_hydranet.engine.evaluator import evaluate

    model = SimpleNamespace(det_head_name="detection")
    dataset = SimpleNamespace(partial_detection=True, supervises=["detection"])
    with pytest.raises(ValueError, match="exhaustive COCO mAP"):
        evaluate(model, [("partial", dataset)], {}, "cpu", None)


def test_partial_instance_config_refuses_wrong_class_order_and_coco_eval():
    from syncai_hydranet.config_schema import _check_detection_head_classes, _Report
    from syncai_hydranet.data.studioa_instances import CLASSES

    cfg = {
        "model": {
            "heads": {
                "detection": {
                    "type": "fcos",
                    "num_classes": len(CLASSES),
                    "classes": list(CLASSES),
                }
            }
        },
        "data": {"datasets": [{"type": "studioa_instances"}]},
    }
    report = _Report()
    _check_detection_head_classes(report, cfg)
    assert not report.errors
    cfg["model"]["heads"]["detection"]["classes"].reverse()
    cfg["data"]["datasets"][0]["split_val"] = "val"
    report = _Report()
    _check_detection_head_classes(report, cfg)
    assert any("class detection head order" in e for e in report.errors)
    assert any("COCO evaluation" in e for e in report.errors)
