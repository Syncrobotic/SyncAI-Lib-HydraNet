"""Unknown objects must receive no false-background gradient in partial FCOS batches."""

import pytest
import torch

from syncai_hydranet.models.heads.detection import FCOSHead
from syncai_hydranet.models.losses import FCOSLoss


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
