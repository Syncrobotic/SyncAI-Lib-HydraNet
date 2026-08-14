"""The detection loss has to survive mixed precision.

Regression test for a crash that reached a training run untouched. Under autocast the
class logits are bf16 or fp16, so the one-hot target built from them must adopt their
dtype -- ``index_put_`` refuses a float32 source into a bf16 destination:

    RuntimeError: Index put requires the source and destination dtypes match,
                  got BFloat16 for the destination and Float for the source.

It survived because reaching it needs three things at once: AMP enabled, a CUDA-class
device (``supports_amp()`` disables AMP on MPS and CPU, leaving everything fp32 where the
old cast happened to line up), and the detection head *actually supervised*. Until COCO
was added no run ever called this loss, which is why 166 passing tests and a full
60-epoch run never touched the line.

The tests below force autocast explicitly rather than relying on the device, so they
reproduce on CPU and therefore run in CI.

pytest tests/test_amp_detection_loss.py -v
"""

import pytest
import torch

from syncai_hydranet.models.heads.detection import build_det_head
from syncai_hydranet.models.losses import FCOSLoss

NUM_CLASSES = 4
HEAD_CFG = {
    "num_classes": NUM_CLASSES,
    "in_levels": [0, 1],
    "channels": 16,
    "num_convs": 1,
    "strides": [8, 16],
}
AMP_DTYPES = [
    pytest.param(torch.bfloat16, id="bfloat16"),
    pytest.param(torch.float16, id="float16"),
]


def _batch():
    torch.manual_seed(0)
    feats = [torch.randn(1, 16, 16, 16), torch.randn(1, 16, 8, 8)]
    boxes = [torch.tensor([[10.0, 10.0, 60.0, 60.0]])]
    labels = [torch.tensor([1])]
    return feats, boxes, labels


def _run(dtype, boxes=None, labels=None):
    """One forward + loss under autocast, returning the total loss."""
    head = build_det_head(HEAD_CFG, 16)
    loss_fn = FCOSLoss(NUM_CLASSES)
    feats, default_boxes, default_labels = _batch()
    with torch.autocast(device_type="cpu", dtype=dtype):
        cls_out, reg_out, ctr_out = head(feats)
        assert cls_out[0].dtype == dtype, "autocast did not take effect; test is vacuous"
        total, _ = loss_fn(
            head,
            cls_out,
            reg_out,
            ctr_out,
            default_boxes if boxes is None else boxes,
            default_labels if labels is None else labels,
        )
    return total


# ------------------------------------------------------- the regression


@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_detection_loss_runs_under_autocast(dtype):
    """The crash this file exists for: fails with the pre-fix .float() cast."""
    assert torch.isfinite(_run(dtype))


@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_loss_is_finite_with_no_positive_targets(dtype):
    """An image with no boxes takes the other branch, which must also survive."""
    total = _run(dtype, boxes=[torch.zeros(0, 4)], labels=[torch.zeros(0, dtype=torch.long)])
    assert torch.isfinite(total)


def test_full_precision_still_works():
    """Guards against fixing AMP by breaking the path every MPS run uses."""
    head = build_det_head(HEAD_CFG, 16)
    feats, boxes, labels = _batch()
    cls_out, reg_out, ctr_out = head(feats)
    total, parts = FCOSLoss(NUM_CLASSES)(head, cls_out, reg_out, ctr_out, boxes, labels)
    assert torch.isfinite(total)
    assert parts, "loss should report its components"


@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_gradients_reach_the_head_under_autocast(dtype):
    """A loss that runs but detaches would pass every check above and train nothing."""
    head = build_det_head(HEAD_CFG, 16)
    loss_fn = FCOSLoss(NUM_CLASSES)
    feats, boxes, labels = _batch()
    with torch.autocast(device_type="cpu", dtype=dtype):
        cls_out, reg_out, ctr_out = head(feats)
        total, _ = loss_fn(head, cls_out, reg_out, ctr_out, boxes, labels)
    total.float().backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads, "no detection-head parameter received a gradient"
    assert sum(float(g.abs().sum()) for g in grads) > 0


@pytest.mark.parametrize("dtype", AMP_DTYPES)
def test_amp_and_fp32_losses_agree(dtype):
    """Reduced precision should cost precision, not correctness."""
    amp = float(_run(dtype).detach())
    head = build_det_head(HEAD_CFG, 16)
    feats, boxes, labels = _batch()
    cls_out, reg_out, ctr_out = head(feats)
    fp32 = float(FCOSLoss(NUM_CLASSES)(head, cls_out, reg_out, ctr_out, boxes, labels)[0])
    assert amp == pytest.approx(fp32, rel=0.05), f"amp {amp} vs fp32 {fp32}"
