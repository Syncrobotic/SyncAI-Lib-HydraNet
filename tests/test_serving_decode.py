"""The serving decoder must agree with FCOSHead.decode on the same tensors.

serving/decode.py mirrors the model's own decode for a process that holds TensorRT
output buffers instead of a model object. Two implementations of one decode is one
too many unless something holds them to the same answer -- this does.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from syncai_hydranet.models.heads.detection import FCOSHead
from syncai_hydranet.serving.decode import FcosDecoder

NUM_CLASSES = 4
SHAPES = [(16, 20), (8, 10), (4, 5), (2, 3), (1, 2)]
IMG = (128, 160)


@pytest.fixture(scope="module")
def levels():
    g = torch.Generator().manual_seed(7)
    cls = [torch.randn(2, NUM_CLASSES, h, w, generator=g) for h, w in SHAPES]
    # reg is already exp'd and stride-scaled in the model's forward, so positive.
    reg = [torch.rand(2, 4, h, w, generator=g) * 40 + 1 for h, w in SHAPES]
    ctr = [torch.randn(2, 1, h, w, generator=g) for h, w in SHAPES]
    return cls, reg, ctr


def test_parity_with_model_decode(levels):
    cls, reg, ctr = levels
    head = FCOSHead(in_channels=16, num_classes=NUM_CLASSES, channels=16, num_convs=1)
    ref = head.decode(cls, reg, ctr, score_thr=0.3, nms_thr=0.6, img_size=IMG)
    dec = FcosDecoder(num_classes=NUM_CLASSES)
    got = dec(
        [c.numpy() for c in cls],
        [r.numpy() for r in reg],
        [c.numpy() for c in ctr],
        score_thr=0.3,
        img_size=IMG,
    )
    assert len(got) == len(ref) == 2
    for r, g in zip(ref, got, strict=True):
        assert len(r["boxes"]) > 0, "degenerate test: nothing passed the threshold"
        np.testing.assert_allclose(g["boxes"], r["boxes"].numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(g["scores"], r["scores"].numpy(), rtol=1e-5)
        np.testing.assert_array_equal(g["labels"], r["labels"].numpy())


def test_per_class_thresholds_gate_each_class_independently(levels):
    cls, reg, ctr = levels
    # max_det lifted: truncation would let boxes below the cut enter the gated run
    # once class 1's boxes free slots, which is not the property under test.
    dec = FcosDecoder(num_classes=NUM_CLASSES, max_det=10_000)
    args = ([c.numpy() for c in cls], [r.numpy() for r in reg], [c.numpy() for c in ctr])
    low = dec(*args, score_thr=[0.05, 0.05, 0.05, 0.05], img_size=IMG)
    gated = dec(*args, score_thr=[0.05, 1.1, 0.05, 0.05], img_size=IMG)
    assert any(1 in r["labels"] for r in low)
    assert all(1 not in r["labels"] for r in gated)
    # NMS is class-aware (batched_nms), so closing class 1's gate must not disturb
    # any other class: every remaining detection existed in the low-threshold set.
    for lo, ga in zip(low, gated, strict=True):
        lo_set = {tuple(np.round(b, 4)) for b in lo["boxes"]}
        assert all(tuple(np.round(b, 4)) in lo_set for b in ga["boxes"])


def test_score_thr_shape_is_validated(levels):
    cls, reg, ctr = levels
    dec = FcosDecoder(num_classes=NUM_CLASSES)
    with pytest.raises(ValueError, match="per class"):
        dec(
            [c.numpy() for c in cls],
            [r.numpy() for r in reg],
            [c.numpy() for c in ctr],
            score_thr=[0.3, 0.3],
        )


def test_pre_nms_topk_leaves_the_kept_set_unchanged():
    """Suppression only ever comes from a higher-scored box, and every higher-scored
    box is inside the top-k too -- so as long as NMS keeps max_det boxes from the
    top-k, the kept set is identical to the untruncated decode."""
    g = torch.Generator().manual_seed(3)
    shapes = [(16, 20), (8, 10), (4, 5), (2, 3), (1, 2)]
    cls = [torch.randn(1, NUM_CLASSES, h, w, generator=g) for h, w in shapes]
    reg = [torch.rand(1, 4, h, w, generator=g) * 40 + 1 for h, w in shapes]
    ctr = [torch.randn(1, 1, h, w, generator=g) for h, w in shapes]
    args = ([c.numpy() for c in cls], [r.numpy() for r in reg], [c.numpy() for c in ctr])
    full = FcosDecoder(num_classes=NUM_CLASSES)(*args, score_thr=0.05, img_size=IMG)
    fast = FcosDecoder(num_classes=NUM_CLASSES, pre_nms_topk=256)(
        *args, score_thr=0.05, img_size=IMG
    )
    for f, q in zip(full, fast, strict=True):
        # The lengths first, and that is the whole point of the test. Comparing
        # `[:min(len(f), len(q))]` -- which this did until 2026-09-04 -- passes when
        # truncation DROPS a detection, because the shorter list simply shortens the
        # comparison. The regression this exists to catch was invisible to it.
        assert len(f["boxes"]) > 0
        assert len(q["boxes"]) == len(f["boxes"]), (
            "pre_nms_topk changed how many boxes survived; the kept set is not unchanged"
        )
        np.testing.assert_allclose(q["boxes"], f["boxes"])
        np.testing.assert_array_equal(q["labels"], f["labels"])
