"""Object-mean regression preserves identity and never reweights the other tasks."""

import pytest
import torch

from syncai_hydranet.models.heads.detection import FCOSHead
from syncai_hydranet.models.losses import FCOSLoss, giou_loss, object_mean_giou


def test_assignment_ids_preserve_overlap_ties_and_same_class_objects():
    head = FCOSHead(8, 2, in_levels=[0, 1], channels=8, num_convs=1, strides=[8, 16])
    boxes = torch.tensor(
        [[0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 32.0, 32.0], [0.0, 0.0, 16.0, 16.0]]
    )
    labels = torch.tensor([0, 0, 1])
    shapes = [(4, 4), (2, 2)]
    full = head.get_targets_with_ids(shapes, [boxes], [labels], "cpu")
    old = head.get_targets(shapes, [boxes], [labels], "cpu")
    torch.testing.assert_close(full[:4], old, rtol=0, atol=0)
    assert full[4][0, :16].reshape(4, 4).tolist() == [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    assert (full[4][0, 16:] == -1).all()
    assert not (full[4] == 2).any()  # equal-area tie stays with the first GT
    empty = head.get_targets_with_ids(shapes, [boxes[:0]], [labels[:0]], "cpu")
    assert (empty[4] == -1).all()


def test_object_mean_not_point_mean_and_replication_invariant():
    pred = torch.tensor([[[4.0] * 4, [2.0] * 4, [2.0] * 4]], requires_grad=True)
    target = torch.full_like(pred, 2.0)
    ids = torch.tensor([[0, 1, 1]])
    loss = object_mean_giou(pred, target, ids)
    assert float(loss.detach()) == pytest.approx(0.375)
    assert float(giou_loss(pred[0], target[0]).detach() / 3) == pytest.approx(0.25)
    selection = torch.tensor([0, 0, 0, 0, 1, 2])
    repeated = object_mean_giou(pred[:, selection], target[:, selection], ids[:, selection])
    torch.testing.assert_close(loss, repeated)
    g = torch.autograd.grad(loss, pred, retain_graph=True)[0]
    rg = torch.autograd.grad(repeated, pred)[0]
    torch.testing.assert_close(g, rg)


def test_batch_ids_are_local_and_denominator_counts_objects_not_images():
    pred = torch.tensor([[[4.0] * 4, [2.0] * 4, [2.0] * 4], [[2.0] * 4] * 3])
    target = torch.full_like(pred, 2.0)
    ids = torch.tensor([[0, 1, 1], [0, -1, -1]])
    assert float(object_mean_giou(pred, target, ids)) == pytest.approx(0.25)
    equal_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
    torch.testing.assert_close(
        object_mean_giou(pred, target, equal_ids), giou_loss(pred, target) / 6
    )


def test_empty_objects_do_not_create_regression_gradients():
    pred = torch.ones(2, 3, 4, requires_grad=True)
    loss = object_mean_giou(pred, torch.zeros_like(pred), torch.full((2, 3), -1))
    loss.backward()
    assert loss == 0 and (pred.grad == 0).all()
    with pytest.raises(ValueError, match="regression_normalization"):
        FCOSLoss(2, regression_normalization="bad")


def loss_case(device, dtype, mode, empty=False):
    head = FCOSHead(8, 2, in_levels=[0], channels=8, num_convs=1, strides=[8]).to(device)
    cls = torch.zeros(1, 2, 4, 4, device=device, dtype=dtype, requires_grad=True)
    reg = torch.full((1, 4, 4, 4), 8.0, device=device, dtype=dtype, requires_grad=True)
    ctr = torch.zeros(1, 1, 4, 4, device=device, dtype=dtype, requires_grad=True)
    boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0], [8.0, 0.0, 32.0, 32.0]], device=device)
    labels = torch.tensor([0, 1], device=device)
    if empty:
        boxes, labels = boxes[:0], labels[:0]
    loss, logs = FCOSLoss(
        2, positive_classification="assigned_object", regression_normalization=mode
    )(
        head,
        [cls],
        [reg],
        [ctr],
        [boxes],
        [labels],
        negative_mask=torch.zeros(1, 32, 32, device=device, dtype=torch.uint8),
    )
    loss.backward()
    return loss, logs, cls.grad, reg.grad, ctr.grad


def test_only_positive_regression_gradients_change():
    old = loss_case("cpu", torch.float32, "positive_point_mean")
    new = loss_case("cpu", torch.float32, "assigned_object_mean")
    for key in ["det_cls", "det_ctr"]:
        torch.testing.assert_close(old[1][key], new[1][key], rtol=0, atol=0)
    torch.testing.assert_close(old[2], new[2], rtol=0, atol=0)
    torch.testing.assert_close(old[4], new[4], rtol=0, atol=0)
    assert not torch.equal(old[3], new[3])
    assert (new[3][0, :, 1:, 0] == 0).all()  # unassigned points


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("empty", [False, True])
def test_cuda_autocast_deterministic_backward(dtype, empty):
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.autocast("cuda", dtype=dtype):
            result = loss_case("cuda", dtype, "assigned_object_mean", empty)
        assert torch.isfinite(result[0])
        assert all(torch.isfinite(t).all() for t in result[2:])
        if empty:
            assert result[0] == 0
            assert all((t == 0).all() for t in result[2:])
    finally:
        torch.use_deterministic_algorithms(previous)
