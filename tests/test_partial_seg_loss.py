"""Unknown pixels must not contribute CE/Dice loss or gradients."""

import pytest
import torch
import torch.nn.functional as F

from syncai_hydranet.labels import IGNORE
from syncai_hydranet.models.losses import SegLoss


@pytest.mark.parametrize("weights", [None, [0.1, 0.3, 2.0]])
def test_partial_ce_matches_weighted_mean_and_all_ignore_has_zero_gradient(weights):
    logits = torch.randn(2, 3, 4, 5, requires_grad=True)
    target = torch.randint(0, 3, (2, 4, 5))
    target[:, :2] = IGNORE
    loss = SegLoss(3, dice_weight=0, class_weights=weights)
    expected = F.cross_entropy(
        logits,
        target,
        ignore_index=IGNORE,
        weight=None if weights is None else torch.tensor(weights),
    )
    torch.testing.assert_close(loss(logits, target), expected)
    empty = torch.full_like(target, IGNORE)
    zero = SegLoss(3, class_weights=weights)(logits, empty)
    assert zero.item() == 0
    zero.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


def test_ignored_pixel_logits_do_not_change_ce_or_dice():
    target = torch.tensor([[[0, IGNORE], [1, IGNORE]]])
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    loss_fn = SegLoss(3)
    modified = logits.detach().clone()
    modified[:, :, :, 1] = torch.tensor([100, -100, 20]).view(1, 3, 1)
    torch.testing.assert_close(loss_fn(logits, target), loss_fn(modified, target))
    loss_fn(logits, target).backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad[:, :, :, 1]) == 0
