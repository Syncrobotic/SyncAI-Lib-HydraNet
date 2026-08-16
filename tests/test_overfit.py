"""Can the training loop actually learn?

Shape tests pass on a model that is wired backwards -- a detached tensor, a target
misaligned with its logits, a head whose gradients never reach the trunk. The cheapest
check that separates "it runs" from "it trains" is to memorise a single batch: if the
loop is correct, a network with millions of parameters drives the loss on four fixed
images to near zero. If it cannot, no amount of data or epochs will help.

Runs on CPU in well under a minute, so it belongs in CI rather than in a nightly.

pytest tests/test_overfit.py -v
"""

import pytest
import torch

from syncai_hydranet.engine.optim import WarmupCosine, build_optimizer
from syncai_hydranet.models.hydranet import build_model

H, W = 128, 160
BATCH = 4
STEPS = 60

MODEL_CFG = {
    "model": {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 32, "num_repeats": 1, "num_levels": 5},
        "heads": {
            "traversability": {
                "type": "semantic_fpn",
                "num_classes": 3,
                "in_levels": [0, 1, 2],
                "channels": 32,
                "dropout": 0.0,  # dropout only adds noise when the goal is memorisation
            }
        },
        "loss_balancing": "fixed",
        "fixed_weights": {"traversability": 1.0},
    }
}
TRAIN_CFG = {"lr": 3.0e-3, "optimizer": "adamw", "weight_decay": 0.0}


def _fixed_batch(seed=0):
    """Four images whose top third, middle and bottom third are three distinct classes.

    Learnable from the pixels alone, so failure means the loop is broken, not that the
    task is hard.
    """
    g = torch.Generator().manual_seed(seed)
    target = torch.zeros(BATCH, H, W, dtype=torch.long)
    target[:, H // 3 : 2 * H // 3] = 1
    target[:, 2 * H // 3 :] = 2
    images = torch.randn(BATCH, 3, H, W, generator=g) * 0.1
    images += torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float()
    return images, target


def _train(steps=STEPS):
    torch.manual_seed(0)
    model = build_model(MODEL_CFG)
    optimizer = build_optimizer(model, TRAIN_CFG)
    scheduler = WarmupCosine(optimizer, warmup_iters=5, total_iters=steps)
    images, target = _fixed_batch()

    model.train()
    losses = []
    for _ in range(steps):
        out = model(images)
        loss, _ = model.compute_losses(out, {"traversability": target}, ["traversability"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
    return model, images, target, losses


@pytest.fixture(scope="module")
def overfit():
    return _train()


def test_loss_falls_by_an_order_of_magnitude(overfit):
    _, _, _, losses = overfit
    assert losses[-1] < losses[0] / 10, f"loss went {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_loss_is_finite_throughout(overfit):
    """A NaN at any point means the loss, not the optimiser, is the problem."""
    _, _, _, losses = overfit
    assert all(torch.isfinite(torch.tensor(x)) for x in losses)


def test_the_batch_is_essentially_memorised(overfit):
    model, images, target, _ = overfit
    model.eval()
    with torch.no_grad():
        pred = model(images)["traversability"].argmax(1)
    accuracy = (pred == target).float().mean().item()
    assert accuracy > 0.95, f"only memorised {accuracy:.1%} of a four-image batch"


def test_gradients_reach_the_shared_trunk(overfit):
    """The heads are cheap; if the backbone never receives gradient, the trunk that
    carries 84% of the parameters is decoration."""
    model, images, target, _ = overfit
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(images)
    loss, _ = model.compute_losses(out, {"traversability": target}, ["traversability"])
    loss.backward()
    grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert grads, "no backbone parameter received a gradient"
    assert sum(float(g.abs().sum()) for g in grads) > 0
