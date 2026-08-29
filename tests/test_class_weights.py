"""Per-class weighting on the segmentation CE, and the guard on its length.

`fixture` reached 92.6% recall at 56.1% precision on batch02's six held-out cameras --
as agreement with SAM 3, not accuracy -- while `column`, `product`, `person` and `floor`
all sat at 91-96% precision against 21-69% recall. One class absorbing and the rest
under-firing is frequency bias. `dice_weight` 1.5 did not settle it, because Dice is
`dice.mean()` and already class-balanced; the unweighted CE beside it is what pulls.

pytest tests/test_class_weights.py -v
"""

import pytest
import torch

from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.models.losses import SegLoss

CLASSES = 4


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, CLASSES, 8, 8, generator=g)
    target = torch.randint(0, CLASSES, (2, 8, 8), generator=g)
    return logits, target


def test_omitting_weights_is_the_old_behaviour_exactly():
    """Off by default: every config that predates this must produce the same number."""
    logits, target = _batch()
    plain = SegLoss(CLASSES, dice_weight=0.0)
    assert plain.class_weights is None
    expected = torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
    assert torch.allclose(plain(logits, target), expected)


def test_uniform_weights_change_nothing():
    """A sanity anchor: if all-ones moved the loss, the wiring would be wrong."""
    logits, target = _batch()
    a = SegLoss(CLASSES, dice_weight=0.0)(logits, target)
    b = SegLoss(CLASSES, dice_weight=0.0, class_weights=[1.0] * CLASSES)(logits, target)
    assert torch.allclose(a, b, atol=1e-6)


def test_upweighting_a_class_raises_the_loss_on_its_errors():
    """The point of the knob: a rare class's mistakes have to cost more.

    Target is all class 3 and the logits favour class 0, so every pixel is a class-3
    error. Weighting class 3 up must increase the loss.
    """
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(1, CLASSES, 8, 8, generator=g)
    logits[:, 0] += 4.0
    target = torch.full((1, 8, 8), 3, dtype=torch.long)
    base = SegLoss(CLASSES, dice_weight=0.0)(logits, target)
    up = SegLoss(CLASSES, dice_weight=0.0, class_weights=[1, 1, 1, 5.0])(logits, target)
    assert up > base


def test_a_wrong_length_is_refused_rather_than_broadcast():
    """A short list would reweight the wrong classes and still train to completion --
    the failure this project keeps finding, one silent index shift away."""
    with pytest.raises(ValueError, match="3 entries for 4 classes"):
        SegLoss(CLASSES, class_weights=[1.0, 2.0, 3.0])


def test_the_weights_follow_the_module_but_stay_out_of_the_checkpoint():
    """A buffer so `.to(device)` moves it -- CE raises on a device mismatch and that would
    only fire once training reached the GPU. Non-persistent so it is absent from
    `state_dict()`: these are a config value, not learned state, and a persistent buffer
    makes every checkpoint that predates this unloadable into a weighted model. The next
    test pins that consequence, because it is the one that would be found in a fine-tune.
    """
    loss = SegLoss(CLASSES, class_weights=[1.0, 2.0, 3.0, 4.0])
    assert loss.class_weights.dtype == torch.float
    assert "class_weights" not in loss.state_dict()
    assert "class_weights" in dict(loss.named_buffers())


def test_a_checkpoint_from_before_this_still_loads_into_a_weighted_model():
    """Turning weights on and fine-tuning from an existing run must work.

    With a persistent buffer this raised `Missing key(s) in state_dict:
    seg_losses.terrain.class_weights`, verified against
    runs/hydranet_retail_objects_site_batch02/best.pt before the fix.
    """
    unweighted = SegLoss(CLASSES, dice_weight=0.0)
    weighted = SegLoss(CLASSES, dice_weight=0.0, class_weights=[1, 1, 1, 5.0])
    weighted.load_state_dict(unweighted.state_dict())  # strict, and must not raise
    assert weighted.class_weights.tolist() == [1, 1, 1, 5.0], "the weights must survive"


def test_a_config_can_turn_it_on():
    """Wired through `build_model`, so the config key is not decoration."""
    cfg = {
        "model": {
            "backbone": {"name": "resnet18", "pretrained": False},
            "neck": {"name": "fpn", "out_channels": 32, "num_repeats": 1, "num_levels": 5},
            "heads": {
                "terrain": {
                    "type": "semantic_fpn",
                    "num_classes": CLASSES,
                    "in_levels": [0, 1, 2],
                    "channels": 32,
                    "loss": {"class_weights": [0.5, 1.0, 2.0, 4.0]},
                }
            },
        },
        "data": {"input_size": [64, 80], "datasets": []},
    }
    model = build_model(cfg)
    assert model.seg_losses["terrain"].class_weights.tolist() == [0.5, 1.0, 2.0, 4.0]
