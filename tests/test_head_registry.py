"""The head registry, and the constraint that shaped it.

`HydraNet.heads()` is what lets `forward`, `compute_losses` and `predict` iterate instead
of unrolling detection by hand. The design constraint is not aesthetic: PyTorch registers
submodules by attribute assignment, so wrapping the heads in module objects would move
every state_dict key one level deeper and stop every checkpoint written so far from
loading. The adapters are therefore a view over modules that never move, and the first
test here is the one that says so.

pytest tests/test_head_registry.py -v
"""

from __future__ import annotations

import torch

from syncai_hydranet.config import load_config
from syncai_hydranet.models.heads.registry import DetectionHead, SegmentationHead
from syncai_hydranet.models.hydranet import HydraNet

CFG = "configs/hydranet_indoor.yaml"


def _model():
    return HydraNet(load_config(CFG, ["model.backbone.pretrained=false"]))


def _seg_only_model():
    """Built from a dict rather than a config file on purpose.

    Dropping the head from a shipped config leaves its datasets declaring
    `supervises: [detection]`, which the schema rightly refuses -- the config would be
    inconsistent, not merely smaller. What is under test here is the model, so the model
    is what gets built.
    """
    return HydraNet(
        {
            "model": {
                "backbone": {"name": "resnet18", "pretrained": False},
                "neck": {"name": "fpn", "out_channels": 16, "num_levels": 5},
                "heads": {
                    "traversability": {
                        "type": "semantic_fpn",
                        "num_classes": 3,
                        "in_levels": [0, 1, 2],
                        "channels": 16,
                    }
                },
                "loss_balancing": "fixed",
                "fixed_weights": {"traversability": 1.0},
            }
        }
    )


def test_the_registry_does_not_move_a_single_state_dict_key():
    """The whole reason the adapters are not nn.Modules. `det_head.cls_pred.weight` is
    what ten checkpoints on disk say, and a wrapper would have made it
    `heads.detection.module.cls_pred.weight` -- silently, for anything loading with
    strict=False."""
    prefixes = {k.split(".")[0] for k in _model().state_dict()}
    assert prefixes == {"backbone", "neck", "seg_heads", "det_head", "balancer"}


def test_every_head_appears_exactly_once():
    heads = _model().heads()
    assert [h.name for h in heads] == ["traversability", "terrain", "detection"]
    assert len({h.name for h in heads}) == len(heads)


def test_segmentation_comes_before_detection():
    """Not cosmetic. The order reaches the loss balancer, which stacks the terms it is
    handed, so changing it changes the arithmetic of every training step."""
    kinds = [type(h) for h in _model().heads()]
    assert kinds.index(DetectionHead) == len(kinds) - 1
    assert all(k is SegmentationHead for k in kinds[:-1])


def test_a_model_with_no_detection_head_yields_only_segmentation():
    m = _seg_only_model()
    heads = m.heads()
    assert all(isinstance(h, SegmentationHead) for h in heads)
    assert "det_head" not in {k.split(".")[0] for k in m.state_dict()}


def test_the_adapters_hold_the_registered_modules_and_not_copies():
    """A copy would train one set of weights and export the other."""
    m = _model()
    by_name = {h.name: h for h in m.heads()}
    assert by_name["traversability"].module is m.seg_heads["traversability"]
    assert by_name["detection"].module is m.det_head
    assert by_name["detection"].loss_fn is m.det_loss


def test_each_head_writes_only_its_own_keys_into_the_forward_dict():
    """Heads are independent by design rule 2: one may be added or removed without
    touching the others, which only holds if none of them reads or overwrites another's
    output."""
    m = _model().eval()
    feats = m.neck(m.backbone(torch.zeros(1, 3, 384, 480)))
    written = {}
    for head in m.heads():
        out: dict = {}
        head.forward_into(out, feats, (384, 480))
        written[head.name] = set(out)
    assert written["traversability"] == {"traversability"}
    assert written["terrain"] == {"terrain"}
    assert written["detection"] == {"det_cls", "det_reg", "det_ctr"}
    seen = [k for keys in written.values() for k in keys]
    assert len(seen) == len(set(seen)), "two heads claim the same output key"


def test_naming_detection_in_supervises_is_taken_at_its_word():
    """Deliberately asymmetric with segmentation, which also checks its targets arrived.
    Adding that check here would turn a dataset whose collate produces no boxes from a
    loud KeyError into a run that trains one head fewer and says nothing."""
    det = next(h for h in _model().heads() if isinstance(h, DetectionHead))
    assert det.supervised_by({}) is True
    seg = next(h for h in _model().heads() if isinstance(h, SegmentationHead))
    assert seg.supervised_by({}) is False
