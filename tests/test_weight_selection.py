"""One rule for choosing which weights a checkpoint hands back.

hydranet-eval and the ONNX export honoured a `--weights` flag while
hydranet-infer-image, hydranet-infer-video, bev_video and annotation_batch hardcoded
`ckpt.get("ema") or ckpt["model"]` -- always the average, with no way to ask for
anything else.

That is not a style difference. docs/TRAINING_GUIDE.md section 5 records a run scoring
0.16 mIoU on EMA weights against 0.95 on the raw ones from the same training, because
the average starts at the random initialisation and takes thousands of steps to forget
it. On a short run those are two different models -- and the tools with no way to say
which they had loaded were exactly the ones rendering pictures for a human to judge.

pytest tests/test_weight_selection.py -v
"""

import pytest
import torch

from syncai_hydranet.utils.checkpoint import select_weights

EMA = {"w": torch.ones(2)}
RAW = {"w": torch.zeros(2)}


def test_ema_is_preferred_when_present():
    assert select_weights({"model": RAW, "ema": EMA}) is EMA


def test_raw_weights_can_be_asked_for():
    """The capability the rendering tools did not have."""
    assert select_weights({"model": RAW, "ema": EMA}, "model") is RAW


def test_ema_falls_back_when_the_checkpoint_has_none():
    assert select_weights({"model": RAW}, "ema") is RAW


def test_an_empty_ema_is_not_mistaken_for_a_real_one():
    """A checkpoint written before the EMA ramp has an empty dict here, and loading it
    would give a model that is entirely its random initialisation."""
    assert select_weights({"model": RAW, "ema": {}}, "ema") is RAW


def test_a_typo_is_refused_rather_than_silently_meaning_raw():
    with pytest.raises(ValueError, match="prefer must be"):
        select_weights({"model": RAW, "ema": EMA}, "exponential")


def test_every_tool_that_loads_a_checkpoint_offers_the_choice():
    """The divergence was invisible because each tool looked reasonable on its own."""
    from syncai_hydranet.cli import evaluate, export_onnx, infer_image, infer_video

    for module in (evaluate, export_onnx, infer_image, infer_video):
        dests = {a.dest for a in module.build_parser()._actions}
        assert "weights" in dests, module.__name__
