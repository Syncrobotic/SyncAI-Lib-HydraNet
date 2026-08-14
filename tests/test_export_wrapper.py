"""The export wrapper must not leave the model in train mode.

``nn.Module`` defaults to ``training=True``, and ``torch.onnx.export`` propagates the
module's mode down the tree. A wrapper that is never told to eval therefore hands the
inner model back in train mode after export, with BatchNorm switched from its running
statistics to whatever the last batch contained.

The exported graph itself is unaffected -- tracing forces eval -- so this is invisible
until something uses the model in the same process afterwards, at which point the numbers
quietly change. Measured on a real checkpoint before the fix: 20.2 absolute difference on
the traversability logits for an identical input.

pytest tests/test_export_wrapper.py -v
"""

import torch

from syncai_hydranet.cli.export_onnx import ExportWrapper
from syncai_hydranet.models.hydranet import build_model

CFG = {
    "model": {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 32, "num_repeats": 1, "num_levels": 5},
        "heads": {
            "traversability": {
                "type": "semantic_fpn",
                "num_classes": 3,
                "in_levels": [0, 1, 2],
                "channels": 32,
            }
        },
        "loss_balancing": "fixed",
        "fixed_weights": {"traversability": 1.0},
    }
}


def _wrapped():
    model = build_model(CFG)
    model.train()  # the state a caller could plausibly leave it in
    return ExportWrapper(model), model


def test_wrapper_is_built_in_eval_mode():
    wrapper, _ = _wrapped()
    assert not wrapper.training


def test_wrapping_puts_the_inner_model_in_eval_mode():
    """Wrapping a training model must not export training-mode BatchNorm behaviour."""
    _, model = _wrapped()
    assert not model.training


def test_batchnorm_layers_are_in_eval_mode():
    """The mode that actually matters: BN using running stats, not batch stats."""
    wrapper, _ = _wrapped()
    bns = [m for m in wrapper.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns, "expected BatchNorm layers in this backbone"
    assert not any(m.training for m in bns)


def test_output_is_deterministic_across_calls():
    """In train mode BatchNorm would update running stats and drift between calls."""
    wrapper, _ = _wrapped()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        first = wrapper(x)[0]
        second = wrapper(x)[0]
    assert torch.equal(first, second)


def test_forward_returns_a_tuple_of_tensors():
    """ONNX needs a flat tuple; a dict or nested list exports wrongly or not at all."""
    wrapper, _ = _wrapped()
    with torch.no_grad():
        out = wrapper(torch.randn(1, 3, 64, 64))
    assert isinstance(out, tuple)
    assert all(torch.is_tensor(t) for t in out)
