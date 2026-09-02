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

from _export_cfg import seg_head, tiny_trunk
from syncai_hydranet.cli.export_onnx import ExportWrapper
from syncai_hydranet.models.hydranet import build_model

CFG = {
    "model": {
        **tiny_trunk(),
        "heads": {"traversability": seg_head()},
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


# ---------------------------------------------------------------------------
# Every head the model builds must reach the graph.
#
# `depth_heads` and `pose_heads` are their own ModuleDicts beside `seg_heads`, so a
# flattener written from `seg_heads` + detection omits them and says nothing. Pose was
# fixed in `a0c51de`; depth was still missing, and `configs/hydranet_hm3d_cctv.yaml`
# builds a depth head and nothing else -- so the tuple came out empty and the exported
# graph had zero outputs. `onnx.checker` passes such a graph; onnxruntime refuses to
# initialise it, which is how CI's export-parity job found it, two minutes in and with an
# error naming neither the head nor the config.
#
# These assert on the wrapper rather than on a real export because that is the level the
# bug lived at, and it runs in milliseconds instead of the ~20 s an ONNX round-trip costs.

DEPTH_ONLY_CFG = {
    "model": {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 32, "num_repeats": 1, "num_levels": 5},
        "heads": {
            "depth": {
                "type": "depth_fpn",
                "in_levels": [0, 1, 2],
                "channels": 32,
                "max_depth": 10.0,
            }
        },
        "loss_balancing": "uncertainty",
    }
}


def test_a_depth_only_model_exports_at_least_one_output():
    """The regression itself: an empty tuple is a graph onnxruntime cannot load."""
    wrapper = ExportWrapper(build_model(DEPTH_ONLY_CFG))
    with torch.no_grad():
        out = wrapper(torch.rand(1, 3, 64, 64) * 255.0)
    assert len(out) == 1, "the depth head produced no graph output"
    assert out[0].shape == (1, 1, 64, 64)


def test_depth_binding_is_named_for_its_units():
    """`_metres`: the head exponentiates, so this is metric depth, not logits."""
    wrapper = ExportWrapper(build_model(DEPTH_ONLY_CFG))
    assert wrapper.depth_output_names == ["depth_metres"]


def test_binding_names_account_for_every_output():
    """`check_parity` zips names against tensors with `strict=True`; a short name list
    would turn a silently mislabelled binding into a mismatch at export time instead."""
    wrapper = ExportWrapper(build_model(DEPTH_ONLY_CFG))
    names = wrapper.seg_output_names + wrapper.pose_output_names + wrapper.depth_output_names
    with torch.no_grad():
        out = wrapper(torch.rand(1, 3, 64, 64) * 255.0)
    assert len(names) == len(out)
