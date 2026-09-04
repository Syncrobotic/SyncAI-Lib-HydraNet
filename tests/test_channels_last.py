"""NHWC is a layout change, so the burden is to show it is only a layout change.

`train.channels_last` exists for one reason: on this box a bf16 training step goes from
153.2 ms to 121.1 ms. That is worth having only if the numbers coming out are the same
ones, so the equivalence test below is the point of this file and the rest is
guarding the two ways the flag can be wired up wrong -- a model that reports the wrong
layout, and an EMA copy that never got converted.

pytest tests/test_channels_last.py -v
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from syncai_hydranet.config import load_config
from syncai_hydranet.config_schema import check_config
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.seeding import apply_channels_last, model_memory_format

# A config that builds a multi-head model, which is all these tests need of it.
# It was the off-road one until 2026-09-04, when that taxonomy was retired;
# hydranet_indoor is the same shape and is still exercised elsewhere.
CFG = str(Path(__file__).resolve().parents[1] / "configs" / "hydranet_indoor.yaml")
H, W = 128, 160  # multiple of 128: P7 has stride 128


@pytest.fixture(scope="module")
def model():
    cfg = load_config(CFG, ["model.backbone.pretrained=false"])
    return build_model(cfg).eval()


# ------------------------------------------------- what layout is this model in


def test_a_plain_model_reports_nchw(model):
    assert model_memory_format(model) is torch.contiguous_format


def test_conversion_is_visible_to_the_reader():
    converted = build_model(load_config(CFG, ["model.backbone.pretrained=false"])).eval()
    apply_channels_last(converted, True)
    assert model_memory_format(converted) is torch.channels_last


def test_disabled_is_a_no_op(model):
    apply_channels_last(model, False)
    assert model_memory_format(model) is torch.contiguous_format


def test_a_1x1_conv_does_not_get_a_vote():
    """The tie this helper exists to break.

    A [out, in, 1, 1] weight satisfies both `is_contiguous()` and
    `is_contiguous(channels_last)`, and the plain check wins that tie. Reading the
    first 4-D parameter would therefore report NCHW for a converted model whose first
    convolution happens to be 1x1 -- and the only symptom would be a transpose per
    batch that nobody put there, i.e. the exact cost the flag was added to remove.
    """
    ambiguous = nn.Conv2d(8, 8, kernel_size=1)
    assert ambiguous.weight.is_contiguous(memory_format=torch.channels_last)

    model = nn.Sequential(ambiguous, nn.Conv2d(8, 8, kernel_size=3))
    apply_channels_last(model, True)
    assert model_memory_format(model) is torch.channels_last


def test_a_model_with_only_1x1_convs_is_reported_as_nchw():
    """The honest limit of the above: with nothing to break the tie, this reports NCHW
    even after conversion. Harmless -- feeding NCHW input to a model whose every weight
    is layout-agnostic costs nothing -- but it should be a known answer, not a surprise.
    """
    model = nn.Sequential(nn.Conv2d(8, 8, kernel_size=1))
    apply_channels_last(model, True)
    assert model_memory_format(model) is torch.contiguous_format


def test_something_without_weights_reports_nchw():
    """`evaluate` is duck-typed and its tests pass stubs, so the layout probe has to
    tolerate an object with no `parameters()`. NCHW is the right answer for one: the
    resulting `contiguous(contiguous_format)` on the input is a no-op."""

    class Stub:
        pass

    assert model_memory_format(Stub()) is torch.contiguous_format


# ------------------------------------------------------------- the equivalence


def _outputs(model, x):
    """Every tensor the model returns, flattened into one comparable list."""
    out = model(x)
    flat = []
    for key in sorted(out):
        v = out[key]
        flat.extend(v if isinstance(v, (list, tuple)) else [v])
    return flat


def _both_layouts(device):
    """The same model's outputs in NCHW and in NHWC, on the same input."""
    import copy

    torch.manual_seed(0)
    cfg = load_config(CFG, ["model.backbone.pretrained=false"])
    nchw = build_model(cfg).eval().to(device)
    nhwc = copy.deepcopy(nchw)
    apply_channels_last(nhwc, True)

    x = torch.randn(2, 3, H, W, device=device)
    with torch.no_grad():
        return (
            _outputs(nchw, x),
            _outputs(nhwc, x.contiguous(memory_format=torch.channels_last)),
        )


def _worst_relative_divergence(a, b) -> float:
    """Largest disagreement, measured against each tensor's own scale.

    A fixed atol is the wrong instrument here: these outputs span three orders of
    magnitude -- segmentation logits sit near 6, box regressions near 130 -- so one
    absolute threshold is either vacuous for the small ones or spuriously tight for
    the large ones.
    """
    assert len(a) == len(b)
    worst = 0.0
    for i, (u, v) in enumerate(zip(a, b, strict=True)):
        assert u.shape == v.shape, f"output {i}"
        scale = max(u.abs().max().item(), 1e-9)
        worst = max(worst, (u - v).abs().max().item() / scale)
    return worst


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nhwc_returns_the_same_numbers(device):
    """Not bit-identical, and deliberately not asserted as such.

    NHWC selects different convolution kernels, which sum in a different order, so the
    last few mantissa bits move. Measured across all 17 output tensors the divergence
    is 1e-6 to 1e-5 of each tensor's own scale -- the same class of difference as
    changing the batch size, and orders below anything a metric would register.

    TF32 is forced off for the comparison, because with it on the two layouts disagree
    by ~1e-2 (see the test below) and this test would be measuring TF32, not layout.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        a, b = _both_layouts(device)
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32

    assert _worst_relative_divergence(a, b) < 1e-4


def test_tf32_and_not_the_layout_is_what_moves_the_numbers():
    """Recorded because the obvious first reading of a failure here is wrong.

    Every real run sets `train.tf32=true`, and under TF32 the two layouts disagree by
    ~1e-2 of scale -- a thousand times the layout-only figure above. TF32 carries about
    ten mantissa bits, so a reordered accumulation shows up immediately; the layouts
    are not doing it, the precision is.

    The consequence is the reason `train.channels_last` defaults to false: flipping it
    partway through a run leaves the two halves not bit-comparable, so a control that
    exists to isolate one variable must not pick it up from a restart.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        a, b = _both_layouts("cuda")
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32

    assert _worst_relative_divergence(a, b) > 1e-4


def test_the_ema_copy_inherits_the_layout():
    """Validation scores the EMA weights, not the live ones.

    `ModelEMA` deep-copies the model, so converting after the EMA is built would leave
    the copy in NCHW: training would take the speedup and validation would quietly not,
    on a model whose reported layout no longer describes what runs.
    """
    from syncai_hydranet.engine.ema import ModelEMA

    cfg = load_config(CFG, ["model.backbone.pretrained=false"])
    model = build_model(cfg)
    apply_channels_last(model, True)
    ema = ModelEMA(model, 0.9998, 2000)
    assert model_memory_format(ema.ema) is torch.channels_last


# ------------------------------------------------------------------ the config


def _cfg_with(**train_overrides):
    cfg = load_config(CFG, ["model.backbone.pretrained=false"])
    cfg["train"].update(train_overrides)
    return cfg


def test_channels_last_is_a_legal_key():
    assert not [w for w in check_config(_cfg_with(channels_last=True)) if "channels_last" in w]


def test_channels_last_without_amp_warns_that_it_buys_nothing():
    warnings = check_config(_cfg_with(channels_last=True, amp=False))
    assert any("channels_last" in w and "autocast" in w for w in warnings)


def test_no_warning_when_the_flag_is_off():
    warnings = check_config(_cfg_with(channels_last=False, amp=False))
    assert not [w for w in warnings if "channels_last" in w]
