"""End-to-end smoke tests on random tensors. No dataset required.

pytest tests/test_smoke.py -v
"""

from pathlib import Path

import pytest
import torch

from syncai_hydranet.config import load_config
from syncai_hydranet.models.hydranet import build_model

# A config that builds a multi-head model, which is all these tests need of it.
# It was the off-road one until 2026-09-04, when that taxonomy was retired;
# hydranet_indoor is the same shape and is still exercised elsewhere.
CFG = str(Path(__file__).resolve().parents[1] / "configs" / "hydranet_indoor.yaml")
H, W = 128, 160  # must be a multiple of 128: P7 has stride 128


@pytest.fixture(scope="module")
def model():
    cfg = load_config(CFG, ["model.backbone.pretrained=false"])
    return build_model(cfg)


def test_forward_shapes(model):
    x = torch.randn(2, 3, H, W)
    out = model(x)
    assert out["traversability"].shape == (2, 3, H, W)
    assert out["terrain"].shape == (2, 12, H, W)
    assert len(out["det_cls"]) == 5
    assert out["det_cls"][0].shape == (2, 80, H // 8, W // 8)
    assert out["det_reg"][0].shape == (2, 4, H // 8, W // 8)
    assert (out["det_reg"][0] > 0).all()  # exp() keeps distances positive


def test_loss_and_backward(model):
    x = torch.randn(2, 3, H, W)
    out = model(x)
    # Segmentation batch, as a RUGD or ADE20K step would produce.
    seg_targets = {
        "traversability": torch.randint(0, 3, (2, H, W)),
        "terrain": torch.randint(0, 12, (2, H, W)),
    }
    loss_seg, _ = model.compute_losses(out, seg_targets, ["traversability", "terrain"])
    assert torch.isfinite(loss_seg)
    loss_seg.backward()

    # Detection batch, as a COCO step would produce.
    out = model(x)
    det_targets = {
        "boxes": [
            torch.tensor([[10.0, 10.0, 60.0, 80.0], [30.0, 20.0, 100.0, 90.0]]),
            torch.zeros(0, 4),
        ],
        "labels": [torch.tensor([1, 5]), torch.zeros(0, dtype=torch.long)],
    }
    loss_det, logs_det = model.compute_losses(out, det_targets, ["detection"])
    assert torch.isfinite(loss_det)
    loss_det.backward()
    assert "detection" in logs_det


def test_partial_supervision_isolates_heads(model):
    """A segmentation-only step must leave the detection head's gradients untouched."""
    model.zero_grad(set_to_none=True)
    x = torch.randn(2, 3, H, W)
    out = model(x)
    targets = {
        "traversability": torch.randint(0, 3, (2, H, W)),
        "terrain": torch.randint(0, 12, (2, H, W)),
    }
    loss, _ = model.compute_losses(out, targets, ["traversability", "terrain"])
    loss.backward()
    det_grads = [p.grad for p in model.det_head.parameters() if p.grad is not None]
    assert all(g.abs().sum() == 0 for g in det_grads)
    # ...while the shared trunk does receive gradient.
    trunk_grad = sum(p.grad.abs().sum() for p in model.neck.parameters() if p.grad is not None)
    assert trunk_grad > 0


def test_train_step(model):
    """One step of the training critical path, asserted rather than merely executed.

    This ran forward -> backward -> step with **no assertion at all** until 2026-09-04,
    so it passed with a NaN loss, with an optimizer that updated nothing, and with every
    parameter turned to NaN by the step it had just taken. Its neighbour at the top of
    this file already asserts `torch.isfinite(loss)`; this one had dropped it.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(2, 3, H, W)
    out = model(x)
    targets = {
        "traversability": torch.randint(0, 3, (2, H, W)),
        "terrain": torch.randint(0, 12, (2, H, W)),
    }
    loss, _ = model.compute_losses(out, targets, ["traversability", "terrain"])
    assert torch.isfinite(loss), "a non-finite loss poisons the parameters on the step below"
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    opt.zero_grad()
    loss.backward()
    opt.step()
    after = [p for p in model.parameters() if p.requires_grad]
    assert all(torch.isfinite(p).all() for p in after), "the step wrote NaN into the weights"
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True)), (
        "the optimizer step changed nothing"
    )


def test_predict(model):
    model.eval()
    x = torch.randn(1, 3, H, W)
    result = model.predict(x, score_thr=0.99)
    assert result["traversability"].shape == (1, H, W)
    assert result["terrain"].shape == (1, H, W)
    assert "boxes" in result["detection"][0]
    model.train()


def test_onnx_export(model, tmp_path):
    onnx = pytest.importorskip("onnx")
    from syncai_hydranet.cli.export_onnx import ExportWrapper

    wrapper = ExportWrapper(model.eval())
    out_file = str(tmp_path / "m.onnx")
    torch.onnx.export(
        wrapper,
        torch.randn(1, 3, H, W),
        out_file,
        input_names=["images"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(out_file))
    model.train()


def test_public_api_is_importable():
    """__all__ that names something the package does not export breaks `from x import *`
    and misleads anyone reading it for the supported surface."""
    import syncai_hydranet
    from syncai_hydranet import data, engine, utils

    for module in (syncai_hydranet, data, engine, utils):
        missing = [n for n in module.__all__ if not hasattr(module, n)]
        assert not missing, f"{module.__name__} exports nothing named {missing}"


def test_version_has_a_single_source():
    """pyproject reads the version out of __init__, so the two cannot drift."""
    from importlib.metadata import version

    import syncai_hydranet

    assert version("syncai-hydranet") == syncai_hydranet.__version__


def test_a_step_with_no_supervised_head_says_so():
    """The config schema rejects this, but the runtime is the last line and the one that
    would otherwise fail as `'NoneType' object has no attribute 'detach'` -- an error
    that names neither the dataset, the head, nor the config key responsible."""
    cfg = load_config(CFG, ["model.backbone.pretrained=false"])
    model = build_model(cfg).eval()
    out = model(torch.randn(1, 3, 64, 80))
    targets = {"traversability": torch.zeros(1, 64, 80, dtype=torch.long)}
    with pytest.raises(ValueError, match="supervis"):
        model.compute_losses(out, targets, [])
