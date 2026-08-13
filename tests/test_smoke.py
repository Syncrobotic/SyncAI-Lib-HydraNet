"""End-to-end smoke tests on random tensors. No dataset required.

pytest tests/test_smoke.py -v
"""

from pathlib import Path

import pytest
import torch

from syncai_hydranet.config import load_config
from syncai_hydranet.models.hydranet import build_model

CFG = str(Path(__file__).resolve().parents[1] / "configs" / "hydranet_regnet800mf.yaml")
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
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(2, 3, H, W)
    out = model(x)
    targets = {
        "traversability": torch.randint(0, 3, (2, H, W)),
        "terrain": torch.randint(0, 12, (2, H, W)),
    }
    loss, _ = model.compute_losses(out, targets, ["traversability", "terrain"])
    opt.zero_grad()
    loss.backward()
    opt.step()


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
