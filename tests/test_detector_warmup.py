"""A detector update must not mutate any scene weights, buffers or logits."""

import importlib.util
from pathlib import Path

import pytest
import torch

from syncai_hydranet.models.hydranet import HydraNet


@pytest.fixture
def worker():
    path = Path(__file__).resolve().parents[1] / "tools/annotation/studioa_train.py"
    spec = importlib.util.spec_from_file_location("warmup_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return {
        "model": {
            "backbone": {"name": "resnet18", "pretrained": False},
            "neck": {"name": "fpn", "out_channels": 16, "num_levels": 5},
            "heads": {
                "scene": {
                    "type": "semantic_fpn",
                    "num_classes": 3,
                    "channels": 16,
                    "dropout": 0.5,
                },
                "detection": {"type": "fcos", "num_classes": 2, "channels": 16, "num_convs": 1},
            },
            "loss_balancing": "fixed",
            "detection_only_training": True,
        }
    }


def test_detector_updates_preserve_scene_through_modes_and_reload(worker):
    torch.set_num_threads(2)
    model = HydraNet(config())
    reference = worker.frozen_state(model)
    before_head = model.det_head.cls_pred.weight.detach().clone()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in trainable} == {id(p) for p in model.det_head.parameters()}
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    images = torch.randn(2, 3, 64, 96)
    loaders = [("studioa", [{"image": images}])]
    before = worker.scene_signature(model, loaders, "cpu")
    for _ in range(3):
        model.eval().train()
        assert model.det_head.training
        output = model(images)
        targets = {
            "boxes": [torch.tensor([[16.0, 16.0, 48.0, 48.0]])] * 2,
            "labels": [torch.tensor([1])] * 2,
            "det_negative_mask": torch.zeros(2, 64, 96, dtype=torch.long),
        }
        loss, _ = model.compute_losses(output, targets, ["detection"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        worker.assert_frozen(model, reference)
    assert not torch.equal(before_head, model.det_head.cls_pred.weight)
    restored = HydraNet(config())
    restored.load_state_dict(model.state_dict())
    restored.eval().train()
    worker.assert_frozen(restored, reference)
    assert worker.scene_signature(restored, loaders, "cpu") == before
    with torch.no_grad():
        restored.neck.state_dict()[next(iter(restored.neck.state_dict()))].add_(1)
    with pytest.raises(ValueError, match="frozen scene state changed"):
        worker.assert_frozen(restored, reference)


def test_warmup_without_detection_rejected():
    cfg = config()
    del cfg["model"]["heads"]["detection"]
    with pytest.raises(ValueError, match="requires a detection head"):
        HydraNet(cfg)
