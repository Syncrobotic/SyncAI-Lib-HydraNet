"""Partial labels establish recall, never ordinary precision or exhaustive mAP."""

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from syncai_hydranet.engine.partial_detection import PartialDetectionAccumulator


def prediction(boxes, labels, scores=None):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels),
        "scores": torch.tensor(scores if scores is not None else [0.9] * len(boxes)),
    }


def test_matching_is_class_aware_one_to_one_and_unknown_is_not_false_alarm():
    acc = PartialDetectionAccumulator(["phone", "tablet"])
    gt = torch.tensor([[2, 2, 8, 8], [2, 2, 8, 8], [12, 2, 18, 8]])
    labels = torch.tensor([0, 0, 1])
    det = prediction([[2, 2, 8, 8], [12, 2, 18, 8], [2, 12, 8, 18]], [0, 0, 1])
    acc.update(det, gt, labels, torch.zeros(20, 20))
    metrics = acc.metrics("val")
    assert metrics["partial_det/val/matched"] == 1
    assert metrics["partial_det/val/recall50_at_020"] == pytest.approx(1 / 3)
    assert metrics["partial_det/val/unknown_predictions"] == 2
    assert metrics["partial_det/val/empty_fp"] == 0
    assert not any("mAP" in k or "precision" in k for k in metrics)


def test_empty_region_requires_whole_box_fraction_and_padding_is_unknown():
    acc = PartialDetectionAccumulator(["phone"])
    mask = torch.zeros(30, 30)
    mask[:10, :10] = 1
    mask[20:, :] = 255
    det = prediction([[0, 0, 10, 10], [0, 0, 20, 10], [0, 20, 10, 30]], [0, 0, 0])
    acc.update(det, torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), mask)
    metrics = acc.metrics("val")
    assert metrics["partial_det/val/empty_fp"] == 1
    assert metrics["partial_det/val/unknown_predictions"] == 2
    assert metrics["partial_det/val/empty_fp_per_megapixel"] == 10000
    assert not any("recall" in k for k in metrics)


def test_thresholds_are_fixed_and_invalid_predictions_fail():
    acc = PartialDetectionAccumulator(["phone"])
    gt = torch.tensor([[0, 0, 10, 10]])
    det = prediction([[0, 0, 5, 10], [0, 0, 10, 10]], [0, 0], [0.9, 0.2])
    acc.update(det, gt, torch.tensor([0]), torch.zeros(20, 20))
    assert acc.matched.tolist() == [1]  # IoU exactly 0.5 accepted, score 0.2 excluded
    with pytest.raises(ValueError, match="clipped"):
        acc.update(
            prediction([[-1, 0, 5, 10]], [0]), gt, torch.tensor([0]), torch.zeros(20, 20)
        )


def test_evaluator_routes_explicit_partial_protocol_without_coco():
    from syncai_hydranet.engine.evaluator import evaluate

    det = prediction([[1, 1, 8, 8]], [0])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seg_heads = {}
            self.det_head_name = "detection"
            self.det_head = SimpleNamespace(decode=lambda *_args, **_kw: [det])

        def forward(self, _images):
            return {"det_cls": [], "det_reg": [], "det_ctr": []}

    model = Model()
    ds = SimpleNamespace(
        partial_detection=True,
        partial_detection_evaluation="reviewed_regions_v1",
        detection_classes=["phone"],
        supervises=["detection"],
    )
    batch = {
        "image": torch.zeros(1, 3, 20, 20),
        "targets": {
            "boxes": [det["boxes"]],
            "labels": [det["labels"]],
            "det_negative_mask": torch.zeros(1, 20, 20),
        },
    }
    result = evaluate(
        model,
        [("val", ds)],
        {"data": {}},
        "cpu",
        logging.getLogger(__name__),
        loaders=[("val", [batch])],
    )
    assert result["partial_det/val/recall50_at_020"] == 1
    assert model.training


def test_joint_config_never_declares_test_and_warm_start_checks_taxonomy(tmp_path):
    from syncai_hydranet.models.hydranet import HydraNet

    path = Path(__file__).resolve().parents[1] / "tools/annotation/studioa_train.py"
    spec = importlib.util.spec_from_file_location("joint_worker_test", path)
    assert spec and spec.loader
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    from syncai_hydranet.data.studioa_supervision import CLASSES

    manifest = {
        "folds": {"Tao-Hsin": {"pixels_by_split": {"train": dict.fromkeys(CLASSES, 1)}}}
    }
    original = worker.pilot_config(tmp_path, tmp_path, manifest, "Tao-Hsin")
    original["model"]["backbone"]["pretrained"] = False
    source = HydraNet(original)
    import copy

    cfg = worker.joint_config(copy.deepcopy(original), tmp_path, "Tao-Hsin")
    assert all("split_test" not in ds for ds in cfg["data"]["datasets"])
    model = HydraNet(cfg)
    worker.warm_start(model, {"cfg": original, "model": source.state_dict()}, cfg)
    torch.testing.assert_close(
        model.state_dict()["seg_heads.scene.classifier.weight"],
        source.state_dict()["seg_heads.scene.classifier.weight"],
    )
    original["data"]["terrain_classes"].reverse()
    with pytest.raises(ValueError, match="compatible scene-only"):
        worker.warm_start(model, {"cfg": original, "model": source.state_dict()}, cfg)
