"""Three defects from the architecture review that share one shape.

None of them crashes. Each produces a plausible number that is wrong:

* four call sites had four detection score thresholds, so "the model finds nothing" and
  "the metric is depressed" looked the same;
* a second detection dataset overwrote the first's ground truth, scoring every box
  against the wrong annotations;
* three RELLIS mappings were inferred from class names, one of which decided whether
  standing water was `caution` or `blocked` -- retired 2026-09-04 with the off-road
  taxonomy, so the follow-up is closed by deletion rather than by verification.

pytest tests/test_review_followups.py -v
"""

import warnings
from typing import ClassVar

import pytest
import torch

from syncai_hydranet.cli import infer_image, infer_video
from syncai_hydranet.data.label_maps import get_scheme
from syncai_hydranet.models.heads.detection import SCORE_THR_EVAL, SCORE_THR_VIEW
from syncai_hydranet.models.hydranet import build_model

# ------------------------------------------------------------------ score thresholds


def test_the_two_thresholds_answer_different_questions():
    """Evaluation wants every box so the recall curve is complete; a viewer wants few
    boxes, mostly right. One shared default would be wrong for one of them."""
    assert SCORE_THR_EVAL < SCORE_THR_VIEW
    assert SCORE_THR_EVAL == 0.05, "COCOeval convention; raising it truncates the curve"


def test_every_call_site_uses_the_named_constants():
    """The failure this prevents: someone tunes a threshold in one place, and the other
    three keep their own numbers."""
    for module in (infer_image, infer_video):
        parser = module.build_parser()
        default = next(a.default for a in parser._actions if a.dest == "score_thr")
        assert default == SCORE_THR_VIEW, module.__name__


def test_predict_defaults_to_the_viewing_threshold():
    cfg = {
        "model": {
            "backbone": {"name": "resnet18", "pretrained": False},
            "neck": {"name": "fpn", "out_channels": 32, "num_levels": 3},
            "heads": {
                "detection": {
                    "type": "fcos",
                    "num_classes": 4,
                    "in_levels": [0, 1, 2],
                    "channels": 16,
                    "num_convs": 1,
                    "strides": [8, 16, 32],
                }
            },
        }
    }
    import inspect

    model = build_model(cfg).eval()
    assert inspect.signature(model.predict).parameters["score_thr"].default == SCORE_THR_VIEW
    with torch.no_grad():  # and it still runs
        model.predict(torch.zeros(1, 3, 64, 64))


@pytest.mark.parametrize("name", ["ade20k_indoor", "indoor_native", "retail_native"])
def test_verified_schemes_are_silent(name):
    """A warning on every scheme would be a warning nobody reads."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_scheme(name)


# ------------------------------------------------- two detection datasets, two ground truths


class FakeCOCO:
    """Stands in for a pycocotools COCO object, and records what was scored against it."""

    def __init__(self, name):
        self.name = name
        self.loaded = None

    def loadRes(self, results):  # noqa: N802 - pycocotools' spelling
        self.loaded = results
        return self


class FakeDetDataset:
    supervises = ("detection",)
    label_to_cat: ClassVar[dict] = {0: 1}

    def __init__(self, name):
        self.coco = FakeCOCO(name)


class FakeDetHead:
    def __init__(self, box):
        self.box = box

    def decode(self, *_args, **_kwargs):
        return [
            {
                "boxes": torch.tensor([self.box], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]


class FakeModel:
    seg_heads: ClassVar[dict] = {}
    det_head_name = "detection"
    training = False

    def __init__(self, box):
        self.det_head = FakeDetHead(box)

    def __call__(self, _images):
        # The keys evaluate() forwards to decode; their contents are irrelevant here
        # because FakeDetHead ignores them.
        return {"det_cls": [], "det_reg": [], "det_ctr": []}

    def eval(self):
        return self

    def train(self, _mode=True):
        return self


def _batch():
    return {
        "image": torch.zeros(1, 3, 8, 8),
        "targets": {},
        "image_ids": [7],
        "geoms": [{"scale": 1.0, "pad": (0, 0), "flipped": False}],
    }


def _evaluate_with(monkeypatch, names, box=(0.0, 0.0, 4.0, 4.0)):
    from syncai_hydranet.engine import evaluator

    stats = [0.5, 0.7] + [0.0] * 10

    class FakeCOCOeval:
        def __init__(self, gt, dt, _iou_type):
            self.gt, self.dt, self.stats = gt, dt, stats

        def evaluate(self): ...
        def accumulate(self): ...
        def summarize(self): ...

    import pycocotools.cocoeval as cocoeval

    monkeypatch.setattr(cocoeval, "COCOeval", FakeCOCOeval)
    monkeypatch.setattr(evaluator, "invert_geom", lambda boxes, _geom: boxes)

    datasets = [(n, FakeDetDataset(n)) for n in names]
    loaders = [(n, [_batch()]) for n in names]
    cfg = {"train": {"batch_size": 2}, "data": {"workers": 0, "terrain_classes": []}}

    class Log:
        def info(self, *_a): ...

    metrics = evaluator.evaluate(
        FakeModel(box), datasets, cfg, torch.device("cpu"), Log(), loaders=loaders
    )
    return metrics, dict(datasets)


def test_one_detection_dataset_keeps_the_unqualified_metric_key(monkeypatch):
    """`train.primary_metric: detection_mAP` and every metrics.jsonl written so far
    depend on this key not moving."""
    metrics, _ = _evaluate_with(monkeypatch, ["coco"])
    assert metrics["detection_mAP"] == 0.5
    assert metrics["detection_mAP50"] == 0.7


def test_two_detection_datasets_are_scored_separately(monkeypatch):
    """Previously both shared one results list and one ground truth, the second
    overwriting the first -- so every box was scored against the wrong annotations and
    the resulting mAP was wrong without any error."""
    metrics, datasets = _evaluate_with(monkeypatch, ["coco", "site"])
    assert set(metrics) == {
        "detection_mAP/coco",
        "detection_mAP50/coco",
        "detection_mAP/site",
        "detection_mAP50/site",
    }
    # Each dataset's own ground truth received its own detections.
    for name, ds in datasets.items():
        assert ds.coco.loaded, f"{name} was never scored"
        assert len(ds.coco.loaded) == 1
