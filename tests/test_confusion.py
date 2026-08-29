"""`per_image_confusions`, now that it is in the package rather than in a script.

It moved out of `scripts/site_confusion.py`, where `scripts/val_sampling_error.py` reached
it through a `sys.path` insert, and its own docstring had already said why it must not be
copied: "the alternative was to copy this loop, whose upsample-before-argmax step is easy
to get quietly wrong". These tests are what makes that claim checkable, since the thing at
risk is not whether the function runs -- it is whether the matrix means what the caller
reads it as.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from syncai_hydranet.engine.confusion import per_image_confusions, run_config
from syncai_hydranet.labels import IGNORE

CLASSES = ["floor", "wall", "column", "fixture", "person", "glass", "stairs"]


def _cfg() -> dict:
    return {
        "model": {
            "backbone": {"name": "resnet18", "pretrained": False},
            "neck": {"name": "fpn", "out_channels": 32, "num_repeats": 1, "num_levels": 5},
            "heads": {
                "terrain": {
                    "type": "semantic_fpn",
                    "num_classes": len(CLASSES),
                    "in_levels": [0, 1, 2],
                    "channels": 32,
                }
            },
            "loss_balancing": "fixed",
            "fixed_weights": {"terrain": 1.0},
        },
        "data": {"input_size": [64, 64], "terrain_classes": CLASSES},
    }


@pytest.fixture
def run(tmp_path):
    """A finished run: `config.yaml` plus a checkpoint, which is all the function reads."""
    from syncai_hydranet.models.hydranet import build_model

    d = tmp_path / "run"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump(_cfg()), encoding="utf-8")
    model = build_model(_cfg())
    torch.save({"model": model.state_dict()}, d / "best.pt")
    return d


@pytest.fixture
def dataset(tmp_path):
    """Three images, one of which has no annotation and must be skipped rather than counted.

    The label PNG carries *raw* ids that a label map translates, and the two are not the
    same number: under `indoor_native` raw 1 is class 1 but raw **0 is IGNORE**, so a test
    that wrote 0 expecting the first class would pass for entirely the wrong reason. One
    image is filled with a raw id the map sends to IGNORE, so its contribution to the
    matrix must be exactly zero without the image being dropped from `kept`.
    """
    root = tmp_path / "ds"
    for split in ("val",):
        (root / "images" / split).mkdir(parents=True)
        (root / "annotations" / split).mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (48, 32), (40 * i, 60, 80)).save(
                root / "images" / split / f"f{i}.jpg"
            )
        for i in range(2):  # f2.jpg deliberately gets none
            raw = np.full((32, 48), 1 if i == 0 else 255, dtype=np.uint8)
            Image.fromarray(raw).save(root / "annotations" / split / f"f{i}.png")
    return root


def _call(run, dataset, **kw):
    args = {
        "checkpoint": "best.pt",
        "weights": "model",
        "label_map": "indoor_native",
        "dataset": str(dataset),
        "split": "val",
        "device": torch.device("cpu"),
    }
    args.update(kw)
    return per_image_confusions(run, **args)


# ------------------------------------------------------------------------- run_config


def test_run_config_reads_a_finished_run_without_validating_it(tmp_path):
    """The whole point of it: a run from this morning need not satisfy this afternoon's
    check, and refusing to *analyse* it loses the only evidence about what it did."""
    d = tmp_path / "run"
    d.mkdir()
    (d / "config.yaml").write_text(
        yaml.safe_dump({"data": {"terrain_classes": ["a", "b"]}, "nonsense_key": 1}),
        encoding="utf-8",
    )
    cfg = run_config(d)
    assert cfg["data"]["terrain_classes"] == ["a", "b"]
    assert cfg["nonsense_key"] == 1


# --------------------------------------------------------------------------- the shape


def test_the_stack_is_one_matrix_per_kept_image(run, dataset):
    mats, classes, kept = _call(run, dataset)
    assert mats.shape == (2, len(CLASSES), len(CLASSES))
    assert classes == CLASSES
    assert [p.name for p in kept] == ["f0.jpg", "f1.jpg"]


def test_an_image_with_no_annotation_is_skipped_not_counted_empty(run, dataset):
    """`kept` is what `mats` is about. Counting f2 as an all-zero matrix would make the
    per-image spread `val_sampling_error.py` measures include an image nobody labelled."""
    _, _, kept = _call(run, dataset)
    assert not any(p.name == "f2.jpg" for p in kept)


def test_kept_and_mats_stay_in_the_same_order(run, dataset):
    """`val_sampling_error.py` resamples images by index across both."""
    mats, _, kept = _call(run, dataset)
    assert len(mats) == len(kept)
    assert kept == sorted(kept)


# ------------------------------------------------------------------- what it counts


def test_every_labelled_pixel_lands_somewhere_and_ignore_lands_nowhere(run, dataset):
    """f0 is one class over 32x48; f1 is the sentinel throughout and must contribute 0."""
    mats, _, kept = _call(run, dataset)
    by_name = dict(zip([p.name for p in kept], mats, strict=True))
    assert by_name["f0.jpg"].sum() == 32 * 48
    assert by_name["f1.jpg"].sum() == 0, "IGNORE pixels are not a class and are not counted"


def test_the_truth_axis_is_the_first_one(run, dataset):
    """`[truth, prediction]`, which is the reading every caller does. A transposed matrix
    is still square, still sums right, and says the opposite thing about which class is
    eating which."""
    mats, _, kept = _call(run, dataset)
    f0 = mats[[p.name for p in kept].index("f0.jpg")]
    rows = np.flatnonzero(f0.sum(axis=1))
    assert rows.tolist() == [1], "indoor_native maps raw 1 to class 1, so only row 1 has truth"


def test_the_matrix_is_integer_counts(run, dataset):
    mats, _, _ = _call(run, dataset)
    assert mats.dtype == np.int64


# --------------------------------------------------------------------------- refusals


def test_an_empty_split_is_refused_by_name(run, tmp_path):
    empty = tmp_path / "empty"
    (empty / "images" / "val").mkdir(parents=True)
    with pytest.raises(SystemExit, match="no images under"):
        _call(run, empty)


def test_the_sentinel_comes_from_the_one_definition():
    """Guarded for real by `tests/test_ignore_is_one_definition.py`; asserted here because
    this module is the one that masks with it now."""
    from syncai_hydranet.engine import confusion

    assert confusion.IGNORE is IGNORE


def test_the_keyword_only_signature_cannot_be_called_positionally(run, dataset):
    """The reason the argparse Namespace went away: `dataset` and `split` are both strings
    and a positional call could swap them silently."""
    with pytest.raises(TypeError):
        per_image_confusions(run, "best.pt", "model", "indoor_native", str(dataset), "val")  # ty: ignore[too-many-positional-arguments]


def test_it_defaults_to_picking_a_device_when_none_is_given(run, dataset):
    mats, _, _ = _call(run, dataset, device=None)
    assert len(mats) == 2


def test_a_string_device_is_accepted(run, dataset):
    mats, _, _ = _call(run, dataset, device="cpu")
    assert len(mats) == 2


def test_a_missing_checkpoint_is_an_error_naming_the_path(run, dataset):
    with pytest.raises(RuntimeError, match=r"nope\.pt"):
        _call(run, dataset, checkpoint="nope.pt")
