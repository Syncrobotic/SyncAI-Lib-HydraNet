"""A data comparison must share update, selection and class-weight budgets."""

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "tools/annotation/studioa_train.py"
spec = importlib.util.spec_from_file_location("scene_comparison_worker", path)
assert spec and spec.loader
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def config(tmp_path):
    manifest = {
        "folds": {
            "Tao-Hsin": {
                "pixels_by_split": {
                    "train": {name: i + 1 for i, name in enumerate(worker.CLASSES)}
                }
            }
        }
    }
    return worker.pilot_config(tmp_path / "data", tmp_path, manifest, "Tao-Hsin")


def test_update_and_validation_budgets_match_despite_dataset_size(tmp_path):
    original = config(tmp_path)
    weights = original["model"]["heads"]["scene"]["loss"]["class_weights"]
    normalized = []
    for count, epochs, interval in [(46, 45, 15), (62, 33, 11)]:
        cfg = worker.scene_comparison_config(copy.deepcopy(original), count, 495, 165, weights)
        assert cfg["train"]["epochs"] == epochs
        assert cfg["train"]["val_interval"] == interval
        steps = count // cfg["train"]["batch_size"]
        assert list(range(interval * steps, epochs * steps + 1, interval * steps)) == [
            165,
            330,
            495,
        ]
        assert cfg["train"]["early_stop_patience"] == 0
        assert cfg["train"]["deterministic"] and not cfg["train"]["tf32"]
        assert "split_test" not in cfg["data"]["datasets"][0]
        assert cfg["model"] == original["model"]
        cfg["train"].pop("epochs")
        cfg["train"].pop("val_interval")
        normalized.append(cfg)
    assert normalized[0] == normalized[1]
    assert original["data"]["datasets"][0]["split_test"] == "test"


@pytest.mark.parametrize(
    "count,updates,interval",
    [
        (3, 495, 165),
        (46, 494, 165),
        (62, 495, 164),
        (46, 495, 220),
        (46, 0, 165),
        (46, 495, 0),
    ],
)
def test_incomplete_epochs_and_unequal_selection_opportunities_rejected(
    tmp_path, count, updates, interval
):
    with pytest.raises(ValueError, match="align"):
        worker.scene_comparison_config(config(tmp_path), count, updates, interval, [1.0] * 19)


def test_weights_are_explicit_not_derived_from_evaluation_pixels(tmp_path):
    cfg = config(tmp_path)
    given = [0.25] * 19
    result = worker.scene_comparison_config(cfg, 62, 495, 165, given)
    assert result["model"]["heads"]["scene"]["loss"]["class_weights"] == given
    for bad in [[1.0] * 18, [float("nan")] * 19, [0.0] * 19]:
        with pytest.raises(ValueError, match="weights"):
            worker.scene_comparison_config(config(tmp_path), 62, 495, 165, bad)
    cfg = config(tmp_path)
    cfg["train"]["grad_accum_steps"] = 2
    with pytest.raises(ValueError, match="one batch"):
        worker.scene_comparison_config(cfg, 62, 495, 165, given)
