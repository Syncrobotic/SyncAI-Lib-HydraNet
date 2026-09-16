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


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)
def test_real_comparison_worker_trains_and_selects_without_building_test_data(
    tmp_path, monkeypatch
):
    import json

    import torch

    from _studioa_fixture import source_package
    from syncai_hydranet.data import datasets
    from syncai_hydranet.data.studioa_review import digest, write_json
    from syncai_hydranet.data.studioa_supervision import check_supervision, export_supervision
    from syncai_hydranet.engine import trainer
    from syncai_hydranet.utils.checkpoint import load_checkpoint

    source_package(tmp_path / "annotations")
    out = tmp_path / "run"
    out.mkdir()
    data = out / "data"
    export_supervision(tmp_path / "annotations", data)
    manifest = check_supervision(data)
    cfg = worker.pilot_config(data, out, manifest, "Tao-Hsin")
    cfg["device"] = "cpu"
    cfg["model"]["backbone"]["pretrained"] = False
    cfg["data"].update(input_size=[32, 64], workers=0)
    cfg["train"].update(batch_size=2, amp=False, warmup_iters=1)
    # The scheduler deliberately starts at zero LR; the second step must update.
    cfg = worker.scene_comparison_config(cfg, 2, 2, 1, [1.0] * 19)
    write_json(out / "config.json", cfg)
    write_json(
        out / "job.json",
        {
            "scene_comparison": True,
            "git_commit": "synthetic-worker-test",
            "counts": manifest["folds"]["Tao-Hsin"]["counts"],
            "files": {
                str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()
            },
        },
    )
    # Exercise the actual worker on CPU, bypassing only its hardware admission check.
    hardware_admission = iter([True])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: next(hardware_admission, False))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "CPU test fixture")
    original = datasets.build_dataset
    splits = []

    def guarded(config, size, split, **kwargs):
        assert split != "test", "comparison must never build the held-out test dataset"
        splits.append(split)
        return original(config, size, split, **kwargs)

    monkeypatch.setattr(datasets, "build_dataset", guarded)
    monkeypatch.setattr(trainer, "build_dataset", guarded)
    worker.run(out)
    report = json.loads((out / "report.json").read_text())
    assert set(splits) == {"train", "val"}
    assert report["status"] == "completed" and report["test_evaluated"] is False
    assert report["last_epoch"] == 2 and not (out / "test.json").exists()
    assert "initial_state.pt" in report["outputs"] and "validation.json" in report["outputs"]
    assert load_checkpoint(out / "model/last.pt")["global_step"] == 2
    initial = torch.load(out / "initial_state.pt", weights_only=True)
    last = load_checkpoint(out / "model/last.pt")["model"]
    assert not torch.equal(
        initial["seg_heads.scene.classifier.weight"], last["seg_heads.scene.classifier.weight"]
    )
