"""End-to-end Trainer runs on a synthetic dataset.

Every other test exercises a piece of the trainer through a stub. Nothing constructed
the real thing, which is how an import that ruff had removed survived a green test run:
the module imported fine, and the line that needed the name was never executed.

The fixture is four 64x80 images on disk and a two-block model, so a full run --
dataloaders, AMP path, accumulation, EMA, validation, checkpoints, run metadata --
takes a couple of seconds on CPU.

pytest tests/test_trainer.py -v
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from syncai_hydranet.config import Config
from syncai_hydranet.engine.trainer import Trainer
from syncai_hydranet.utils.checkpoint import load_checkpoint

ADE_FLOOR, ADE_WALL = 4, 1


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    for split, n in (("train", 4), ("val", 2)):
        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        rng = np.random.default_rng(0)
        for i in range(n):
            ann = np.full((64, 80), ADE_WALL, np.uint8)
            ann[32:] = ADE_FLOOR
            rgb = np.zeros((64, 80, 3), np.uint8)
            rgb[32:] = 180
            rgb = np.clip(rgb + rng.integers(-10, 10, rgb.shape), 0, 255).astype(np.uint8)
            Image.fromarray(rgb).save(img_dir / f"{i}.jpg")
            Image.fromarray(ann).save(ann_dir / f"{i}.png")
    return root


def _cfg(out_dir, data_root, **train):
    return Config(
        {
            "experiment": "pytest_run",
            "output_dir": str(out_dir),
            "seed": 0,
            "device": "cpu",
            "model": {
                "backbone": {"name": "resnet18", "pretrained": False},
                "neck": {"name": "fpn", "out_channels": 16, "num_levels": 5},
                "heads": {
                    "traversability": {
                        "type": "semantic_fpn",
                        "num_classes": 3,
                        "in_levels": [0, 1, 2],
                        "channels": 16,
                    }
                },
                "loss_balancing": "fixed",
                "fixed_weights": {"traversability": 1.0},
            },
            "data": {
                "input_size": [64, 80],
                "letterbox": True,
                "datasets": [
                    {
                        "name": "tiny",
                        "type": "seg_folder",
                        "root": str(data_root),
                        "split_train": "train",
                        "split_val": "val",
                        "label_map": "ade20k_indoor",
                        "supervises": ["traversability"],
                    }
                ],
                "workers": 0,
            },
            "train": {
                "epochs": 2,
                "batch_size": 2,
                "lr": 1.0e-3,
                "warmup_iters": 1,
                "amp": False,
                "ema": False,
                "log_interval": 1,
                **train,
            },
        }
    )


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory, data_root):
    out = tmp_path_factory.mktemp("run") / "exp"
    trainer = Trainer(_cfg(out, data_root))
    trainer.train()
    return trainer


# --------------------------------------------------------------- a whole run


def test_run_produces_every_artefact(finished_run):
    out = finished_run.out_dir
    for name in ("last.pt", "best.pt", "meta.json", "config.yaml", "metrics.jsonl"):
        assert (out / name).is_file(), f"{name} missing"


def test_metrics_file_has_one_row_per_epoch(finished_run):
    rows = [
        json.loads(x) for x in (finished_run.out_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [r["epoch"] for r in rows] == [1, 2]
    assert all("traversability_mIoU" in r for r in rows)
    assert all(r["primary_metric"] == "traversability_mIoU" for r in rows)


def test_meta_records_the_dataset_it_read(finished_run):
    meta = json.loads((finished_run.out_dir / "meta.json").read_text())
    ds = meta["datasets"][0]
    assert ds["train_size"] == 4 and ds["val_size"] == 2
    assert ds["splits"]["train"]["images"]["files"] == 4
    assert meta["steps_per_epoch"] == 2  # 4 images, batch 2


def test_checkpoint_is_at_the_final_epoch(finished_run):
    ckpt = load_checkpoint(finished_run.out_dir / "last.pt")
    assert ckpt["epoch"] == 2
    assert ckpt["global_step"] == 4  # 2 epochs x 2 steps
    assert ckpt["scheduler"]["it"] == 4


def test_the_model_learned_something(finished_run):
    """The fixture is trivially separable, so a working loop must beat chance."""
    rows = [
        json.loads(x) for x in (finished_run.out_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["traversability_mIoU"] > 0.0


# ------------------------------------------------------------------- resume


def test_resume_continues_from_the_saved_epoch(data_root, finished_run):
    cfg = _cfg(finished_run.out_dir, data_root, epochs=4)
    trainer = Trainer(cfg, resuming=True)
    trainer.load(str(finished_run.out_dir / "last.pt"), resume=True)
    assert trainer.start_epoch == 2
    assert trainer.global_step == 4
    assert trainer.scheduler.it == 4

    trainer.train()
    assert load_checkpoint(finished_run.out_dir / "last.pt")["epoch"] == 4


def test_a_second_fresh_run_does_not_overwrite_the_first(data_root, finished_run):
    cfg = _cfg(finished_run.out_dir, data_root, epochs=1)
    trainer = Trainer(cfg)  # resuming=False, and the directory is occupied
    assert trainer.out_dir != finished_run.out_dir
    assert trainer.out_dir.name.startswith(f"{finished_run.out_dir.name}-")


# ------------------------------------------------------- accumulation wiring


def test_accumulation_reduces_optimizer_steps(tmp_path, data_root):
    """Two micro-batches per epoch at accum=2 is one optimizer step, and the schedule
    must count that step, not the micro-batches."""
    out = tmp_path / "accum"
    trainer = Trainer(_cfg(out, data_root, grad_accum_steps=2, epochs=1))
    assert trainer.steps_per_epoch == 1
    trainer.train()
    ckpt = load_checkpoint(out / "last.pt")
    assert ckpt["global_step"] == 1
    assert ckpt["scheduler"]["it"] == 1


def test_accumulation_larger_than_an_epoch_is_refused(tmp_path, data_root):
    with pytest.raises(ValueError, match="no optimizer step"):
        Trainer(_cfg(tmp_path / "bad", data_root, grad_accum_steps=99))


def test_unknown_primary_metric_fails_at_the_first_validation(tmp_path, data_root):
    """Better than discovering at the end that best.pt was chosen by something else."""
    trainer = Trainer(_cfg(tmp_path / "pm", data_root, epochs=1, primary_metric="nope"))
    with pytest.raises(KeyError, match="traversability_mIoU"):
        trainer.train()


def test_resume_reports_config_drift(data_root, finished_run, caplog):
    """Extending epochs is routine; changing the learning rate on a resume usually is
    not, and the run's meta.json would otherwise record settings the weights never saw."""
    cfg = _cfg(finished_run.out_dir, data_root, epochs=8, lr=5.0e-3)
    trainer = Trainer(cfg, resuming=True)
    ckpt = load_checkpoint(finished_run.out_dir / "last.pt")

    with caplog.at_level("WARNING"):
        drift = trainer.report_config_drift(ckpt)

    paths = {where for where, _, _ in drift}
    assert "train.lr" in paths and "train.epochs" in paths
    assert any("train.lr" in r.message for r in caplog.records)


def test_no_drift_reported_for_an_unchanged_config(data_root, finished_run):
    cfg = _cfg(finished_run.out_dir, data_root, epochs=4)
    trainer = Trainer(cfg, resuming=True)
    ckpt = load_checkpoint(finished_run.out_dir / "last.pt")
    ckpt["cfg"] = dict(cfg)
    assert trainer.report_config_drift(ckpt) == []


# ----------------------------------------------------- the non-finite gradient guard


def _nan_after(trainer, n_micro_batches: int):
    """Wrap `compute_losses` so it returns NaN from the nth micro-batch onward."""
    real = trainer.model.compute_losses
    state = {"i": 0}

    def wrapped(outputs, targets, supervises):
        loss, logs = real(outputs, targets, supervises)
        state["i"] += 1
        if state["i"] > n_micro_batches:
            loss = loss * float("nan")
        return loss, logs

    trainer.model.compute_losses = wrapped
    return state


def test_a_non_finite_loss_skips_the_step_instead_of_poisoning_the_weights(tmp_path, data_root):
    """The guard `GradScaler` would provide if this project trained in fp16.

    It does not: `needs_grad_scaler` disables the scaler for bfloat16, bf16 is the
    default and what every run has used, so nothing skipped a poisoned step. Clipping is
    no substitute -- a NaN gradient makes the norm NaN, and clipping by a NaN coefficient
    writes NaN into every parameter. Runs the real loop rather than re-implementing it.
    """
    trainer = Trainer(_cfg(tmp_path / "exp", data_root, epochs=1))
    _nan_after(trainer, 0)  # every batch is poisoned
    trainer.train_one_epoch(0)

    assert trainer.nonfinite_steps > 0, "the guard never fired on an all-NaN epoch"
    params = list(trainer.model.parameters())
    assert all(torch.isfinite(p).all() for p in params), "NaN reached the parameters"


def test_the_guard_aborts_rather_than_training_through_a_dead_run(tmp_path, data_root):
    """Skipping exists for the isolated bad batch. A run that keeps producing them is
    gone, and an epoch of wall-clock spent proving it is what this refuses to pay."""
    from syncai_hydranet.engine.trainer import MAX_NONFINITE_STEPS

    trainer = Trainer(_cfg(tmp_path / "exp2", data_root, epochs=1))
    _nan_after(trainer, 0)
    trainer.nonfinite_steps = MAX_NONFINITE_STEPS - 1
    with pytest.raises(RuntimeError, match="non-finite gradient norms"):
        trainer.train_one_epoch(0)


def test_a_clean_epoch_leaves_the_counter_at_zero(tmp_path, data_root):
    """The other half: the guard must not fire on the fixture that trains fine, or it
    would be skipping real steps and reporting a healthy run as sick."""
    trainer = Trainer(_cfg(tmp_path / "exp3", data_root, epochs=1))
    trainer.train_one_epoch(0)
    assert trainer.nonfinite_steps == 0
