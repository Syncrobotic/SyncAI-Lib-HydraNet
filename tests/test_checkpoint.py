"""Checkpoint round-trip and resume-correctness tests.

A resumed run must continue the LR schedule, not replay it. These tests use a stub
Trainer (no datasets, no GPU) so they run in CI.

pytest tests/test_checkpoint.py -v
"""

import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from syncai_hydranet.engine.optim import WarmupCosine
from syncai_hydranet.engine.trainer import CKPT_FORMAT, Trainer
from syncai_hydranet.utils.checkpoint import load_checkpoint, save_checkpoint

WARMUP, TOTAL = 10, 100


def _optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.1)


class _FakeLoader:
    """Stands in for MultiTaskLoader; only ``len`` is used on the resume path."""

    def __init__(self, steps):
        self.steps = steps

    def __len__(self):
        return self.steps


def _stub_trainer(steps_per_epoch=10):
    """A Trainer with just the attributes save/load touch."""
    t = object.__new__(Trainer)
    t.model = nn.Linear(4, 2)
    t.ema = None
    t.optimizer = _optimizer(t.model)
    t.scheduler = WarmupCosine(t.optimizer, WARMUP, TOTAL)
    t.scaler = torch.amp.GradScaler(enabled=False)
    t.global_step = 0
    t.best_metric = -1.0
    # Patience state rides in the checkpoint for the same reason best_metric does: a
    # preempted run that resumes with a reset counter either stops early or never stops.
    t.epochs_since_best = 0
    t.start_epoch = 0
    t.cfg = {"train": {"epochs": 10}, "data": {"input_size": [128, 160]}}
    t.train_loader = _FakeLoader(steps_per_epoch)
    t.accum_steps = 1
    t.steps_per_epoch = steps_per_epoch  # optimizer steps, so accum is already divided out
    t.logger = logging.getLogger("test")
    return t


# ------------------------------------------------------------------ scheduler


def test_resumed_schedule_matches_uninterrupted_run():
    """The whole point: LRs after a resume must equal the ones a single run would give."""
    ref_model = nn.Linear(4, 2)
    ref_opt = _optimizer(ref_model)
    ref_sched = WarmupCosine(ref_opt, WARMUP, TOTAL)
    reference = []
    for _ in range(TOTAL):
        ref_sched.step()
        reference.append(ref_opt.param_groups[0]["lr"])

    model = nn.Linear(4, 2)
    opt = _optimizer(model)
    sched = WarmupCosine(opt, WARMUP, TOTAL)
    for _ in range(60):
        sched.step()
    state = sched.state_dict()

    fresh_opt = _optimizer(nn.Linear(4, 2))
    fresh = WarmupCosine(fresh_opt, WARMUP, TOTAL)
    fresh.load_state_dict(state)
    # load_state_dict re-applies immediately: the LR must already be the resumed one,
    # not the base LR, or the first step after resume runs far too hot.
    assert fresh_opt.param_groups[0]["lr"] == reference[59]

    resumed = []
    for _ in range(40):
        fresh.step()
        resumed.append(fresh_opt.param_groups[0]["lr"])
    assert resumed == reference[60:]


def test_the_first_optimizer_step_does_not_run_at_the_base_lr():
    """The 500x bug: the trainer steps the optimizer BEFORE the scheduler, so the LR the
    param groups hold at construction is what the first step runs at.

    `WarmupCosine.__init__` recorded `base_lrs` and never applied the schedule, so that
    LR was the base -- `2.0e-4` against a warmup wanting `4.0e-7` on the shipped config.
    Asserted on the param group rather than on `_factor`, because the param group is what
    the optimizer reads.
    """
    opt = _optimizer(nn.Linear(4, 2))
    base = opt.param_groups[0]["lr"]
    WarmupCosine(opt, WARMUP, TOTAL)
    assert opt.param_groups[0]["lr"] < base / WARMUP * 2, (
        "the schedule was not applied at construction; the first step runs at the base LR"
    )


def test_schedule_decays_and_ends_near_zero():
    opt = _optimizer(nn.Linear(4, 2))
    sched = WarmupCosine(opt, WARMUP, TOTAL)
    sched.step()
    assert opt.param_groups[0]["lr"] < 0.1  # still warming up
    for _ in range(WARMUP - 1):
        sched.step()
    assert opt.param_groups[0]["lr"] == 0.1  # warmup complete, full base LR
    for _ in range(TOTAL - WARMUP):
        sched.step()
    assert opt.param_groups[0]["lr"] < 1e-6  # cosine floor


# ------------------------------------------------------------------ round trip


def test_checkpoint_round_trip_restores_full_train_state(tmp_path):
    t = _stub_trainer()
    for _ in range(37):
        t.scheduler.step()
    t.global_step = 37
    t.best_metric = 0.4231

    path = tmp_path / "last.pt"
    torch.save(t.state_dict(epoch=4), path)

    ckpt = load_checkpoint(path)  # weights_only=True: no arbitrary unpickling
    assert ckpt["format"] == CKPT_FORMAT

    fresh = _stub_trainer()
    fresh.model.load_state_dict(ckpt["model"])
    fresh.load_train_state(ckpt)

    assert fresh.scheduler.it == 37
    assert fresh.global_step == 37
    assert fresh.best_metric == 0.4231
    assert fresh.start_epoch == 4
    assert fresh.optimizer.param_groups[0]["lr"] == t.optimizer.param_groups[0]["lr"]
    assert fresh.cfg == t.cfg


def test_checkpoint_survives_the_safe_loader(tmp_path):
    """The config travels inside the checkpoint, so it must stay plain-container only."""
    t = _stub_trainer()
    t.cfg = {
        "experiment": "x",
        "seed": 42,
        "data": {"input_size": [512, 640], "datasets": [{"name": "ade20k", "ratio": 1.0}]},
        "train": {"lr": 2.0e-4, "amp": True, "device": None},
    }
    path = tmp_path / "last.pt"
    torch.save(t.state_dict(epoch=1), path)
    assert load_checkpoint(path)["cfg"] == t.cfg


def test_legacy_checkpoint_estimates_schedule_position(tmp_path):
    """Checkpoints written before format 2 have no scheduler state; resuming them must
    estimate the position rather than replay warmup at full LR."""
    t = _stub_trainer(steps_per_epoch=10)
    legacy = {
        "model": t.model.state_dict(),
        "ema": None,
        "optimizer": t.optimizer.state_dict(),
        "epoch": 3,
        "cfg": t.cfg,
    }
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)

    fresh = _stub_trainer(steps_per_epoch=10)
    fresh.load_train_state(load_checkpoint(path))
    assert fresh.scheduler.it == 30  # 3 epochs * 10 steps
    assert fresh.start_epoch == 3
    assert fresh.optimizer.param_groups[0]["lr"] < 0.1  # past warmup, already decaying


def test_last_pt_carries_the_current_best_not_the_previous_one(tmp_path):
    """last.pt used to be written before best_metric was updated, so resuming from it
    restored a stale best and the next, worse epoch was accepted as the new best."""
    t = _stub_trainer()
    t.out_dir = tmp_path
    t.primary_metric = "traversability_mIoU"

    assert t.record_epoch(1, {"traversability_mIoU": 0.40}) is True
    assert load_checkpoint(tmp_path / "last.pt")["best_metric"] == 0.40

    assert t.record_epoch(2, {"traversability_mIoU": 0.30}) is False
    assert load_checkpoint(tmp_path / "last.pt")["epoch"] == 2
    assert load_checkpoint(tmp_path / "best.pt")["epoch"] == 1  # still the better one

    fresh = _stub_trainer()
    fresh.out_dir = tmp_path
    fresh.primary_metric = "traversability_mIoU"
    fresh.load_train_state(load_checkpoint(tmp_path / "last.pt"))
    assert fresh.record_epoch(3, {"traversability_mIoU": 0.35}) is False


def test_ema_progress_survives_a_resume(tmp_path):
    """The decay ramp is a function of the update count. Losing it restarts the ramp,
    dragging a mature average back towards whatever the model is at the resume point."""
    from syncai_hydranet.engine.ema import ModelEMA

    t = _stub_trainer()
    t.ema = ModelEMA(t.model, decay=0.99, warmup_steps=100)
    for _ in range(250):
        t.ema.update(t.model)

    path = tmp_path / "last.pt"
    torch.save(t.state_dict(epoch=1), path)
    ckpt = load_checkpoint(path)
    assert ckpt["ema_updates"] == 250

    fresh = _stub_trainer()
    fresh.ema = ModelEMA(fresh.model, decay=0.99, warmup_steps=100)
    fresh.ema.ema.load_state_dict(ckpt["ema"])
    fresh.load_train_state(ckpt)
    assert fresh.ema.updates == 250
    assert fresh.ema.decay_at(fresh.ema.updates + 1) == pytest.approx(
        t.ema.decay_at(t.ema.updates + 1)
    )


def test_best_metric_survives_so_best_pt_is_not_overwritten(tmp_path):
    """Without this, the first validation after a resume beats -1.0 and clobbers the
    best checkpoint with a worse model."""
    t = _stub_trainer()
    t.best_metric = 0.61
    path = tmp_path / "best.pt"
    torch.save(t.state_dict(epoch=9), path)

    fresh = _stub_trainer()
    fresh.load_train_state(load_checkpoint(path))
    assert fresh.best_metric == 0.61
    assert fresh.best_metric > 0.55  # a worse epoch must not win


# ------------------------------------------------- a write that dies mid-flight


def test_a_saved_checkpoint_reads_back(tmp_path):
    path = save_checkpoint({"model": {"w": torch.ones(3)}}, tmp_path / "last.pt")
    assert path == tmp_path / "last.pt"
    assert torch.equal(load_checkpoint(path)["model"]["w"], torch.ones(3))


def test_a_crash_mid_write_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """The whole point. `last.pt` is overwritten every epoch and runs here get preempted.

    Writing in place meant a SIGKILL inside `torch.save` left a truncated file where the
    only resume point had been -- the previous epoch's copy was already gone, because it
    was the same path.
    """
    path = tmp_path / "last.pt"
    save_checkpoint({"epoch": 1, "w": torch.ones(3)}, path)
    good = path.read_bytes()

    def _die(*_args, **_kwargs):
        raise KeyboardInterrupt("preempted mid-write")

    monkeypatch.setattr(torch, "save", _die)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint({"epoch": 2, "w": torch.zeros(3)}, path)

    assert path.read_bytes() == good, "the previous epoch's checkpoint was destroyed"
    assert load_checkpoint(path)["epoch"] == 1


def test_the_temporary_file_is_hidden_and_is_not_a_pt(tmp_path, monkeypatch):
    """`resolve_out_dir` decides a directory is occupied by globbing `*.pt`.

    A temporary called `last.pt.tmp` would not match that glob either, but it would show
    up in every listing of a run's checkpoints. Hidden and suffixed keeps it out of both.
    """
    seen = []
    real_save = torch.save

    def _spy(obj, f, *a, **k):
        seen.append(Path(f).name)
        return real_save(obj, f, *a, **k)

    monkeypatch.setattr(torch, "save", _spy)
    save_checkpoint({"epoch": 1}, tmp_path / "best.pt")

    assert seen == [".best.pt.tmp"]
    assert not list(tmp_path.glob("*.tmp")), "the temporary outlived the rename"
    assert [p.name for p in tmp_path.glob("*.pt")] == ["best.pt"]
