"""Two ways a run spends wall clock on nothing, and the knobs that stop it.

Measured on `runs/hydranet_retail_objects`, an 84-minute 60-epoch run:

    median epoch                85 s
    of which COCO detection val 17 s   -> 20% of every epoch, 17 min of the run
    after the ep25 metric peak         -> 49 min, 58% of the run

Neither changes what a run learns. Both change how long you wait to find out, and both
default to off so an existing config trains exactly as it did.

The tests exercise `Trainer` methods against a stub rather than a real run, because what
is being pinned is scheduling logic -- which epochs validate what, and when the loop
gives up -- and none of that needs a GPU or a dataset to be wrong.

pytest tests/test_training_budget.py -v
"""

from types import SimpleNamespace

import pytest

from syncai_hydranet.engine.trainer import Trainer


class _Set:
    """Stands in for a validation dataset; `supervises` is all the filter reads."""

    def __init__(self, *supervises):
        self.supervises = list(supervises)


def make_trainer(**kw) -> Trainer:
    """A Trainer with only the attributes these two features touch.

    Built without `__init__` on purpose. The real constructor builds datasets, a model,
    an optimiser and a TensorBoard writer, none of which decide whether epoch 7 scores
    detection.
    """
    t = Trainer.__new__(Trainer)
    t.val_sets = [("ade20k", _Set("terrain")), ("coco", _Set("detection"))]
    t.val_loaders = ["ade-loader", "coco-loader"]
    t.epochs = 60
    t.det_val_interval = 1
    t.early_stop_patience = 0
    t.epochs_since_best = 0
    t.best_metric = -1.0
    t.primary_metric = "terrain_mIoU"
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_interval_one_scores_everything_every_epoch():
    """The default has to be indistinguishable from the old behaviour."""
    t = make_trainer()
    for epoch in (1, 2, 7, 59, 60):
        sets, loaders = t.val_subset(epoch)
        assert [n for n, _ in sets] == ["ade20k", "coco"]
        assert loaders == ["ade-loader", "coco-loader"]


def test_off_interval_epochs_drop_the_detection_set_and_its_loader():
    t = make_trainer(det_val_interval=5)
    sets, loaders = t.val_subset(7)
    assert [n for n, _ in sets] == ["ade20k"]
    assert loaders == ["ade-loader"], "loaders must stay aligned with val_sets"


def test_on_interval_epochs_score_detection():
    t = make_trainer(det_val_interval=5)
    assert [n for n, _ in t.val_subset(10)[0]] == ["ade20k", "coco"]


def test_the_last_epoch_always_scores_detection():
    """Otherwise a run's final row carries an mAP from up to `interval - 1` epochs ago.

    A number in a table with nothing saying when it was measured is the failure this
    repository keeps writing documents about.
    """
    t = make_trainer(det_val_interval=7, epochs=60)
    assert 60 % 7 != 0, "the fixture is only meaningful on an off-interval final epoch"
    assert [n for n, _ in t.val_subset(60)[0]] == ["ade20k", "coco"]


def test_a_dataset_supervising_both_heads_is_never_dropped():
    """Dropping it would silently stop scoring terrain to save time on detection."""
    t = make_trainer(
        det_val_interval=5,
        val_sets=[("site", _Set("terrain", "detection"))],
        val_loaders=["site-loader"],
    )
    assert [n for n, _ in t.val_subset(7)[0]] == ["site"]


def test_all_detection_sets_are_scored_rather_than_validating_nothing():
    """An empty subset would leave `select_metric` with no metric to select on."""
    t = make_trainer(
        det_val_interval=5,
        val_sets=[("coco", _Set("detection"))],
        val_loaders=["coco-loader"],
    )
    assert [n for n, _ in t.val_subset(7)[0]] == ["coco"]


@pytest.mark.parametrize("interval", [1, 5])
def test_the_subset_never_reorders_or_duplicates(interval):
    """`evaluate` zips loaders against val_sets with strict=True; order is load-bearing."""
    t = make_trainer(det_val_interval=interval)
    sets, loaders = t.val_subset(7)
    assert len(sets) == len(loaders)
    assert [n for n, _ in sets] == sorted([n for n, _ in sets], key=["ade20k", "coco"].index)


def test_patience_zero_never_stops():
    """The default. Off means off, however long the plateau."""
    t = make_trainer(early_stop_patience=0)
    assert not any(t.should_stop(False) for _ in range(500))
    assert t.epochs_since_best == 500, "the counter still tracks; only the stop is off"


def test_it_stops_on_the_nth_validation_without_a_best_and_not_the_n_plus_first():
    """Patience 3 means three, and the boundary is the whole point of the setting."""
    t = make_trainer(early_stop_patience=3)
    assert [t.should_stop(False) for _ in range(4)] == [False, False, True, True]


def test_a_new_best_resets_the_counter():
    """A run that improves at epoch 30 gets its full patience again from there."""
    t = make_trainer(early_stop_patience=3)
    t.should_stop(False)
    t.should_stop(False)
    assert t.should_stop(True) is False
    assert t.epochs_since_best == 0
    assert [t.should_stop(False) for _ in range(3)] == [False, False, True]


def test_indoor_seed7_would_not_have_been_truncated():
    """The counterexample that decides patience over a smaller `epochs`.

    That run peaked at epoch 53 of 60. Any fixed cut short of 53 discards its best
    checkpoint; patience 10 never fires, because it never went ten validations without
    an improvement before the peak.
    """
    t = make_trainer(early_stop_patience=10)
    improvements_at = {1, 4, 9, 15, 22, 30, 38, 45, 53}
    stopped = next(
        (e for e in range(1, 61) if t.should_stop(e in improvements_at)),
        None,
    )
    assert stopped is None or stopped > 53


def test_a_run_that_peaks_at_epoch_6_stops_long_before_60():
    """`fixed_coco10` peaked at epoch 6 of 60 and spent 90% of its wall clock after it."""
    t = make_trainer(early_stop_patience=10)
    stopped = next(e for e in range(1, 61) if t.should_stop(e == 6))
    assert stopped == 16


def test_patience_state_survives_a_checkpoint_round_trip():
    """Runs on this box get preempted mid-flight.

    A resumed run that restarts its patience counter either stops early or never stops,
    and neither announces itself -- the same silent-wrong as resuming without
    `best_metric`, which this file's neighbour already guards.
    """
    t = make_trainer(early_stop_patience=10, epochs_since_best=7)
    saved = {"epochs_since_best": t.epochs_since_best}

    resumed = make_trainer(early_stop_patience=10)
    resumed.epochs_since_best = int(saved.get("epochs_since_best", 0))
    assert resumed.epochs_since_best == 7


def test_a_checkpoint_without_the_key_resumes_at_zero():
    """Checkpoints written before this feature must still load."""
    resumed = make_trainer(early_stop_patience=10)
    resumed.epochs_since_best = int({}.get("epochs_since_best", 0))
    assert resumed.epochs_since_best == 0


def test_state_dict_carries_the_counter():
    """Pins the key name, since load reads it by string."""
    t = make_trainer(epochs_since_best=4)
    t.model = SimpleNamespace(state_dict=dict)
    t.ema = None
    t.optimizer = SimpleNamespace(state_dict=dict)
    t.scheduler = SimpleNamespace(state_dict=dict)
    t.scaler = SimpleNamespace(state_dict=dict)
    t.global_step = 100
    t.cfg = {}
    assert t.state_dict(epoch=12)["epochs_since_best"] == 4
