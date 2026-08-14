"""A dataset that never gets sampled is worse than a dataset that is absent.

`sample_ratio` multiplies a loader's batch count and the product is truncated to an int.
Set it low enough -- or point it at a small dataset -- and a dataset contributes **zero**
batches per epoch. Training then runs happily, the config still lists the dataset, and
the export guard (which reads the config, not the schedule) still lets the model ship
with a head that was trained on nothing but its initialisation.

Nothing else in the pipeline can tell the difference, so the loader refuses.

pytest tests/test_multitask_ratio.py -v
"""

import pytest
import torch
from torch.utils.data import Dataset

from syncai_hydranet.data.multitask import MultiTaskLoader


class Tiny(Dataset):
    """Minimal seg-shaped dataset: MultiTaskLoader only needs len() and collatable items."""

    def __init__(self, n: int, supervises=("terrain",)):
        self.n = n
        self.supervises = list(supervises)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "image": torch.zeros(3, 8, 8),
            "targets": {"terrain": torch.zeros(8, 8, dtype=torch.long)},
            "supervises": self.supervises,
        }


def _loader(sizes, ratios, batch_size=2):
    return MultiTaskLoader(
        [Tiny(n) for n in sizes],
        [f"ds{i}" for i in range(len(sizes))],
        ratios,
        batch_size=batch_size,
        workers=0,
        seed=0,
    )


def test_ratios_are_exact_not_approximate():
    """10 batches at 1.0 plus 10 at 0.5 is 15 steps, every epoch, not on average."""
    loader = _loader([20, 20], [1.0, 0.5])
    assert len(loader) == 15
    assert loader.steps == [10, 5]


def test_a_ratio_that_rounds_to_zero_is_refused():
    """0.05 x 10 batches = 0. The head this dataset supervises would train on nothing."""
    with pytest.raises(ValueError, match="no batches at all"):
        _loader([20, 20], [1.0, 0.05])


def test_the_message_names_the_dataset_and_the_arithmetic():
    """A refusal that does not say which knob to turn just moves the debugging."""
    with pytest.raises(ValueError) as exc:
        _loader([20, 20], [1.0, 0.05])
    msg = str(exc.value)
    assert "ds1" in msg and "0.05" in msg and "sample_ratio" in msg


def test_a_dataset_too_small_for_one_batch_is_refused():
    """The same failure arrives by a different route: drop_last leaves zero batches."""
    with pytest.raises(ValueError, match="no batches at all"):
        _loader([20, 1], [1.0, 1.0], batch_size=2)


def test_every_scheduled_batch_is_delivered_and_tagged():
    loader = _loader([20, 20], [1.0, 0.5])
    seen = [b["dataset"] for b in loader]
    assert len(seen) == len(loader)
    assert seen.count("ds0") == 10
    assert seen.count("ds1") == 5
