"""The EMA weights are what validation scores, so their warm-up behaviour is a
correctness concern, not a tuning detail.

pytest tests/test_ema.py -v
"""

import math

import pytest
import torch
import torch.nn as nn

from syncai_hydranet.engine.trainer import ModelEMA

DECAY = 0.9998
WARMUP = 2000


def _model(value: float):
    m = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        m.weight.fill_(value)
    return m


def _weight(m):
    return float(m.weight.flatten()[0])


# ------------------------------------------------------------------ the ramp


def test_decay_starts_low_and_approaches_the_target():
    ema = ModelEMA(_model(0.0), DECAY, WARMUP)
    assert ema.decay_at(1) < 0.01  # first update is nearly a straight copy
    assert ema.decay_at(WARMUP) == pytest.approx(DECAY * (1 - 1 / math.e), rel=1e-3)
    assert ema.decay_at(10 * WARMUP) == pytest.approx(DECAY, rel=1e-4)


def test_first_update_nearly_adopts_the_model():
    """The whole failure this fixes: with a flat 0.9998 the average stays at its random
    initialisation for thousands of steps while validation reports on it."""
    ema = ModelEMA(_model(0.0), DECAY, WARMUP)
    ema.update(_model(1.0))
    assert _weight(ema.ema) > 0.99


def test_a_flat_decay_would_not():
    """Contrast case, so the test above is not just asserting arithmetic."""
    ema = ModelEMA(_model(0.0), DECAY, warmup_steps=0)
    ema.update(_model(1.0))
    assert _weight(ema.ema) < 0.001


def test_short_run_is_no_longer_dominated_by_the_initialisation():
    """160 steps used to leave 45% of the random initialisation in the EMA."""
    ema = ModelEMA(_model(0.0), DECAY, WARMUP)
    assert ema.residual_init_fraction(160) < 0.01
    assert ModelEMA(_model(0.0), DECAY, 0).residual_init_fraction(160) > 0.95


def test_residual_fraction_is_monotonic_and_bounded():
    ema = ModelEMA(_model(0.0), DECAY, WARMUP)
    assert ema.residual_init_fraction(0) == 1.0
    values = [ema.residual_init_fraction(n) for n in (1, 10, 100, 1000)]
    assert values == sorted(values, reverse=True)
    assert values[-1] >= 0.0


# ----------------------------------------------------------------- smoothing


def test_it_still_smooths_once_warm():
    """The ramp must not turn the EMA into a plain copy of the model."""
    ema = ModelEMA(_model(0.0), DECAY, warmup_steps=10)
    steady = _model(1.0)
    for _ in range(500):
        ema.update(steady)
    assert _weight(ema.ema) == pytest.approx(1.0, abs=1e-3)

    ema.update(_model(0.0))  # one outlier step
    assert _weight(ema.ema) > 0.99  # barely moves


def test_update_count_tracks_calls():
    ema = ModelEMA(_model(0.0), DECAY, WARMUP)
    for _ in range(7):
        ema.update(_model(1.0))
    assert ema.updates == 7


def test_integer_buffers_are_copied_not_averaged():
    """num_batches_tracked and friends are counters; averaging them is meaningless."""
    model = nn.BatchNorm2d(3)
    ema = ModelEMA(model, DECAY, WARMUP)
    model.num_batches_tracked += 41
    ema.update(model)
    assert int(ema.ema.num_batches_tracked) == int(model.num_batches_tracked)
