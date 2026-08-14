"""Gradient accumulation and seeding.

Accumulation is only worth having if N micro-batches produce the gradient one batch of
N times the size would have. That is an arithmetic claim, so it gets an arithmetic test
rather than a "does it run" test.

pytest tests/test_accumulation.py -v
"""

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from syncai_hydranet.utils.seeding import (
    loader_generator,
    needs_grad_scaler,
    resolve_amp_dtype,
    seed_everything,
    worker_init_fn,
)

ACCUM = 4
MICRO = 2


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3))


def _data():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(ACCUM * MICRO, 6, generator=g)
    y = torch.randint(0, 3, (ACCUM * MICRO,), generator=g)
    return x, y


def _grads(model):
    return [p.grad.clone() for p in model.parameters()]


# ------------------------------------------------------------- accumulation


def test_accumulated_gradient_equals_the_full_batch_gradient():
    x, y = _data()

    full = _model()
    full.zero_grad()
    nn.functional.cross_entropy(full(x), y).backward()
    expected = _grads(full)

    accumulated = _model()
    accumulated.zero_grad()
    for i in range(ACCUM):
        xs = x[i * MICRO : (i + 1) * MICRO]
        ys = y[i * MICRO : (i + 1) * MICRO]
        # The division the trainer applies: without it the gradient is ACCUM times too
        # large and the effective learning rate silently scales with accumulation.
        (nn.functional.cross_entropy(accumulated(xs), ys) / ACCUM).backward()

    for got, want in zip(_grads(accumulated), expected, strict=True):
        assert torch.allclose(got, want, atol=1e-6)


def test_forgetting_the_division_is_detectable():
    """Guards the test above: if it passed either way it would be worthless."""
    x, y = _data()
    model = _model()
    model.zero_grad()
    for i in range(ACCUM):
        xs, ys = x[i * MICRO : (i + 1) * MICRO], y[i * MICRO : (i + 1) * MICRO]
        nn.functional.cross_entropy(model(xs), ys).backward()
    undivided = _grads(model)

    full = _model()
    full.zero_grad()
    nn.functional.cross_entropy(full(x), y).backward()
    assert not torch.allclose(undivided[0], _grads(full)[0], atol=1e-6)


# ------------------------------------------------------------------ seeding


def test_seed_everything_covers_python_numpy_and_torch():
    """Augmentations use Python's random and the mask paths use NumPy, so seeding torch
    alone leaves most of the pipeline unseeded."""
    seed_everything(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(123)
    assert first == (random.random(), float(np.random.rand()), float(torch.rand(1)))


def test_loader_generators_are_independent_of_global_rng():
    """Sample order must not depend on how much other code drew from torch first."""
    torch.manual_seed(0)
    a = torch.randperm(20, generator=loader_generator(7))
    torch.rand(1000)  # unrelated consumption
    b = torch.randperm(20, generator=loader_generator(7))
    assert torch.equal(a, b)


def test_different_seeds_give_different_order():
    a = torch.randperm(20, generator=loader_generator(1))
    b = torch.randperm(20, generator=loader_generator(2))
    assert not torch.equal(a, b)


def test_workers_do_not_share_a_seed(monkeypatch):
    """Two workers drawing the same augmentations halves the effective variety."""
    monkeypatch.setattr(torch, "initial_seed", lambda: 999)
    worker_init_fn(0)
    first = (random.random(), float(np.random.rand()))
    worker_init_fn(1)
    assert first != (random.random(), float(np.random.rand()))

    worker_init_fn(0)
    assert first == (random.random(), float(np.random.rand()))  # still reproducible


# -------------------------------------------------------------------- dtype


def test_amp_dtypes():
    assert resolve_amp_dtype("float16") is torch.float16
    assert resolve_amp_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(ValueError, match="unknown amp_dtype"):
        resolve_amp_dtype("fp16")


def test_grad_scaler_is_only_for_float16():
    """Scaling exists to stop fp16 gradients flushing to zero; bfloat16 keeps fp32's
    exponent range and needs none of it."""
    assert needs_grad_scaler(True, torch.float16) is True
    assert needs_grad_scaler(True, torch.bfloat16) is False
    assert needs_grad_scaler(False, torch.float16) is False
