"""Seeding and backend flags.

``torch.manual_seed`` alone leaves two holes. Python's ``random`` drives every
augmentation in ``data/transforms.py``, and each DataLoader worker is a separate
process with its own interpreter state, so what a run actually sees depends on worker
count and start order unless the seeds are derived explicitly.

Full determinism is a separate, expensive request: it disables the cuDNN autotuner and
forces slower deterministic kernels. It belongs behind ``train.deterministic``, off by
default, for when a result has to be bit-reproducible rather than merely repeatable.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch in the current process."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, run-reproducible seed.

    torch seeds its own generator per worker, but the augmentations here use Python's
    ``random``, and NumPy is used in the mask paths. Deriving both from the worker's
    torch seed keeps them reproducible and, more importantly, keeps two workers from
    drawing the same augmentations.
    """
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % (2**32))


def loader_generator(seed: int) -> torch.Generator:
    """A generator for DataLoader shuffling, independent of global RNG consumption."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def configure_backends(
    device: torch.device,
    *,
    deterministic: bool = False,
    cudnn_benchmark: bool = True,
    tf32: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    """Set the CUDA/cuDNN knobs that trade reproducibility for speed.

    Input size is fixed for the whole run, which is exactly the case the cuDNN autotuner
    is built for, so benchmarking is on by default. TF32 costs a little mantissa on
    matmuls for a large speedup on Ampere and later, and is standard for training at
    this precision.
    """
    log = logger.info if logger else (lambda _m: None)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log("deterministic mode: cuDNN autotuning off, deterministic kernels forced")
        return

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        log(f"cuda backends: cudnn_benchmark={cudnn_benchmark}, tf32={tf32}")


def apply_channels_last(
    model: torch.nn.Module,
    enabled: bool,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Move 4-D weights to NHWC, which is the layout tensor cores read natively.

    It removes the implicit transposes cuDNN would otherwise insert around every
    convolution. Measured on an RTX PRO 6000 at batch 48, 512x640, bf16 autocast:
    153.2 -> 121.1 ms, about -21% (`scripts/bench_channels_last.py`). Under fp32 it is
    worth nothing -- 219.6 -> 220.9 ms -- because those convolutions run on CUDA cores,
    which do not care about layout. The saving is entirely in work not done, which is
    why high GPU utilisation beforehand did not rule it out: transposing counts as
    utilisation.

    It is the strides that change, **not only the strides**: NHWC selects different
    convolution kernels, which accumulate in a different order. Under the TF32 every run
    here enables, the two layouts' outputs differ by ~1e-2 of scale (1e-5 with TF32 off);
    `tests/test_channels_last.py` pins both. So this defaults to off -- a run with the
    flag flipped partway through is not bit-comparable to itself, and a control that
    exists to isolate one variable must not collect a second one from a restart.
    """
    if not enabled:
        return
    model.to(memory_format=torch.channels_last)
    if logger:
        logger.info("memory format: channels_last (NHWC)")


def model_memory_format(model: torch.nn.Module) -> torch.memory_format:
    """The layout a model's convolutions want, read off its own weights.

    Inference and evaluation get this without the training flag threaded through to
    them: a checkpoint's memory format is not in its state_dict, so asking the live
    module is the only reliable source.

    A weight is only allowed to vote if it distinguishes the two formats. A 1x1 conv
    weight satisfies both `is_contiguous()` and `is_contiguous(channels_last)`, and
    `is_contiguous()` wins that tie by convention -- so a model whose first 4-D
    parameter is a 1x1 conv would otherwise report NCHW however it was converted.

    Anything without `parameters()` -- the evaluator's test stubs are the live case --
    has no weights to have a preference, and NCHW is the answer that costs nothing:
    `contiguous(contiguous_format)` on an already-contiguous input is a no-op. Kept
    duck-typed rather than requiring nn.Module because `evaluate` is, and widening its
    contract for a layout hint would be the tail wagging the dog.
    """
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return torch.contiguous_format
    for p in parameters():
        if (
            p.dim() == 4
            and p.is_contiguous(memory_format=torch.channels_last)
            and not p.is_contiguous()
        ):
            return torch.channels_last
    return torch.contiguous_format


def needs_grad_scaler(amp: bool, dtype: torch.dtype) -> bool:
    """Loss scaling is an fp16 workaround, not a mixed-precision requirement.

    fp16's exponent range is narrow enough that small gradients flush to zero, which is
    what the scaler exists to prevent. bfloat16 keeps fp32's exponent range, so scaling
    would only add overhead and an extra failure mode.
    """
    return amp and dtype is torch.float16


def resolve_amp_dtype(name: str) -> torch.dtype:
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in dtypes:
        raise ValueError(f"unknown amp_dtype {name!r}, expected one of {', '.join(dtypes)}")
    return dtypes[name]
