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
