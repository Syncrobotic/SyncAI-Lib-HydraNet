"""Checkpoint loading.

``torch.load`` defaults to a full pickle load, which executes arbitrary code from the
file. Checkpoints get shared between machines and downloaded from release pages, so
every load in this package goes through :func:`load_checkpoint` instead.
"""

from __future__ import annotations

from pathlib import Path

import torch

CKPT_FORMAT = 2  # 1 = weights + optimizer only; 2 = full trainer state


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a HydraNet checkpoint with the restricted unpickler.

    Our checkpoints only ever hold tensors and plain YAML-derived containers, so
    ``weights_only=True`` is sufficient. A checkpoint that fails to load under it
    contains something it should not, and is exactly the case worth stopping on.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:  # the failure mode varies by torch version
        raise RuntimeError(
            f"could not safely load {path}: {e}\n"
            "It holds pickled Python objects beyond tensors and plain containers. "
            "Only load it with torch.load(..., weights_only=False) if you produced it "
            "yourself and trust its origin."
        ) from e
