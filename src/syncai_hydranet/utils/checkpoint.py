"""Reading and writing checkpoints.

``torch.load`` unpickles, and unpickling executes arbitrary code from the file. Its
``weights_only`` default was False through torch 2.5 and True from 2.6, and this project
floors at ``torch>=2.1`` -- so a bare ``torch.load`` is not "safe now", it is "safe on
whichever torch the reader happens to have installed". Checkpoints get shared between
machines and downloaded from release pages, so every load goes through
:func:`load_checkpoint`, which pins the restricted unpickler and says what to do when a
file genuinely needs more.

That was the stated rule and only ``src/`` followed it. ``scripts/offline_tracks.py``
passed ``weights_only=False`` outright -- on a checkpoint holding
``{"model": state_dict, "attributes": [str, ...]}``, which the restricted unpickler reads
without complaint, so the escape hatch bought nothing and cost the guarantee.
``tests/test_checkpoint_loading_policy.py`` now greps for it, over ``src/`` and
``scripts/`` alike.

Writing has a rule too, and it is not about pickles: see :func:`save_checkpoint`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

CKPT_FORMAT = 2  # 1 = weights + optimizer only; 2 = full trainer state


def save_checkpoint(obj: Any, path: str | Path) -> Path:
    """Write a checkpoint through a temporary file, so a crash cannot destroy the old one.

    ``torch.save`` straight to the destination is not atomic, and ``last.pt`` is
    overwritten every epoch. A 128 MB write interrupted halfway leaves a truncated file
    where the only resume point used to be -- and the previous epoch's copy is already
    gone, because it was the same path. `load_checkpoint` then refuses it correctly, by
    which point a multi-day run has nowhere to restart from.

    This is not a hypothetical on this box. `Trainer.state_dict` carries
    ``epochs_since_best`` specifically because "runs on this box get preempted", and a
    preemption is a SIGKILL that can land in the middle of a write as easily as anywhere
    else.

    Writing to a sibling and renaming makes the swap atomic: ``Path.replace`` is
    ``rename(2)``, which either happened or did not, and the loser is the temporary
    file. The temporary name is hidden and does not end in ``.pt``, which keeps it out
    of both ``resolve_out_dir``'s ``*.pt`` occupancy glob and any tooling that lists a
    run's checkpoints.

    **What this does not cover.** A rename is atomic with respect to *this process
    dying*; it is not a durability barrier against the *machine* dying, because the
    temporary file's bytes may still be in the page cache. That would need an ``fsync``
    of the file and of the directory, at the cost of flushing 128 MB to disk every epoch.
    The failure this box actually sees is preemption, which the page cache survives.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
    return path


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


def select_weights(ckpt: dict, prefer: str = "ema") -> dict:
    """Choose between a checkpoint's EMA and raw weights.

    Two rules used to coexist. hydranet-eval and the ONNX export honoured a --weights
    flag; hydranet-infer-image, hydranet-infer-video, hydranet-scene and annotation_batch
    hardcoded `ckpt.get("ema") or ckpt["model"]`, always taking the average with no way
    to ask for anything else.

    That is not a style difference. `git show b7457c2:docs/TRAINING_GUIDE.md` §5 recorded a run
    scoring 0.16 mIoU on EMA weights and 0.95 on the raw ones from the same training,
    because the average starts at the random initialisation and takes thousands of
    steps to forget it. On a short run the two sets of weights are different models,
    and the tools that render pictures were the ones with no way to say which.
    """
    if prefer not in ("ema", "model"):
        raise ValueError(f"prefer must be 'ema' or 'model', not {prefer!r}")
    if prefer == "ema" and ckpt.get("ema"):
        return ckpt["ema"]
    return ckpt["model"]
