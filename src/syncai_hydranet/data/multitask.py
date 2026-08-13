"""Multi-dataset, multi-task loading.

Each step draws one full batch from a single dataset (round-robin, weighted by
sample_ratio) and only the heads that dataset supervises contribute to the loss.

Why not mix datasets within a batch: the effective batch size per head would fluctuate,
making loss scales jitter, and the trunk's BatchNorm statistics would swing between two
very different image distributions. A single-source batch removes both problems.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader


def collate(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    supervises = batch[0]["supervises"]
    targets: dict = {}
    keys = batch[0]["targets"].keys()
    for k in keys:
        vals = [b["targets"][k] for b in batch]
        if k in ("boxes", "labels"):
            targets[k] = vals  # variable length, keep as a list
        else:
            targets[k] = torch.stack(vals)  # segmentation masks
    out = {"image": images, "targets": targets, "supervises": supervises}
    if "image_id" in batch[0]:
        out["image_ids"] = [b["image_id"] for b in batch]
    return out


class MultiTaskLoader:
    """Combine several DataLoaders into one iterator.

    Steps per epoch is ``sum(len(loader_i) * ratio_i)``. The per-epoch schedule is built
    up front and shuffled, so the sampling ratio is exact rather than approximate.
    """

    def __init__(
        self,
        datasets: list,
        names: list[str],
        ratios: list[float],
        batch_size: int,
        workers: int,
        seed: int = 0,
        pin_memory: bool = True,
    ):
        self.names = names
        self.loaders = [
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=workers,
                collate_fn=collate,
                pin_memory=pin_memory,
                persistent_workers=workers > 0,
            )
            for ds in datasets
        ]
        self.ratios = ratios
        self.rng = random.Random(seed)
        self.steps_per_epoch = sum(
            int(len(loader) * r) for loader, r in zip(self.loaders, ratios, strict=True)
        )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[dict]:
        iters = [iter(loader) for loader in self.loaders]
        schedule = []
        for i, (loader, r) in enumerate(zip(self.loaders, self.ratios, strict=True)):
            schedule += [i] * int(len(loader) * r)
        self.rng.shuffle(schedule)
        for i in schedule:
            try:
                batch = next(iters[i])
            except StopIteration:
                # Ratios below 1 truncate an epoch; a fresh iterator reshuffles, so
                # successive epochs see different subsets.
                iters[i] = iter(self.loaders[i])
                batch = next(iters[i])
            batch["dataset"] = self.names[i]
            yield batch
