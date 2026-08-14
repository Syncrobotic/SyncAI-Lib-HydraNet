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

from ..utils.seeding import loader_generator, worker_init_fn


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
    if "geom" in batch[0]:
        out["geoms"] = [b["geom"] for b in batch]
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
                # Shuffling and augmentation are seeded per loader rather than left to
                # the global RNG, so the sample order does not depend on how much other
                # code happened to draw from torch beforehand. Offsetting by index keeps
                # two datasets from marching in lockstep.
                generator=loader_generator(seed + i),
                worker_init_fn=worker_init_fn,
            )
            for i, ds in enumerate(datasets)
        ]
        self.ratios = ratios
        self.rng = random.Random(seed)
        self.steps = [
            int(len(loader) * r) for loader, r in zip(self.loaders, ratios, strict=True)
        ]
        # A dataset whose share rounds to zero is never sampled, so the heads it
        # supervises are trained on nothing -- while the config still lists it, and the
        # export guard (which reads the config, not the schedule) still lets the model
        # ship. The head then emits confident output from initial weights. Refusing here
        # is the only place that can tell the difference.
        starved = [
            f"{name} (ratio {r} x {len(loader)} batches -> 0)"
            for name, loader, r, n in zip(names, self.loaders, ratios, self.steps, strict=True)
            if n == 0
        ]
        if starved:
            raise ValueError(
                "these datasets would contribute no batches at all: "
                + "; ".join(starved)
                + ". Raise sample_ratio, or remove the dataset -- but then the heads it "
                "supervises are unsupervised, and export will refuse the model."
            )
        self.steps_per_epoch = sum(self.steps)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[dict]:
        iters = [iter(loader) for loader in self.loaders]
        schedule = []
        for i, n in enumerate(self.steps):
            schedule += [i] * n
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
