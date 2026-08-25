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
        if k in ("boxes", "labels", "pose"):
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
        # Kept, not merely iterated: `detection_class_steps` below is the only place the
        # step share can be computed at all, because it needs both the schedule (here)
        # and what each dataset can label (there).
        self.datasets = list(datasets)
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

    def detection_class_steps(self) -> dict[str, tuple[int, int]]:
        """Per detection class: ``(steps that supervise it, total detection steps)``.

        The measurement `config_schema.minority_sourced_terrain_classes` says belongs
        here rather than in a config check, and says why: `sample_ratio` is in the config
        and the number of images behind it is not, so a share derived from ratios alone
        is an invented figure with a confident shape. Here both are known -- `self.steps`
        *is* the schedule -- so the number is counted rather than estimated.

        Empty unless some dataset carries a `det_vocab`. Without one there is a single
        detection source by construction (two would collide on label 0), so every class
        is at 100% and the table would say nothing.

        It reports the *supervised* share and not a negative share, and that is a
        consequence of the class mask rather than an omission: a dataset that cannot
        label a class now contributes no gradient to it instead of contributing a false
        negative. What remains worth reading is how thin a channel's supervision is --
        `person` trained on 10% of steps is a bootstrap, and the log line is where that
        becomes visible before the run rather than after it.
        """
        out: dict[str, tuple[int, int]] = {}
        det = [
            (ds, n)
            for ds, n in zip(self.datasets, self.steps, strict=True)
            if getattr(ds, "vocab", None) is not None
            and "detection" in getattr(ds, "supervises", [])
        ]
        total = sum(n for _ds, n in det)
        if not total:
            return out
        for ds, _n in det:
            for name in ds.vocab.classes:
                out.setdefault(name, (0, total))
        for ds, n in det:
            for label in ds.supplied_labels:
                name = ds.vocab.classes[label]
                out[name] = (out[name][0] + n, total)
        return out

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
