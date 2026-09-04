"""What the optimiser is, and how its learning rate moves.

Both were in `trainer.py`, and neither is about the training loop: one is a parameter-
group policy and the other is a schedule with a `state_dict`. `tests/test_overfit.py` and
`tests/test_checkpoint.py` were already importing them past `Trainer` to use them on
their own, which is the tell that they were in the wrong file.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, tcfg) -> torch.optim.Optimizer:
    """Lower LR for the backbone (transfer-learning convention); no weight decay on
    biases and norm parameters."""
    lr = float(tcfg["lr"])
    bb_mult = float(tcfg.get("backbone_lr_mult", 1.0))
    wd = float(tcfg.get("weight_decay", 0.05))
    decay, no_decay, bb_decay, bb_no_decay = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = name.startswith("backbone.")
        no_wd = p.ndim <= 1  # bias or norm
        target = (
            bb_no_decay
            if is_bb and no_wd
            else bb_decay
            if is_bb
            else no_decay
            if no_wd
            else decay
        )
        target.append(p)
    groups = [
        {"params": decay, "lr": lr, "weight_decay": wd},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": bb_decay, "lr": lr * bb_mult, "weight_decay": wd},
        {"params": bb_no_decay, "lr": lr * bb_mult, "weight_decay": 0.0},
    ]
    if tcfg.get("optimizer", "adamw") == "adamw":
        return torch.optim.AdamW(groups)
    return torch.optim.SGD(groups, momentum=0.9, nesterov=True)


class WarmupCosine:
    """Linear warmup into a cosine decay, applied to the optimizer's own param groups.

    **The schedule is applied at construction, and that line is the whole of a bug this
    class shipped with.** `Trainer.train_one_epoch` steps the optimizer *before* the
    scheduler, so whatever the param groups hold when the loop starts is what the first
    optimizer step runs at. Without the `_apply()` below they hold `base_lr` — measured
    on the shipped `warmup_iters: 500`, `lr: 2.0e-4` pairing, the first step ran at
    **500x** the intended warmup value, on freshly initialised heads, and AdamW's first
    moment then carried it forward. `load_state_dict` had always re-applied immediately
    and said why, so the resume path was right and the fresh-start path was not.

    The first step now runs at `_factor(0) == 0`: a warmup that starts at zero costs one
    no-op optimizer step and is the shape the rest of the schedule already assumed.
    Making it start at `1/warmup` instead would shift every later iteration by one and
    change what a restored `it` means, for one step of a run that takes thousands.
    """

    def __init__(self, optimizer, warmup_iters: int, total_iters: int):
        self.opt = optimizer
        self.warmup = max(warmup_iters, 1)
        self.total = total_iters
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.it = 0
        self._apply()

    def _factor(self, it: int) -> float:
        if it <= self.warmup:
            return it / self.warmup
        t = (it - self.warmup) / max(self.total - self.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    def _apply(self):
        f = self._factor(self.it)
        for g, base in zip(self.opt.param_groups, self.base_lrs, strict=True):
            g["lr"] = base * f

    def step(self):
        self.it += 1
        self._apply()

    def state_dict(self) -> dict:
        return {"it": self.it}

    def load_state_dict(self, state: dict) -> None:
        """Restore the schedule position and immediately re-apply it.

        Without the re-apply the first resumed step would run at the base LR, which for
        a run resumed near the end of cosine decay is orders of magnitude too high.
        """
        self.it = int(state["it"])
        self._apply()
