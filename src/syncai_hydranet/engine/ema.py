"""The weight average validation actually runs on.

Split out of `trainer.py` because it is a self-contained idea with its own tests, not a
part of the training loop: it takes a model, is told when a step happened, and holds a
second set of weights. `Trainer` is one caller of it.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn


class ModelEMA:
    """Exponential moving average of the weights, with a warmed-up decay.

    The average starts from the model's *initial* random weights. At a fixed decay of
    0.9998 that initialisation is still 97% of the EMA after 160 steps (0.9998^160;
    an earlier version of this sentence said 45%, which takes ~4,000 steps), and validation
    -- which runs on the EMA -- reported 0.16 mIoU for a model whose raw weights scored
    0.95. The failure is silent and looks exactly like a model that did not learn.

    The decay is therefore ramped: ``decay * (1 - exp(-updates / warmup_steps))``, so
    the first updates copy the model almost outright and the smoothing strengthens as
    the average acquires real history. This is the standard fix (YOLOv5, timm) and it
    makes EMA safe on short runs instead of merely warned about.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup_steps: int = 2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = max(int(warmup_steps), 0)
        self.updates = 0

    def decay_at(self, updates: int) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return self.decay * (1 - math.exp(-updates / self.warmup_steps))

    def residual_init_fraction(self, steps: int) -> float:
        """How much of the random initialisation survives after ``steps`` updates."""
        if steps <= 0:
            return 1.0
        if self.warmup_steps <= 0:
            return float(self.decay**steps)
        n = np.arange(1, steps + 1)
        decays = self.decay * (1 - np.exp(-n / self.warmup_steps))
        return float(np.exp(np.log(decays).sum()))

    @torch.no_grad()
    def update(self, model: nn.Module):
        """One EMA step, batched across tensors rather than looped over them.

        The obvious loop -- ``v.mul_(d).add_(msd[k], alpha=1-d)`` per entry -- issues two
        CUDA kernels per tensor, and this model's state_dict holds 504 floating-point
        ones. That is ~1,000 launches every optimizer step, for 8.3 M parameters of
        actual arithmetic. Measured on an RTX PRO 6000: **4.81 ms per update, against
        0.20 ms for the ``_foreach`` form** -- 8% of a 60 ms training step spent almost
        entirely in launch overhead, which is also why the trainer process sat at 114%
        CPU while neither the GPU nor the loader was saturated.

        ``torch._foreach_*`` is the same arithmetic on the same tensors, grouped into a
        handful of kernels. Verified bit-identical against the loop (max abs diff 0.0),
        which matters more here than the speed: EMA weights are what validation scores
        and what ships, so a "faster" update that drifted would be invisible.

        Integer buffers are copied outright -- num_batches_tracked has no meaningful
        average -- and kept out of the batched call, which requires a uniform dtype.
        """
        self.updates += 1
        d = self.decay_at(self.updates)
        msd = model.state_dict()
        ema_f, model_f = [], []
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                ema_f.append(v)
                model_f.append(msd[k].detach())
            else:
                v.copy_(msd[k])
        if ema_f:
            torch._foreach_mul_(ema_f, d)
            torch._foreach_add_(ema_f, model_f, alpha=1 - d)
