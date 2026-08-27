"""A dense metric-depth head, and the loss that makes it metric rather than merely shaped.

The third head family the registry's docstring anticipated. It was built for the
quadruped's occupancy direction, whose occupancy and ground-height heads both needed depth
before they needed anything else. That line was removed on 2026-08-19 (`eca3814`) and this
head stayed, because the CCTV side has its own depth experiment:
`configs/hydranet_hm3d_cctv.yaml` builds this head and nothing else.

It exists at all because measuring a public teacher settled that we cannot simply borrow
one: Depth-Anything V2 Metric-Indoor, scored zero-shot on NYUv2, over-predicts by a flat
15%. Its *geometry* is sound -- delta1 goes 0.687 -> 0.919 once a single global factor is
removed -- so what a head of our own has to supply is the part that did not transfer: the
metres. `data/nyu_depth.py` carries that factor and its full derivation.

---------------------------------------------------------------------------
WHY THIS REUSES THE SEGMENTATION HEAD INSTEAD OF BEING ONE

`SemanticFPNHead` already does exactly the required shape: fuse P3..P5, one 1x1
classifier, bilinear back to input resolution. Depth differs from segmentation in the
channel count (one, not C) and in what that channel *means*, not in how the pixels are
produced. So the module is a thin wrapper rather than a copy -- a copied decoder is a
second place to fix every time the neck changes.

**It predicts log-depth and exponentiates.** Two reasons, and neither is cosmetic. A raw
linear output can go negative, and a negative depth is not a large error, it is a
`log()` of a negative number inside the loss on the first bad step. And depth error is
naturally multiplicative -- 10 cm at 0.4 m and 10 cm at 8 m are not the same mistake --
so a network regressing in log space is regressing in the units the metric is scored in.

---------------------------------------------------------------------------
THE INVALID-PIXEL SENTINEL IS 0.0, NOT 255

Segmentation heads in this repo mask with `labels.IGNORE` = 255, and `config_schema`
now *raises* if a head declares any other `ignore_index`. None of that applies here and
reaching for it would be a real bug: 255 is the mask-file contract for an 8-bit
single-channel class id, while depth is continuous metres, and a sensor's no-return is
0.0 -- which in a depth map is also a perfectly well-formed number. So validity is a
predicate (`0 < d <= max_depth`), computed per batch, and never a magic value.

NYUv2's Kinect writes 0.0 where it got nothing back, which is most of the frame edge and
every specular surface. Roughly a third of a raw frame can be invalid; a loss that
averaged over it would be training on the mask, not the scene.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .segmentation import SemanticFPNHead


class DepthFPNHead(nn.Module):
    """Semantic-FPN trunk with a single log-depth channel.

    `max_depth` clamps the exponential rather than scaling a sigmoid. A sigmoid would make
    the head's usable range a training-time constant baked into the weights -- change it
    later and every checkpoint means something different. A clamp is applied at the same
    point on the same values but leaves the parameters alone, so raising the ceiling for
    an outdoor run does not invalidate the indoor ones.
    """

    def __init__(
        self,
        in_channels: int,
        in_levels=(0, 1, 2),
        channels: int = 64,
        dropout: float = 0.1,
        max_depth: float = 10.0,
    ):
        super().__init__()
        self.trunk = SemanticFPNHead(
            in_channels=in_channels,
            num_classes=1,
            in_levels=in_levels,
            channels=channels,
            dropout=dropout,
        )
        self.max_depth = float(max_depth)

    def forward(self, feats: list[torch.Tensor], out_size: tuple[int, int]) -> torch.Tensor:
        log_d = self.trunk(feats, out_size)  # [B, 1, H, W], unbounded
        # Clamped in log space before the exponential, not after: exp() of a large
        # positive activation overflows to inf in fp16 long before any clamp on the
        # result could see it, and an inf in the loss poisons the whole step.
        return torch.exp(
            log_d.clamp(min=-7.0, max=float(torch.log(torch.tensor(self.max_depth))))
        )


class SILogLoss(nn.Module):
    """Scale-invariant log loss, deliberately run at less than full scale invariance.

    The standard depth loss is SILog: the variance of the log-error, plus `lambda` times
    the square of its mean. At `lambda = 1` it is fully scale-invariant -- a model that
    gets every ratio right and every metre wrong scores perfectly. That is precisely the
    failure we just measured in the public teacher, so training our own head under a loss
    blind to it would reproduce the problem on purpose.

    `lambda = 0.85` is the usual compromise and the one used here: it still lets the model
    prioritise structure early, while leaving 15% of the objective anchored to absolute
    scale. `alpha` is the conventional output scaling; it changes the loss magnitude, not
    its minimum, and is kept only so the number is comparable to published runs.
    """

    def __init__(self, lam: float = 0.85, alpha: float = 10.0, max_depth: float = 10.0):
        super().__init__()
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.max_depth = float(max_depth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        valid = (target > 0) & (target <= self.max_depth) & torch.isfinite(target)
        if not bool(valid.any()):
            # A batch with no returns at all is a data problem, not a training signal.
            # Zero *through the graph* rather than a fresh scalar, so the step still has
            # a gradient path and DDP does not deadlock waiting for a rank that skipped.
            return pred.sum() * 0.0
        d = torch.log(pred[valid].clamp(min=1e-3)) - torch.log(target[valid].clamp(min=1e-3))
        return self.alpha * torch.sqrt((d**2).mean() - self.lam * d.mean() ** 2 + 1e-8)


def build_depth_head(cfg, in_channels: int) -> DepthFPNHead:
    return DepthFPNHead(
        in_channels=in_channels,
        in_levels=cfg.get("in_levels", [0, 1, 2]),
        channels=cfg.get("channels", 64),
        dropout=cfg.get("dropout", 0.1),
        max_depth=cfg.get("max_depth", 10.0),
    )
