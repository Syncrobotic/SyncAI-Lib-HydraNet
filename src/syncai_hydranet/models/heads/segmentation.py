"""Semantic-FPN segmentation head, shared by the traversability and terrain heads.

P3..P5 each pass through a few convolutions, are upsampled to P3 resolution and summed,
then a 1x1 classifier produces logits which are bilinearly resized back to the input
resolution. No decoder and no dilated convolutions: multi-scale fusion already happened
in the neck, so a U-Net style decoder here would be duplicated work costing more
parameters than the entire neck.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..neck import conv_bn_act


class SemanticFPNHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        in_levels: Sequence[int] = (0, 1, 2),
        channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_levels = list(in_levels)
        self.scale_convs = nn.ModuleList()
        for lv in self.in_levels:
            n_up = lv - self.in_levels[0]  # upsampling factor relative to P3
            blocks = [conv_bn_act(in_channels, channels)]
            for _ in range(n_up):
                blocks.append(conv_bn_act(channels, channels))
            self.scale_convs.append(nn.Sequential(*blocks))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(channels, num_classes, 1)
        self.num_classes = num_classes

    def fuse(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """The shared trunk: every input level brought to P3 and summed.

        Split out of `forward` so `SegDepthFPNHead` can hang a second predictor on the
        same fused map. Running the trunk twice would be two towers wearing one name --
        the whole point of a joint head is that the features are shared.
        """
        base = feats[self.in_levels[0]]
        target = base.shape[-2:]
        fused = None
        for conv, lv in zip(self.scale_convs, self.in_levels, strict=True):
            x = conv(feats[lv])
            if x.shape[-2:] != target:
                x = F.interpolate(x, size=target, mode="nearest")
            fused = x if fused is None else fused + x
        return fused

    def forward(self, feats: list[torch.Tensor], out_size: tuple[int, int]) -> torch.Tensor:
        logits = self.classifier(self.dropout(self.fuse(feats)))
        # Bilinear here (not nearest) because boundary smoothness affects the result;
        # TensorRT supports bilinear resize natively.
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class SegDepthFPNHead(SemanticFPNHead):
    """Class logits and metric depth from ONE fused feature map.

    The four-head design (hydranet-old/docs/FOURHEAD_TODO.md) specifies head 2 as
    "10-class mask + metres per pixel", one tower with two predictors, and this is that head
    ported forward. It is not interchangeable with declaring `semantic_fpn` and
    `depth_fpn` side by side: those build two trunks that never see each other's
    gradients, where this one makes the same features answer both questions -- which is
    the argument for a joint head, since a surface's class and its distance are the same
    geometry read two ways.

    Depth is predicted in log space and exponentiated last, for the reason `heads/depth.py`
    gives: depth spans two orders of magnitude in one frame and a linear output spends its
    capacity on the far field. `exp` as the final operator also keeps the graph inside the
    Conv/BN/ReLU/Resize/Exp set TensorRT converts without plugins.

    **The depth branch deliberately does not see `self.dropout`.** Dropping activations
    perturbs a regressed value directly, while a classifier only has to keep the argmax.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        in_levels: Sequence[int] = (0, 1, 2),
        channels: int = 64,
        dropout: float = 0.1,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
    ):
        super().__init__(in_channels, num_classes, in_levels, channels, dropout)
        self.depth_predictor = nn.Conv2d(channels, 1, 1)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        # Start at the geometric mean of the range rather than at exp(0) = 1 m, and keep
        # the weight near-zero but NOT zero: the gradient reaching the tower through a
        # 1x1 convolution *is* its weight, so a zero init would sever this branch for the
        # whole first step.
        nn.init.normal_(self.depth_predictor.weight, std=1e-4)
        nn.init.constant_(
            self.depth_predictor.bias,
            float(math.log(min_depth * max_depth) / 2),
        )

    def forward(  # type: ignore[override]
        self, feats: list[torch.Tensor], out_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fuse(feats)
        logits = F.interpolate(
            self.classifier(self.dropout(fused)),
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )
        log_d = F.interpolate(
            self.depth_predictor(fused), size=out_size, mode="bilinear", align_corners=False
        )
        # Clamp in log space, then exponentiate: the gradient stays finite at both ends,
        # unlike a sigmoid-scaled output which vanishes there. Clamping after exp() would
        # also be too late in fp16, where a large activation overflows to inf first.
        return logits, torch.exp(
            log_d.clamp(math.log(self.min_depth), math.log(self.max_depth))
        )


def build_seg_depth_head(cfg, in_channels: int) -> SegDepthFPNHead:
    return SegDepthFPNHead(
        in_channels=in_channels,
        num_classes=cfg["num_classes"],
        in_levels=cfg.get("in_levels", [0, 1, 2]),
        channels=cfg.get("channels", 64),
        dropout=cfg.get("dropout", 0.1),
        min_depth=cfg.get("min_depth", 0.1),
        max_depth=cfg.get("max_depth", 10.0),
    )


def build_seg_head(cfg, in_channels: int) -> SemanticFPNHead:
    return SemanticFPNHead(
        in_channels=in_channels,
        num_classes=cfg["num_classes"],
        in_levels=cfg.get("in_levels", [0, 1, 2]),
        channels=cfg.get("channels", 64),
        dropout=cfg.get("dropout", 0.1),
    )
