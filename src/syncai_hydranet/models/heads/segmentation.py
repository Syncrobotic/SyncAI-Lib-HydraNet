"""Semantic-FPN segmentation head, shared by the traversability and terrain heads.

P3..P5 each pass through a few convolutions, are upsampled to P3 resolution and summed,
then a 1x1 classifier produces logits which are bilinearly resized back to the input
resolution. No decoder and no dilated convolutions: multi-scale fusion already happened
in the neck, so a U-Net style decoder here would be duplicated work costing more
parameters than the entire neck.
"""

from __future__ import annotations

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

    def forward(self, feats: list[torch.Tensor], out_size: tuple[int, int]) -> torch.Tensor:
        base = feats[self.in_levels[0]]
        target = base.shape[-2:]
        fused = None
        for conv, lv in zip(self.scale_convs, self.in_levels, strict=True):
            x = conv(feats[lv])
            if x.shape[-2:] != target:
                x = F.interpolate(x, size=target, mode="nearest")
            fused = x if fused is None else fused + x
        logits = self.classifier(self.dropout(fused))
        # Bilinear here (not nearest) because boundary smoothness affects the result;
        # TensorRT supports bilinear resize natively.
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


def build_seg_head(cfg, in_channels: int) -> SemanticFPNHead:
    return SemanticFPNHead(
        in_channels=in_channels,
        num_classes=cfg["num_classes"],
        in_levels=cfg.get("in_levels", [0, 1, 2]),
        channels=cfg.get("channels", 64),
        dropout=cfg.get("dropout", 0.1),
    )
