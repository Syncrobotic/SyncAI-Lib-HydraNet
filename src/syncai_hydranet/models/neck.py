"""Neck: FPN and BiFPN, both restricted to TensorRT-native operators.

Takes the backbone's [C2, C3, C4, C5] and returns [P3, P4, P5, P6, P7]:
- segmentation heads consume P3..P5 (high resolution)
- the detection head consumes P3..P7 (multi-scale)

BiFPN's fast-normalized fusion uses only ReLU, sum and divide, matching softmax's
effect while staying inside the TRT operator set. Upsampling is always nearest: it is
the fastest on TRT and avoids align_corners ambiguity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(cin: int, cout: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class DepthwiseSeparable(nn.Module):
    """Depthwise + pointwise convolution: roughly 1/8 the cost of a dense 3x3."""

    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class WeightedFusion(nn.Module):
    """BiFPN fast-normalized fusion: ``w_i = relu(a_i) / (sum relu(a_j) + eps)``."""

    def __init__(self, n: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n))
        self.eps = 1e-4

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        out = xs[0] * w[0]
        for i in range(1, len(xs)):
            out = out + xs[i] * w[i]
        return out


class BiFPNLayer(nn.Module):
    """One BiFPN layer: weighted top-down pass followed by a weighted bottom-up pass."""

    def __init__(self, ch: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        self.td_fuse = nn.ModuleList(WeightedFusion(2) for _ in range(num_levels - 1))
        self.td_conv = nn.ModuleList(DepthwiseSeparable(ch) for _ in range(num_levels - 1))
        # Bottom-up: interior levels fuse 3 inputs (original, top-down, downsampled
        # lower level); the topmost fuses 2.
        self.bu_fuse = nn.ModuleList(
            WeightedFusion(3 if 0 < i < num_levels - 1 else 2) for i in range(1, num_levels)
        )
        self.bu_conv = nn.ModuleList(DepthwiseSeparable(ch) for _ in range(num_levels - 1))

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        n = self.num_levels
        td = [None] * n
        td[n - 1] = feats[n - 1]
        for i in range(n - 2, -1, -1):
            up = F.interpolate(td[i + 1], size=feats[i].shape[-2:], mode="nearest")
            td[i] = self.td_conv[i](self.td_fuse[i]([feats[i], up]))
        out = [None] * n
        out[0] = td[0]
        for i in range(1, n):
            down = F.max_pool2d(out[i - 1], 3, stride=2, padding=1)
            ins = [td[i], down] if i == n - 1 else [feats[i], td[i], down]
            out[i] = self.bu_conv[i - 1](self.bu_fuse[i - 1](ins))
        return out


class BiFPN(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 96,
        num_levels: int = 5,
        num_repeats: int = 2,
    ):
        super().__init__()
        # C2 is dropped: at 1/4 resolution its feature map is 4x the area of P3, and
        # the segmentation heads recover the detail with a final upsample instead.
        self.lateral = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)
            )
            for c in in_channels[1:]
        )
        n_extra = num_levels - 3  # P6 and P7 are downsampled from P5
        self.extra = nn.ModuleList(
            conv_bn_act(out_channels, out_channels, 3, 2) for _ in range(n_extra)
        )
        self.layers = nn.ModuleList(
            BiFPNLayer(out_channels, num_levels) for _ in range(num_repeats)
        )
        self.out_channels = out_channels
        self.num_levels = num_levels

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        ps = [lat(f) for lat, f in zip(self.lateral, feats[1:], strict=True)]  # P3, P4, P5
        for ex in self.extra:
            ps.append(ex(ps[-1]))  # P6, P7
        for layer in self.layers:
            ps = layer(ps)
        return ps


class FPN(nn.Module):
    """Plain FPN: top-down only, simpler and cheaper than BiFPN."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 96,
        num_levels: int = 5,
        num_repeats: int = 1,  # noqa: ARG002 - accepted for parity with BiFPN
    ):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels[1:])
        self.smooth = nn.ModuleList(
            conv_bn_act(out_channels, out_channels) for _ in in_channels[1:]
        )
        n_extra = num_levels - 3
        self.extra = nn.ModuleList(
            conv_bn_act(out_channels, out_channels, 3, 2) for _ in range(n_extra)
        )
        self.out_channels = out_channels
        self.num_levels = num_levels

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [lat(f) for lat, f in zip(self.lateral, feats[1:], strict=True)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="nearest"
            )
        ps = [sm(lat) for sm, lat in zip(self.smooth, laterals, strict=True)]
        for ex in self.extra:
            ps.append(ex(ps[-1]))
        return ps


def build_neck(cfg, in_channels: list[int]):
    name = cfg.get("name", "bifpn")
    cls = {"bifpn": BiFPN, "fpn": FPN}[name]
    return cls(
        in_channels=in_channels,
        out_channels=cfg.get("out_channels", 96),
        num_levels=cfg.get("num_levels", 5),
        num_repeats=cfg.get("num_repeats", 2),
    )
