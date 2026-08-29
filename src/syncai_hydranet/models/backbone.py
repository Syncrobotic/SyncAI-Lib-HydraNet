"""Backbone: torchvision RegNet / ResNet exposing multi-scale features C2..C5.

Chosen for TensorRT: only Conv/BN/ReLU, no squeeze-excite and no Swish, so whole
stages fuse into single kernels. The constraint came from a Jetson Orin and outlived it
-- kernel fusion is what the serving card wants too, and the graph was already shaped for
it. RegNetX also gives a better
accuracy/latency ratio than ResNet at equal FLOPs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

# name -> (torchvision builder, per-stage output channels or None to infer)
_REGISTRY = {
    "regnet_x_400mf": ("regnet_x_400mf", None),
    "regnet_x_800mf": ("regnet_x_800mf", None),
    "regnet_x_1_6gf": ("regnet_x_1_6gf", None),
    "resnet18": ("resnet18", [64, 128, 256, 512]),
    "resnet34": ("resnet34", [64, 128, 256, 512]),
    "resnet50": ("resnet50", [256, 512, 1024, 2048]),
}


class Backbone(nn.Module):
    """Uniform interface: ``forward(x) -> [C2, C3, C4, C5]`` at strides 4/8/16/32."""

    def __init__(
        self, name: str = "regnet_x_800mf", pretrained: bool = True, frozen_stages: int = 0
    ):
        super().__init__()
        if name not in _REGISTRY:
            raise ValueError(f"unknown backbone: {name}, available: {list(_REGISTRY)}")
        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, _REGISTRY[name][0])(weights=weights)

        if name.startswith("resnet"):
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
            self.out_channels = list(_REGISTRY[name][1])
        else:
            self.stem = net.stem
            blocks = list(net.trunk_output.children())  # 4 stages
            self.stages = nn.ModuleList(blocks)
            self.out_channels = [_stage_out_channels(b) for b in blocks]

        self._freeze(frozen_stages)

    def _freeze(self, n: int) -> None:
        if n <= 0:
            return
        for p in self.stem.parameters():
            p.requires_grad = False
        for stage in self.stages[: n - 1]:
            for p in stage.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats  # [C2 (1/4), C3 (1/8), C4 (1/16), C5 (1/32)]


def _stage_out_channels(stage: nn.Module) -> int:
    """Infer a RegNet stage's output width from its last BatchNorm."""
    last = None
    for m in stage.modules():
        if isinstance(m, nn.BatchNorm2d):
            last = m.num_features
    if last is None:
        raise RuntimeError("could not infer stage output channels")
    return last


def build_backbone(cfg) -> Backbone:
    return Backbone(
        name=cfg.get("name", "regnet_x_800mf"),
        pretrained=cfg.get("pretrained", True),
        frozen_stages=cfg.get("frozen_stages", 0),
    )
