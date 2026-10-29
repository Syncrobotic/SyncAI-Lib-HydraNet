"""DINOv3 ViT backbone → Simple Feature Pyramid (ViTDet-style) → [C2,C3,C4,C5].
Drop-in for the torchvision Backbone: same forward contract + .out_channels."""
from __future__ import annotations
import os
import torch, torch.nn as nn
os.environ.setdefault("HF_HOME", "/data/models/dinov3/hf")

# ViT-T is the only tiny DINOv3 checkpoint timm ships and it exists solely in the
# `_qkvb` (qkv-bias) variant, hence the different suffix.
_MAP = {"dinov3_vitt": "vit_tiny_patch16_dinov3_qkvb",
        "dinov3_vits": "vit_small_patch16_dinov3",
        "dinov3_vitb": "vit_base_patch16_dinov3"}

class DINOv3Backbone(nn.Module):
    def __init__(self, name: str = "dinov3_vits", pretrained: bool = True, frozen: bool = False):
        super().__init__()
        import timm
        self.vit = timm.create_model(_MAP.get(name, _MAP["dinov3_vits"]),
                                     pretrained=pretrained, num_classes=0)
        self.npref = getattr(self.vit, "num_prefix_tokens", 5)
        d = self.vit.embed_dim
        if d % 32:
            raise ValueError(f"{name}: embed_dim {d} is not divisible by the 32 GroupNorm groups")
        self.up4 = nn.Sequential(nn.ConvTranspose2d(d, d, 2, 2), nn.GroupNorm(32, d),
                                 nn.GELU(), nn.ConvTranspose2d(d, d, 2, 2))
        self.up2 = nn.ConvTranspose2d(d, d, 2, 2)
        self.down = nn.Conv2d(d, d, 3, 2, 1)
        self.out_channels = [d, d, d, d]
        if frozen:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        f = self.vit.forward_features(x)[:, self.npref:, :]
        gh, gw = H // 16, W // 16
        c4 = f.transpose(1, 2).reshape(B, self.vit.embed_dim, gh, gw).contiguous()
        return [self.up4(c4), self.up2(c4), c4, self.down(c4)]  # C2,C3,C4,C5
