"""A specialist glazing head on frozen, existing HydraNet scene features.

The small head can be trained from cached features on CPU. It is a separate model;
training it cannot change the shipped person detector or its terrain taxonomy.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from syncai_hydranet.shipped import load_model


class GlazingEncoder(nn.Module):
    def __init__(self, config, checkpoint):
        super().__init__()
        model, _cfg, _device = load_model(
            config, checkpoint, device="cpu", weights="ema", validate=False
        )
        self.backbone = model.backbone
        self.neck = model.neck
        self.fusion = model.seg_heads["terrain"]
        self.channels = self.fusion.classifier.in_channels
        self.fusion.classifier = nn.Identity()
        self.fusion.dropout = nn.Identity()
        self.requires_grad_(False)
        self.eval()

    def forward(self, image):
        features = self.neck(self.backbone(image))
        return self.fusion(features, tuple(features[0].shape[-2:]))


class GlazingHead(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 4, 1),
        )

    def forward(self, features):
        return self.layers(features)


def glazing_loss(logits, labels, weights):
    """CE plus foreground Dice; padding and unknown palette colours are ignored."""
    valid = labels != 255
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits, labels, weight=weights, ignore_index=255)
    target = F.one_hot(labels.masked_fill(~valid, 0), 4).permute(0, 3, 1, 2)
    probability = logits.softmax(1) * valid[:, None]
    target = target * valid[:, None]
    intersection = (probability * target).sum((0, 2, 3))
    union = probability.sum((0, 2, 3)) + target.sum((0, 2, 3))
    present = target.sum((0, 2, 3))[1:] > 0
    dice = (1 - (2 * intersection[1:] + 1) / (union[1:] + 1))[present]
    return ce + (dice.mean() if present.any() else logits.sum() * 0)


def confusion_scores(confusion):
    """Absent classes have no IoU, rather than being counted as perfect predictions."""
    matrix = torch.as_tensor(confusion, dtype=torch.float64)
    tp = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - tp
    iou = [float(tp[i] / union[i]) if union[i] else None for i in range(len(tp))]
    foreground = [v for v in iou[1:] if v is not None]
    return {
        "iou": iou,
        "foreground_miou": sum(foreground) / len(foreground) if foreground else None,
        "door_precision": float(tp[1] / matrix[:, 1].sum().clamp_min(1)),
        "door_recall": float(tp[1] / matrix[1].sum().clamp_min(1)),
        "confusion": matrix.to(torch.int64).tolist(),
    }
