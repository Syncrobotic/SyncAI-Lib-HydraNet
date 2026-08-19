"""FCOS anchor-free detection head.

Per FPN level the forward pass emits::

    cls_logits  [B, C, H, W]
    bbox_reg    [B, 4, H, W]   (l, t, r, b distances, already exp'd and stride-scaled)
    centerness  [B, 1, H, W]

Target assignment (training) and decode/NMS (inference) both live in this file but
outside ``forward``, so the exported ONNX graph stays free of dynamic control flow.

Anchor-free means no anchor sizes, aspect ratios or matching IoU thresholds to retune
per site, which matters for a robot that gets redeployed to new environments.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torchvision

from .text_classifier import TextEmbeddingClassifier

INF = 1e8


class Scale(nn.Module):
    def __init__(self, init: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _num_groups(ch: int, max_groups: int = 32) -> int:
    g = min(max_groups, ch)
    while ch % g != 0:
        g -= 1
    return g


def _tower(ch: int, n: int) -> nn.Sequential:
    """GroupNorm, not BatchNorm: these convolutions are shared across five FPN levels
    whose feature statistics differ widely, which would poison BN's running stats."""
    layers = []
    for _ in range(n):
        layers += [
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.GroupNorm(_num_groups(ch), ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


# Two score thresholds, because they answer different questions and a single default
# would be wrong for one of them.
#
# COCOeval integrates precision over the whole recall curve, so discarding low-scoring
# boxes before it runs truncates that curve and *lowers* mAP. 0.05 is the convention
# every COCO codebase uses; it is a metric setting and never a deployment one.
#
# Anything a human or a planner looks at wants the opposite: few boxes, mostly right.
# 0.30 on this model is already generous -- its best detections score around 0.15 -- but
# a number chosen per call site is how four call sites ended up with four values.
SCORE_THR_EVAL = 0.05
SCORE_THR_VIEW = 0.30

# Fixed ceiling CCTV in a shop needs a third, and it is lower than either. Measured on
# two site cameras, 40 frames each (runs/review_20260816/):
#
#     threshold   Taichung-cam01        Kaohsiung-cam08
#       0.15       11.8 boxes/frame      10.0 boxes/frame
#       0.25        3.4                   0.0
#       0.30        ~2                    0.0
#       0.35        1.6                   0.0
#
# Read the right-hand column. At the shipped viewing default this camera returns **no
# detections at all** -- not few, none -- across an Apple store with a podium of
# MacBooks, wall shelving of boxed stock and customers in frame. The whole distribution
# sits below the cut, so nothing downstream can recover it and the frame looks like a
# scene with nothing in it rather than like a threshold set too high.
#
# 0.20 is a compromise and is stated as one. It restores boxes on the dead cameras
# without the ~10/frame that 0.15 produces, and it is a stopgap for a calibration
# problem rather than a fix: scores are not comparable between these cameras, so one
# global number cannot be right for both. The fix is per-class thresholds fitted to
# hand-labelled site boxes, and this project has none yet -- METHODOLOGY.md ranks
# that work and RETAIL.md s6 explains why the test split has to come first.
#
# Deliberately NOT the default for SCORE_THR_VIEW: 0.30 was chosen for the robot's
# forward-facing camera, where a false box is a stop the robot did not need, and
# nothing here re-measured that case.
SCORE_THR_RETAIL = 0.20


class FCOSHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        in_levels: Sequence[int] = (0, 1, 2, 3, 4),
        channels: int = 96,
        num_convs: int = 4,
        strides: Sequence[int] = (8, 16, 32, 64, 128),
        cls_head: str = "linear",
        embed_dim: int = 512,
    ):
        super().__init__()
        self.in_levels = list(in_levels)
        self.strides = list(strides)
        self.num_classes = num_classes
        self.proj = (
            nn.Conv2d(in_channels, channels, 1) if in_channels != channels else nn.Identity()
        )
        self.cls_tower = _tower(channels, num_convs)
        self.reg_tower = _tower(channels, num_convs)
        # `linear` is the shipped head: one learned vector per class, class list frozen
        # into the weights. `text_embedding` scores against a matrix of class-name
        # embeddings instead, which makes the vocabulary a config rather than a retrain --
        # see heads/text_classifier.py for why the cam08 audit makes that worth having.
        if cls_head not in ("linear", "text_embedding"):
            raise ValueError(f"cls_head must be linear or text_embedding, got {cls_head!r}")
        self.cls_head = cls_head
        if cls_head == "text_embedding":
            self.cls_pred = TextEmbeddingClassifier(channels, embed_dim, num_classes)
        else:
            self.cls_pred = nn.Conv2d(channels, num_classes, 3, 1, 1)
            # Focal loss prior: start with a positive probability around 0.01.
            nn.init.constant_(self.cls_pred.bias, -math.log((1 - 0.01) / 0.01))
        self.reg_pred = nn.Conv2d(channels, 4, 3, 1, 1)
        self.ctr_pred = nn.Conv2d(channels, 1, 3, 1, 1)
        self.scales = nn.ModuleList(Scale(1.0) for _ in self.in_levels)
        # Per-level regression ranges, as in the FCOS paper.
        self.regress_ranges = [(-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)]

    def forward(self, feats: list[torch.Tensor]):
        cls_out, reg_out, ctr_out = [], [], []
        for i, lv in enumerate(self.in_levels):
            x = self.proj(feats[lv])
            c = self.cls_tower(x)
            r = self.reg_tower(x)
            cls_out.append(self.cls_pred(c))
            ctr_out.append(self.ctr_pred(r))
            # exp(scale * pred) * stride keeps the distances positive.
            reg_out.append(torch.exp(self.scales[i](self.reg_pred(r))) * self.strides[i])
        return cls_out, reg_out, ctr_out

    # ---------------- target assignment (training only, never exported) ----------

    @torch.no_grad()
    def get_targets(self, feats_shapes, boxes_list, labels_list, device):
        """boxes: ``[N,4]`` xyxy in input-image coordinates; labels: ``[N]``, 0-based."""
        points, level_ids = self._grid_points(feats_shapes, device)
        cls_t, reg_t, ctr_t = [], [], []
        for boxes, labels in zip(boxes_list, labels_list, strict=True):
            c, r, ct = self._assign_single(points, level_ids, boxes, labels)
            cls_t.append(c)
            reg_t.append(r)
            ctr_t.append(ct)
        return points, torch.stack(cls_t), torch.stack(reg_t), torch.stack(ctr_t)

    def _grid_points(self, shapes, device):
        pts, lids = [], []
        for i, (h, w) in enumerate(shapes):
            s = self.strides[i]
            ys = (torch.arange(h, device=device) + 0.5) * s
            xs = (torch.arange(w, device=device) + 0.5) * s
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            pts.append(torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1))
            lids.append(torch.full((h * w,), i, device=device, dtype=torch.long))
        return torch.cat(pts), torch.cat(lids)

    def _assign_single(self, points, level_ids, boxes, labels):
        n_pts = points.shape[0]
        cls_t = torch.full((n_pts,), self.num_classes, device=points.device, dtype=torch.long)
        reg_t = points.new_zeros((n_pts, 4))
        ctr_t = points.new_zeros((n_pts,))
        if boxes.numel() == 0:
            return cls_t, reg_t, ctr_t
        xs, ys = points[:, 0:1], points[:, 1:2]
        left = xs - boxes[:, 0]
        top = ys - boxes[:, 1]
        right = boxes[:, 2] - xs
        bottom = boxes[:, 3] - ys
        ltrb = torch.stack([left, top, right, bottom], dim=-1)  # [n_pts, n_boxes, 4]
        inside = ltrb.min(dim=-1).values > 0
        max_dist = ltrb.max(dim=-1).values
        ranges = points.new_tensor(self.regress_ranges)[level_ids]
        in_range = (max_dist >= ranges[:, 0:1]) & (max_dist <= ranges[:, 1:2])
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        areas = areas[None].repeat(n_pts, 1)
        areas[~(inside & in_range)] = INF
        min_area, min_idx = areas.min(dim=1)
        pos = min_area < INF
        cls_t[pos] = labels[min_idx[pos]]
        reg_t[pos] = ltrb[pos, min_idx[pos]]
        lr = reg_t[pos][:, [0, 2]]
        tb = reg_t[pos][:, [1, 3]]
        # Centerness: how central the point is inside its box. Multiplied into the score
        # at inference, it suppresses the low-quality predictions made near box edges.
        ctr_t[pos] = torch.sqrt(
            (lr.min(-1).values / lr.max(-1).values.clamp(min=1e-6))
            * (tb.min(-1).values / tb.max(-1).values.clamp(min=1e-6))
        )
        return cls_t, reg_t, ctr_t

    # ---------------- decode + NMS (inference post-processing) -------------------

    @torch.no_grad()
    def decode(
        self,
        cls_out,
        reg_out,
        ctr_out,
        score_thr=SCORE_THR_VIEW,
        nms_thr=0.6,
        max_det=100,
        img_size=None,
    ):
        device = cls_out[0].device
        shapes = [c.shape[-2:] for c in cls_out]
        points, _ = self._grid_points(shapes, device)
        b_size = cls_out[0].shape[0]
        flat_cls = torch.cat(
            [c.permute(0, 2, 3, 1).reshape(b_size, -1, self.num_classes) for c in cls_out],
            dim=1,
        ).sigmoid()
        flat_reg = torch.cat(
            [r.permute(0, 2, 3, 1).reshape(b_size, -1, 4) for r in reg_out], dim=1
        )
        flat_ctr = torch.cat(
            [c.permute(0, 2, 3, 1).reshape(b_size, -1) for c in ctr_out], dim=1
        ).sigmoid()
        results = []
        for b in range(b_size):
            scores = flat_cls[b] * flat_ctr[b][:, None]
            score, label = scores.max(dim=1)
            keep = score > score_thr
            score, label = score[keep], label[keep]
            pts, ltrb = points[keep], flat_reg[b][keep]
            boxes = torch.stack(
                [
                    pts[:, 0] - ltrb[:, 0],
                    pts[:, 1] - ltrb[:, 1],
                    pts[:, 0] + ltrb[:, 2],
                    pts[:, 1] + ltrb[:, 3],
                ],
                dim=-1,
            )
            if img_size is not None:
                h, w = img_size
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
            keep_idx = torchvision.ops.batched_nms(boxes, score, label, nms_thr)[:max_det]
            results.append(
                {"boxes": boxes[keep_idx], "scores": score[keep_idx], "labels": label[keep_idx]}
            )
        return results


def build_det_head(cfg, in_channels: int) -> FCOSHead:
    return FCOSHead(
        in_channels=in_channels,
        num_classes=cfg["num_classes"],
        in_levels=cfg.get("in_levels", [0, 1, 2, 3, 4]),
        channels=cfg.get("channels", 96),
        num_convs=cfg.get("num_convs", 4),
        strides=cfg.get("strides", [8, 16, 32, 64, 128]),
        cls_head=cfg.get("cls_head", "linear"),
        embed_dim=cfg.get("embed_dim", 512),
    )
