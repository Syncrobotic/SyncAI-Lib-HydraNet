"""Box-conditioned bottom-up keypoints: heatmaps at P3, decoded inside detection boxes.

The design walks straight out of two measurements in docs/PLAN.md. First, resolution:
99.9% of the 66,599 verified site person boxes clear 32 px at network scale (median
178 px, 22 stride-8 cells tall), so a heatmap head at P3 has room for the great majority
and is *expected* to be poor on the bottom 5%. Second, cost: the shared trunk buys a
second per-frame task for ~3%; this head keeps `forward` pure convolution -- a small
tower and a 1x1 -- so the export contract survives.

**Grouping is the detection head's problem, on purpose.** Classical bottom-up pose pays
for associative embeddings or part-affinity fields to answer "which keypoint belongs to
which person". This network already answers that question every frame with boxes, so
decoding is per-box argmax over the box's own heatmap window. No embedding channels, no
grouping graph, nothing to export. The cost of that bargain is honest: two people whose
boxes overlap can steal each other's peaks inside the intersection, which is exactly the
crowded case top-down pose also fumbles -- and the crop-stage fallback PLAN 2.2 names
stays open if it matters in practice.

Heatmaps stay at P3 resolution (stride 8). A segmentation head upsamples to input size
because its consumer is per-pixel; a keypoint's consumer is a coordinate, and argmax at
stride 8 plus the peak's sub-cell offset does not get better by bilinear upsampling --
it gets 64x more memory traffic.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# COCO-17 order, the one ViTPose emits and analytics/events/pose.py consumes.
NUM_KEYPOINTS = 17
HEATMAP_STRIDE = 8


class PoseP3Head(nn.Module):
    """A small conv tower on P3 -> NUM_KEYPOINTS heatmap logits at stride 8."""

    def __init__(self, in_channels: int, channels: int = 96, num_convs: int = 3):
        super().__init__()
        tower: list[nn.Module] = []
        ch = in_channels
        for _ in range(num_convs):
            tower += [
                nn.Conv2d(ch, channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, channels),
                nn.ReLU(inplace=True),
            ]
            ch = channels
        self.tower = nn.Sequential(*tower)
        self.predictor = nn.Conv2d(channels, NUM_KEYPOINTS, 1)
        # Heatmaps are almost entirely background: bias the logits so training starts
        # from "nothing anywhere" instead of spending its first epochs learning that.
        bias = self.predictor.bias
        assert bias is not None  # nn.Conv2d carries a bias unless bias=False is asked for
        nn.init.constant_(bias, -4.0)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        return self.predictor(self.tower(feats[0]))  # [B, K, H/8, W/8] logits


class PoseHeatmapLoss(nn.Module):
    """Focal BCE against Gaussians rendered from the teacher's keypoints, on the fly.

    Targets arrive as per-image keypoint arrays rather than pre-rendered maps: rendering
    here costs a few kernel launches and keeps the dataset format one line per person --
    the same shape the teacher writes and the evaluator reads. A keypoint below
    `min_conf` renders nothing: a teacher that was not sure is not a label, and an
    absent Gaussian is background supervision, which for an occluded joint is usually
    the truth on this viewpoint.
    """

    def __init__(self, sigma_cells: float = 1.5, min_conf: float = 0.3, alpha: float = 2.0):
        super().__init__()
        self.sigma = sigma_cells
        self.min_conf = min_conf
        self.alpha = alpha

    def render(self, kps: list[torch.Tensor], shape, device) -> torch.Tensor:
        """[B lists of (N,17,3) in input pixels] -> [B,K,h,w] Gaussian maps at stride 8."""
        b = len(kps)
        _, _, h, w = shape
        target = torch.zeros((b, NUM_KEYPOINTS, h, w), device=device)
        ys = torch.arange(h, device=device).view(-1, 1)
        xs = torch.arange(w, device=device).view(1, -1)
        for i, persons in enumerate(kps):
            if persons is None or len(persons) == 0:
                continue
            persons = persons.to(device)
            for person in persons:
                for k in range(NUM_KEYPOINTS):
                    x, y, c = person[k]
                    if c < self.min_conf:
                        continue
                    cx = (x / HEATMAP_STRIDE) - 0.5
                    cy = (y / HEATMAP_STRIDE) - 0.5
                    if not (0 <= cx < w and 0 <= cy < h):
                        continue
                    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * self.sigma**2))
                    target[i, k] = torch.maximum(target[i, k], g)
        return target

    def forward(self, logits: torch.Tensor, kps: list[torch.Tensor]) -> torch.Tensor:
        target = self.render(kps, logits.shape, logits.device)
        pred = torch.sigmoid(logits)
        # focal-style modulation on a dense MSE: the Gaussian peak pixels are rare, so a
        # plain mean is 99% background gradient
        weight = torch.where(
            target > 0.1, 1.0 + self.alpha * target, torch.full_like(target, 0.1)
        )
        return (weight * (pred - target) ** 2).mean() * 100.0


@torch.no_grad()
def decode_boxes(
    heatmaps: torch.Tensor, boxes: torch.Tensor, min_box_px: float = 16.0
) -> torch.Tensor:
    """Per-box keypoints from one image's heatmaps: (K,h,w) x (N,4 xyxy) -> (N,K,3).

    Coordinates come back in input pixels; the confidence is the sigmoid peak. A box
    shorter than `min_box_px` returns zero-confidence keypoints rather than an argmax
    over two cells -- below the measured 32 px floor there is nothing to decode, and a
    confident wrong skeleton is worse for the temporal model than an absent one.
    """
    k, h, w = heatmaps.shape
    out = torch.zeros((len(boxes), k, 3), device=heatmaps.device)
    probs = torch.sigmoid(heatmaps)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if (y1 - y0) < min_box_px or (x1 - x0) < min_box_px:
            continue
        c0 = int(max(0, math.floor(x0 / HEATMAP_STRIDE)))
        r0 = int(max(0, math.floor(y0 / HEATMAP_STRIDE)))
        c1 = int(min(w, math.ceil(x1 / HEATMAP_STRIDE) + 1))
        r1 = int(min(h, math.ceil(y1 / HEATMAP_STRIDE) + 1))
        if c1 <= c0 or r1 <= r0:
            continue
        window = probs[:, r0:r1, c0:c1]
        flat = window.flatten(1)
        conf, idx = flat.max(dim=1)
        wy = idx // (c1 - c0)
        wx = idx % (c1 - c0)
        # sub-cell refinement: shift a quarter cell toward the larger neighbour, the
        # standard cheap alternative to an offset channel
        px = (wx + c0).float()
        py = (wy + r0).float()
        for kk in range(k):
            x_i, y_i = int(wx[kk]), int(wy[kk])
            if 0 < x_i < window.shape[2] - 1:
                left, right = window[kk, y_i, x_i - 1], window[kk, y_i, x_i + 1]
                px[kk] += 0.25 * torch.sign(right - left).item()
            if 0 < y_i < window.shape[1] - 1:
                up, down = window[kk, y_i - 1, x_i], window[kk, y_i + 1, x_i]
                py[kk] += 0.25 * torch.sign(down - up).item()
        out[i, :, 0] = (px + 0.5) * HEATMAP_STRIDE
        out[i, :, 1] = (py + 0.5) * HEATMAP_STRIDE
        out[i, :, 2] = conf
    return out


def build_pose_head(cfg: dict, in_channels: int) -> PoseP3Head:
    return PoseP3Head(
        in_channels,
        channels=cfg.get("channels", 96),
        num_convs=cfg.get("num_convs", 3),
    )


def build_pose_loss(cfg: dict) -> PoseHeatmapLoss:
    return PoseHeatmapLoss(
        sigma_cells=cfg.get("sigma_cells", 1.5),
        min_conf=cfg.get("teacher_min_conf", 0.3),
    )
