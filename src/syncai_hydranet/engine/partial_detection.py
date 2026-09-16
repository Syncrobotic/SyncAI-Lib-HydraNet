"""Diagnostics for reviewed subsets; unknown regions cannot establish false positives."""

from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import box_iou

PROTOCOL = "reviewed_regions_v1"
SCORE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.50
EMPTY_FRACTION = 0.95


class PartialDetectionAccumulator:
    """Fixed-threshold positive recall and false alarms wholly inside reviewed emptiness.

    This is neither precision nor AP: unreviewed objects and unmatched predictions
    outside the explicit negative regions have unknown truth status.
    """

    def __init__(self, classes):
        self.classes = tuple(classes)
        self.support = np.zeros(len(classes), dtype=np.int64)
        self.matched = np.zeros(len(classes), dtype=np.int64)
        self.empty_fp = self.unknown_predictions = self.empty_pixels = self.images = 0

    def update(self, prediction, boxes, labels, negative):
        boxes = boxes.detach().cpu().float()
        labels = labels.detach().cpu().long()
        negative = negative.detach().cpu().numpy()
        if negative.ndim != 2 or not np.isin(negative, [0, 1, 255]).all():
            raise ValueError("partial evaluation requires a 0/1/255 reviewed mask")
        if len(boxes) != len(labels) or ((labels < 0) | (labels >= len(self.classes))).any():
            raise ValueError("invalid partial evaluation targets")
        self.images += 1
        self.empty_pixels += int((negative == 1).sum())
        self.support += np.bincount(labels.numpy(), minlength=len(self.classes))
        pb = prediction["boxes"].detach().cpu().float()
        ps = prediction["scores"].detach().cpu().float()
        pl = prediction["labels"].detach().cpu().long()
        if not torch.isfinite(pb).all() or not torch.isfinite(ps).all():
            raise ValueError("nonfinite partial detection prediction")
        used = torch.zeros(len(boxes), dtype=torch.bool)
        h, w = negative.shape
        for idx in ps.argsort(descending=True):
            if ps[idx] <= SCORE_THRESHOLD:
                continue
            box = pb[idx]
            if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise ValueError("partial prediction must be clipped and nondegenerate")
            eligible = (~used) & (labels == pl[idx])
            if eligible.any():
                overlaps = box_iou(box[None], boxes)[0]
                overlaps[~eligible] = -1
                best = int(overlaps.argmax())
                if overlaps[best] >= IOU_THRESHOLD:
                    used[best] = True
                    self.matched[int(labels[best])] += 1
                    continue
            x0, y0 = np.floor(box[:2].numpy()).astype(int)
            x1, y1 = np.ceil(box[2:].numpy()).astype(int)
            if (negative[y0:y1, x0:x1] == 1).mean() >= EMPTY_FRACTION:
                self.empty_fp += 1
            else:
                self.unknown_predictions += 1

    def metrics(self, name):
        prefix = f"partial_det/{name}"
        result = {
            f"{prefix}/images": float(self.images),
            f"{prefix}/support": float(self.support.sum()),
            f"{prefix}/matched": float(self.matched.sum()),
            f"{prefix}/empty_fp": float(self.empty_fp),
            f"{prefix}/empty_pixels": float(self.empty_pixels),
            f"{prefix}/unknown_predictions": float(self.unknown_predictions),
        }
        if self.support.sum():
            result[f"{prefix}/recall50_at_020"] = float(self.matched.sum()) / float(
                self.support.sum()
            )
        if self.empty_pixels:
            result[f"{prefix}/empty_fp_per_megapixel"] = self.empty_fp * 1e6 / self.empty_pixels
        for name, support, matched in zip(
            self.classes, self.support, self.matched, strict=True
        ):
            result[f"{prefix}/{name}/support"] = float(support)
            result[f"{prefix}/{name}/matched"] = float(matched)
            if support:
                result[f"{prefix}/{name}/recall50_at_020"] = float(matched) / float(support)
        return result
