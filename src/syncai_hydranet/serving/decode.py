"""Host-side FCOS decode for the exported engine's raw detection outputs.

The exported graph deliberately carries no NMS and no dynamic control flow (see
cli/export_onnx.py), so decode happens on the host. ``FCOSHead.decode`` does this
job inside the model; a TensorRT serving process has no model object, only named
output buffers, so this mirrors that method over plain arrays. The parity test
(tests/test_serving_decode.py) holds the two implementations to the same answer on
the same tensors -- this file must never drift into a second opinion.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torchvision


class FcosDecoder:
    """Decode per-level (cls, reg, ctr) maps into scored, NMS'd boxes.

    ``score_thr`` may be a scalar or a per-class sequence -- the per-class working
    thresholds are the serving layer's contract (see camera.DEFAULT_THRESHOLDS for
    the measurement behind that).
    """

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int] = (8, 16, 32, 64, 128),
        nms_thr: float = 0.6,
        max_det: int = 100,
        pre_nms_topk: int | None = None,
    ):
        """``pre_nms_topk`` keeps only the top-k candidates per frame before NMS.
        None reproduces FCOSHead.decode exactly (the parity test runs with None);
        serving sets it because a low score floor admits thousands of candidates
        per frame and NMS over them was measured at ~30 ms/frame in the pilot --
        the top-k can only drop boxes that scored below every kept one, so with
        k comfortably above max_det the kept set is unchanged in practice."""
        self.num_classes = int(num_classes)
        self.strides = list(strides)
        self.nms_thr = float(nms_thr)
        self.max_det = int(max_det)
        self.pre_nms_topk = None if pre_nms_topk is None else int(pre_nms_topk)
        self._points: torch.Tensor | None = None
        self._shapes: list[tuple[int, int]] | None = None

    def _grid_points(self, shapes: list[tuple[int, int]]) -> torch.Tensor:
        """(x+0.5)*stride grid centres, concatenated over levels; cached, because a
        fixed-batch engine's level shapes never change between calls."""
        if self._points is not None and shapes == self._shapes:
            return self._points
        pts = []
        for (h, w), s in zip(shapes, self.strides, strict=True):
            ys = (torch.arange(h, dtype=torch.float32) + 0.5) * s
            xs = (torch.arange(w, dtype=torch.float32) + 0.5) * s
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            pts.append(torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1))
        self._points = torch.cat(pts)
        self._shapes = list(shapes)
        return self._points

    def __call__(
        self,
        cls_levels: Sequence[np.ndarray],
        reg_levels: Sequence[np.ndarray],
        ctr_levels: Sequence[np.ndarray],
        score_thr: float | Sequence[float],
        img_size: tuple[int, int] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """[B,C,H,W]/[B,4,H,W]/[B,1,H,W] per level -> one dict per batch element."""
        cls_t = [torch.from_numpy(np.ascontiguousarray(c)) for c in cls_levels]
        reg_t = [torch.from_numpy(np.ascontiguousarray(r)) for r in reg_levels]
        ctr_t = [torch.from_numpy(np.ascontiguousarray(c)) for c in ctr_levels]
        shapes = [(int(c.shape[-2]), int(c.shape[-1])) for c in cls_t]
        points = self._grid_points(shapes)
        b_size = cls_t[0].shape[0]
        flat_cls = torch.cat(
            [c.permute(0, 2, 3, 1).reshape(b_size, -1, self.num_classes) for c in cls_t],
            dim=1,
        ).sigmoid()
        flat_reg = torch.cat(
            [r.permute(0, 2, 3, 1).reshape(b_size, -1, 4) for r in reg_t], dim=1
        )
        flat_ctr = torch.cat(
            [c.permute(0, 2, 3, 1).reshape(b_size, -1) for c in ctr_t], dim=1
        ).sigmoid()

        thr = torch.as_tensor(score_thr, dtype=torch.float32)
        if thr.ndim not in (0, 1) or (thr.ndim == 1 and len(thr) != self.num_classes):
            raise ValueError(
                f"score_thr must be a scalar or one value per class "
                f"({self.num_classes}), got shape {tuple(thr.shape)}"
            )

        results = []
        for b in range(b_size):
            scores = flat_cls[b] * flat_ctr[b][:, None]
            score, label = scores.max(dim=1)
            keep = score > (thr if thr.ndim == 0 else thr[label])
            score, label = score[keep], label[keep]
            pts, ltrb = points[keep], flat_reg[b][keep]
            if self.pre_nms_topk is not None and len(score) > self.pre_nms_topk:
                top = score.topk(self.pre_nms_topk).indices
                score, label, pts, ltrb = score[top], label[top], pts[top], ltrb[top]
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
            keep_idx = torchvision.ops.batched_nms(boxes, score, label, self.nms_thr)
            keep_idx = keep_idx[: self.max_det]
            results.append(
                {
                    "boxes": boxes[keep_idx].numpy().astype(np.float64),
                    "scores": score[keep_idx].numpy().astype(np.float64),
                    "labels": label[keep_idx].numpy().astype(np.int64),
                }
            )
        return results


# The dense head already runs on every frame and scores `person` at IoU 0.885 while this
# one's boxes score mAP@50 0.302. Measured 2026-08-26 over four commissioned cameras,
# 40 frames each: at the shipped 0.35 score threshold the dense map carried **73 person
# regions with no box at all -- 20% more people than the box head returned** -- and
# cropping every one of them showed the same population: shoppers whose lower body is
# behind a counter or a display table. Truncated, not absent.
#
# Dropping the score threshold finds them (73 uncovered at 0.35, 3 at 0.15) and admits
# false ones with them. So admit the low-scoring boxes and make the dense head vouch:
#
#     min person-pixel fraction    of score >= 0.35 boxes    of score < 0.35 boxes
#                          0.40                     100%                      71%
#
# 0.40 is the largest fraction that costs nothing on the set already trusted -- every box
# today's threshold accepts clears it -- which is why it is the default rather than a
# number picked to make a graph look good.
MIN_PERSON_FRACTION = 0.40


def person_pixel_fraction(box, class_map: np.ndarray, person_id: int) -> float:
    """Fraction of a box's pixels the dense head calls `person`. 0.0 for an empty box.

    One number per box and no connected components: serving cannot afford a labelling
    pass per frame, and it does not need one -- the question is whether *this* box has
    person underneath it, not which person.
    """
    h, w = class_map.shape[-2:]
    x0 = max(0, round(float(box[0])))
    y0 = max(0, round(float(box[1])))
    x1 = min(w, round(float(box[2])))
    y1 = min(h, round(float(box[3])))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((class_map[y0:y1, x0:x1] == person_id).mean())


def confirm_with_dense(
    result: dict[str, np.ndarray],
    class_map: np.ndarray,
    person_label: int,
    person_id: int,
    min_fraction: float = MIN_PERSON_FRACTION,
) -> dict[str, np.ndarray]:
    """Keep a `person` box only where the dense head puts person pixels under it.

    Applied to one decoded frame, in place of nothing -- today the score threshold is the
    only filter, and it is the one throwing away the truncated shoppers. Other classes
    pass through untouched: `bag`, `boxed_stock` and `device` have no dense channel to
    vouch for them in the `retail_surfaces` taxonomy.

    Returns a new dict; the input is not modified.
    """
    labels = np.asarray(result["labels"])
    if not len(labels):
        return dict(result)
    boxes = np.asarray(result["boxes"])
    keep = np.ones(len(labels), dtype=bool)
    for i in np.nonzero(labels == person_label)[0]:
        keep[i] = person_pixel_fraction(boxes[i], class_map, person_id) >= min_fraction
    return {k: np.asarray(v)[keep] for k, v in result.items()}
