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
