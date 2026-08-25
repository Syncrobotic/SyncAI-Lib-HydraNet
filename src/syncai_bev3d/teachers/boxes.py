"""Turning a teacher's raw output into boxes, and removing the ones it said twice.

Both teachers over-answer, and they do it in two different ways that need two different
removals -- which is why there are two greedy passes here rather than one:

**Across prompts, within a frame** (`dedupe`). The prompts inside one concept are a union
by design: `product box` and `boxed product` are two ways of asking for the same shelf.
Without this, every item appears once per prompt that found it. Measured on two frames of
Taichung-cam01: 266 `boxed_stock` boxes from two prompts over roughly 146 actual items. A
detector trained on that learns that one object is two.

**Across frames, for one camera** (`drop_static`). A box that recurs in the same pixels of
every frame is furniture, not a shopper -- the inverse of the consensus vote, which keeps
what agrees. It is **off by default in every caller** and returns both sides rather than
deleting, because the one time it was measured it removed people; see its docstring.

The two share this module and one `iou`. They did not before: `sam3_prelabel` carried a
private scalar IoU next to `analytics.tracker.iou`, so the project had two definitions of
the least ambiguous quantity in it.
"""

from __future__ import annotations

import numpy as np

from syncai_hydranet.analytics.tracker import iou


def boxes_from_masks(pairs) -> np.ndarray:
    """Tight boxes around each (mask, score) a teacher returned, as (N, 5) xyxy+score.

    An empty mask is dropped rather than becoming a degenerate box: a prompt that fired
    on nothing has no location to report, and a zero-area box is a training target that
    no assignment rule can match.
    """
    out = []
    for mask, score in pairs:
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        out.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, score])
    return np.asarray(out, dtype=float).reshape(-1, 5)


def nms(boxes: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS over (N, 5) xyxy+score boxes of one class, input order preserved.

    Order is restored at the end rather than left in score order, so a caller writing
    these next to the frame they came from gets a stable file for a stable input.
    """
    if len(boxes) < 2:
        return boxes
    order = np.argsort(-boxes[:, 4])
    keep: list[int] = []
    for i in order:
        if all(iou(boxes[i : i + 1, :4], boxes[j : j + 1, :4])[0, 0] < iou_thr for j in keep):
            keep.append(i)
    return boxes[sorted(keep)]


def dedupe(boxes: list[dict], iou_thr: float = 0.55) -> list[dict]:
    """Greedy per-class NMS across prompts, over COCO-style dicts with xywh ``bbox``.

    Separate from the segmentation path on purpose, because `sam3.compose` resolves the
    same overlap a different and equally deliberate way: two prompts of one class agreeing
    is a union of *pixels*, which needs no arbitration. Two prompts of one class agreeing
    on an *instance* is a duplicate, which does.

    Categories are visited in sorted order so the output is a function of the input alone;
    iterating the set of ids left the order up to the hash.
    """
    out: list[dict] = []
    for cat in sorted({b["category_id"] for b in boxes}):
        cand = sorted(
            (b for b in boxes if b["category_id"] == cat),
            key=lambda b: b["score"],
            reverse=True,
        )
        keep: list[dict] = []
        for b in cand:
            if keep and iou(_xyxy([b]), _xyxy(keep)).max() > iou_thr:
                continue
            keep.append(b)
        out.extend(keep)
    return out


def drop_static(per_frame: list[np.ndarray], iou_thr: float, share: float):
    """Split boxes into moving and recurring. Returns (kept, dropped) per frame.

    **Dropped, not deleted.** Kaohsiung-cam04 dropped 21 of 103 and that camera is a
    service counter where staff stand at one station for the whole sample window -- so a
    gate aimed at hanging merchandise can equally remove the shopper who is standing
    still, who is the one a loitering alarm exists for. Returning both sides is what lets
    that be looked at instead of assumed.

    The inverse of the consensus vote, and deliberately so: `sam3_prompts.py`'s hanging
    packets agree in every frame, which is what a vote rewards. Recurrence is the signal
    that the thing is not a shopper.
    """
    n = len(per_frame)
    need = max(round(share * n), 2)
    kept, dropped = [], []
    for i, boxes in enumerate(per_frame):
        if not len(boxes):
            kept.append(boxes)
            continue
        recur = np.zeros(len(boxes), dtype=int)
        for j, other in enumerate(per_frame):
            if i == j or not len(other):
                continue
            m = iou(boxes[:, :4], other[:, :4])
            recur += (m.max(axis=1) >= iou_thr).astype(int)
        keep = recur < need
        dropped.append(boxes[~keep])
        kept.append(boxes[keep])
    return kept, dropped


def _xyxy(boxes: list[dict]) -> np.ndarray:
    """COCO xywh dicts as an (N, 4) xyxy array, which is what `iou` speaks."""
    arr = np.asarray([b["bbox"] for b in boxes], dtype=float).reshape(-1, 4)
    arr[:, 2:] += arr[:, :2]
    return arr
