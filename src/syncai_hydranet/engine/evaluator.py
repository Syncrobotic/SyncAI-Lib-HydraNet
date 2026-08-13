"""Validation: per-head segmentation mIoU and COCO detection mAP."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.multitask import collate
from ..data.transforms import invert_geom


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.n = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        p = pred.flatten().cpu().numpy()
        t = target.flatten().cpu().numpy()
        valid = t != 255
        p, t = p[valid], t[valid]
        idx = t * self.n + p
        self.mat += np.bincount(idx, minlength=self.n**2).reshape(self.n, self.n)

    def miou(self) -> tuple[float, np.ndarray]:
        inter = np.diag(self.mat).astype(np.float64)
        union = self.mat.sum(1) + self.mat.sum(0) - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        return float(np.nanmean(iou)), iou


@torch.no_grad()
def evaluate(model, val_sets, cfg, device, logger, samples: dict | None = None) -> dict:
    """Return a metrics dict.

    If ``samples`` is a dict it is filled with the first validation batch per
    segmentation head as ``{head: (images, preds, targets)}``, which the trainer turns
    into TensorBoard comparison grids.
    """
    # Restore whatever mode the caller had it in. Training passes the EMA copy, which
    # lives in eval mode; leaving it in train mode would let a stray forward pass move
    # its BatchNorm statistics, and hydranet-eval would return a model set to train.
    was_training = model.training
    model.eval()
    metrics: dict[str, float] = {}
    seg_cms: dict[str, ConfusionMatrix] = {}
    det_results = []
    coco_gt = None
    bs = max(int(cfg["train"]["batch_size"]) // 2, 1)
    workers = int(cfg["data"].get("workers", 4))

    for _name, ds in val_sets:
        loader = DataLoader(
            ds, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=collate
        )
        sup = ds.supervises
        for batch in loader:
            images = batch["image"].to(device)
            out = model(images)

            for head in model.seg_heads:
                if head in sup and head in batch["targets"]:
                    if head not in seg_cms:
                        seg_cms[head] = ConfusionMatrix(model.seg_heads[head].num_classes)
                    pred = out[head].argmax(dim=1)
                    tgt = batch["targets"][head].to(device)
                    seg_cms[head].update(pred, tgt)
                    if samples is not None and head not in samples:
                        k = min(4, images.shape[0])
                        samples[head] = (images[:k].cpu(), pred[:k].cpu(), tgt[:k].cpu())

            if model.det_head is not None and model.det_head_name in sup:
                coco_gt = ds.coco
                dets = model.det_head.decode(
                    out["det_cls"],
                    out["det_reg"],
                    out["det_ctr"],
                    score_thr=0.05,
                    nms_thr=0.6,
                    img_size=images.shape[-2:],
                )
                # Map boxes back to original image coordinates for COCOeval. The dataset
                # records the exact scale and padding it applied; deriving it here from
                # the frame size would be wrong for every letterboxed image, because the
                # padding is not part of the original frame.
                for img_id, det, geom in zip(
                    batch["image_ids"], dets, batch["geoms"], strict=True
                ):
                    boxes = invert_geom(det["boxes"].cpu().numpy(), geom)
                    for box, score, label in zip(
                        boxes,
                        det["scores"].cpu().numpy(),
                        det["labels"].cpu().numpy(),
                        strict=True,
                    ):
                        det_results.append(
                            {
                                "image_id": int(img_id),
                                "category_id": ds.label_to_cat[int(label)],
                                "bbox": [
                                    float(box[0]),
                                    float(box[1]),
                                    float(box[2] - box[0]),
                                    float(box[3] - box[1]),
                                ],
                                "score": float(score),
                            }
                        )

    trav_names = ("blocked", "caution", "go")
    terrain_names = tuple(cfg["data"].get("terrain_classes", []))

    for head, cm in seg_cms.items():
        miou, per_class = cm.miou()
        metrics[f"{head}_mIoU"] = miou
        # Log every class separately: the safety-critical indoor classes (glass,
        # stairs) are tiny, so mIoU alone hides the fact that they never converged.
        names = trav_names if head == "traversability" else terrain_names
        for i, iou in enumerate(per_class):
            if np.isfinite(iou):
                cname = names[i] if i < len(names) else str(i)
                metrics[f"IoU/{head}/{i:02d}_{cname}"] = float(iou)
        logger.info(
            f"[val] {head} mIoU = {miou:.4f} | per-class IoU = "
            + " ".join(f"{x:.3f}" if np.isfinite(x) else "-" for x in per_class)
        )

    if det_results and coco_gt is not None:
        from pycocotools.cocoeval import COCOeval

        coco_dt = coco_gt.loadRes(det_results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        metrics["detection_mAP"] = float(ev.stats[0])
        metrics["detection_mAP50"] = float(ev.stats[1])
        logger.info(f"[val] detection mAP = {ev.stats[0]:.4f}, mAP@50 = {ev.stats[1]:.4f}")

    model.train(was_training)
    return metrics


def select_metric(metrics: dict, name: str) -> float:
    """Pull the one number that decides which checkpoint wins.

    Averaging mIoU with mAP, as this used to, is not a quantity: the scales differ by
    a factor of two, so segmentation quietly outvoted detection, and the average
    changed meaning whenever a dataset was added or removed. One named metric is
    comparable across runs; everything else stays in metrics.jsonl for analysis.
    """
    if name in metrics:
        return float(metrics[name])
    raise KeyError(
        f"train.primary_metric={name!r} was not produced by validation. "
        f"Available: {', '.join(sorted(metrics))}"
    )
