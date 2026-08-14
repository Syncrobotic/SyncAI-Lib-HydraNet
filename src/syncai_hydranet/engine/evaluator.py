"""Validation: per-head segmentation mIoU and COCO detection mAP."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.multitask import collate
from ..data.transforms import invert_geom
from ..models.heads.detection import SCORE_THR_EVAL


class ConfusionMatrix:
    """Accumulates on whichever device the predictions are already on.

    ``update`` used to copy both maps to the host and count with ``np.bincount``. At
    512x640 that is a few million pixels per batch crossing the bus, per head, per epoch,
    to compute something the GPU counts for free. Counting stays on-device and only the
    matrix -- ``n x n`` integers -- comes back, in ``miou``.

    The arithmetic is unchanged: same indices, same integer counts, so a mIoU computed
    either way is bit-identical. ``tests/test_evaluator.py`` asserts that rather than
    assuming it.
    """

    def __init__(self, num_classes: int):
        self.n = num_classes
        self.mat: torch.Tensor | None = None  # created on the first update's device

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if self.mat is None:
            self.mat = torch.zeros(self.n * self.n, dtype=torch.int64, device=pred.device)
        p = pred.reshape(-1)
        t = target.reshape(-1).to(p.device)
        valid = t != 255
        idx = t[valid] * self.n + p[valid]
        self.mat += torch.bincount(idx, minlength=self.n * self.n)

    def miou(self) -> tuple[float, np.ndarray]:
        """Mean IoU, and the per-class IoUs it averaged.

        A class absent from both prediction and ground truth has no IoU to compute and
        becomes NaN, which ``nanmean`` drops. That is the only defensible choice, but it
        means **the denominator changes with the dataset**: the indoor scheme declares 12
        terrain classes while ADE20K contains 8 of them, so this returns a mean over 8.
        Annotating the missing classes will therefore tend to *lower* mIoU even as the
        model improves. ``mIoU_classes`` is emitted alongside so the two are never
        compared without noticing.
        """
        if self.mat is None:
            return float("nan"), np.full(self.n, np.nan)
        mat = self.mat.reshape(self.n, self.n).cpu().numpy()
        inter = np.diag(mat).astype(np.float64)
        union = mat.sum(1) + mat.sum(0) - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        return float(np.nanmean(iou)), iou


def head_disagreement(trav_pred, terrain_pred, trav_map: dict, valid) -> tuple[int, int]:
    """Pixels where the traversability head contradicts the terrain head, and the total.

    The traversability target is a deterministic function of the terrain target, so the
    two heads should agree by construction -- but nothing in the model enforces it, and
    they are free to disagree at inference. "Glass, and walkable" is the failure this
    counts. It is reported rather than corrected, because which head should win is a
    deployment decision, not an evaluation one.
    """
    lut = torch.full((max(trav_map) + 1,), 255, dtype=torch.long, device=terrain_pred.device)
    for k, v in trav_map.items():
        lut[k] = v
    derived = lut[terrain_pred.clamp(max=len(lut) - 1)]
    both = valid & (derived != 255)
    return int((both & (derived != trav_pred)).sum()), int(both.sum())


def build_val_loaders(val_sets, cfg, device=None) -> list[tuple[str, DataLoader]]:
    """One DataLoader per validation set, built to be reused across epochs.

    Validation used to construct these inside ``evaluate``, so a 60-epoch run over two
    datasets spun up 120 worker pools and tore them down again. ``persistent_workers``
    only helps something that outlives a single pass, which is why the trainer builds
    these once in ``__init__`` and hands them back every epoch.
    """
    # Validation activations are smaller than training's -- no autograd graph is kept --
    # so this could be larger than the training batch. It is deliberately not: a smaller
    # batch keeps peak memory below the training peak, which means validation can never
    # be the thing that OOMs a run that was otherwise fitting.
    bs = max(int(cfg["train"]["batch_size"]) // 2, 1)
    workers = int(cfg["data"].get("workers", 4))
    pin = bool(device is not None and getattr(device, "type", None) == "cuda")
    return [
        (
            name,
            DataLoader(
                ds,
                batch_size=bs,
                shuffle=False,
                num_workers=workers,
                collate_fn=collate,
                pin_memory=pin,
                persistent_workers=workers > 0,
            ),
        )
        for name, ds in val_sets
    ]


@torch.no_grad()
def evaluate(
    model, val_sets, cfg, device, logger, samples: dict | None = None, loaders=None
) -> dict:
    """Return a metrics dict.

    If ``samples`` is a dict it is filled with the first validation batch per
    segmentation head as ``{head: (images, preds, targets)}``, which the trainer turns
    into TensorBoard comparison grids.

    ``loaders`` lets a caller that validates repeatedly build them once; without it they
    are built here, which is right for a one-shot ``hydranet-eval``.
    """
    # Restore whatever mode the caller had it in. Training passes the EMA copy, which
    # lives in eval mode; leaving it in train mode would let a stray forward pass move
    # its BatchNorm statistics, and hydranet-eval would return a model set to train.
    was_training = model.training
    model.eval()
    metrics: dict[str, float] = {}
    seg_cms: dict[str, ConfusionMatrix] = {}
    # Detections are kept per dataset. They used to share one list and one `coco_gt`
    # that each detection dataset overwrote, so with two of them every box was scored
    # against the last one's ground truth -- a wrong mAP, silently, and only when
    # someone added a second detection source.
    det_results: dict[str, list] = {}
    coco_gts: dict[str, object] = {}
    disagree = disagree_total = 0
    if loaders is None:
        loaders = build_val_loaders(val_sets, cfg, device)

    for (name, loader), (_, ds) in zip(loaders, val_sets, strict=True):
        sup = ds.supervises
        trav_map = getattr(getattr(ds, "scheme", None), "trav", None)
        for batch in loader:
            images = batch["image"].to(device)
            out = model(images)

            seg_preds: dict[str, torch.Tensor] = {}
            for head in model.seg_heads:
                if head in sup and head in batch["targets"]:
                    if head not in seg_cms:
                        seg_cms[head] = ConfusionMatrix(model.seg_heads[head].num_classes)
                    pred = out[head].argmax(dim=1)
                    tgt = batch["targets"][head].to(device)
                    seg_cms[head].update(pred, tgt)
                    seg_preds[head] = (pred, tgt)
                    if samples is not None and head not in samples:
                        k = min(4, images.shape[0])
                        samples[head] = (images[:k].cpu(), pred[:k].cpu(), tgt[:k].cpu())

            if trav_map and {"traversability", "terrain"} <= set(seg_preds):
                trav_pred, trav_tgt = seg_preds["traversability"]
                d, n = head_disagreement(
                    trav_pred, seg_preds["terrain"][0], trav_map, trav_tgt != 255
                )
                disagree += d
                disagree_total += n

            if model.det_head is not None and model.det_head_name in sup:
                coco_gts[name] = ds.coco
                results_for_ds = det_results.setdefault(name, [])
                dets = model.det_head.decode(
                    out["det_cls"],
                    out["det_reg"],
                    out["det_ctr"],
                    score_thr=SCORE_THR_EVAL,
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
                        results_for_ds.append(
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
        # How many classes that mean covers. Without it, a run on richer data looks like
        # a regression when it is really averaging over more, harder classes.
        metrics[f"{head}_mIoU_classes"] = float(np.isfinite(per_class).sum())
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

    for ds_name, results in det_results.items():
        if not results:
            continue
        from pycocotools.cocoeval import COCOeval

        coco_gt = coco_gts[ds_name]
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        # One detection dataset keeps the unqualified key, which is what
        # train.primary_metric and every existing metrics.jsonl refer to. A second one
        # gets its own suffixed keys rather than quietly redefining that number.
        suffix = "" if len(det_results) == 1 else f"/{ds_name}"
        metrics[f"detection_mAP{suffix}"] = float(ev.stats[0])
        metrics[f"detection_mAP50{suffix}"] = float(ev.stats[1])
        logger.info(
            f"[val] {ds_name} detection mAP = {ev.stats[0]:.4f}, mAP@50 = {ev.stats[1]:.4f}"
        )

    if disagree_total:
        frac = disagree / disagree_total
        metrics["head_disagreement"] = float(frac)
        logger.info(f"[val] traversability vs terrain disagree on {100 * frac:.2f}% of pixels")

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
