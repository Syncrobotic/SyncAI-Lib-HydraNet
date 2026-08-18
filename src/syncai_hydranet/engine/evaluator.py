"""Validation: per-head segmentation mIoU and COCO detection mAP."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.multitask import collate
from ..data.transforms import invert_geom
from ..models.heads.detection import SCORE_THR_EVAL
from ..utils.seeding import model_memory_format


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
        # Every class NaN -- a batch that was entirely `ignore`, which the loss also
        # skips -- is a real state, not an error, so it returns NaN rather than warning
        # about it. `np.nanmean` of an all-NaN slice emits a RuntimeWarning and returns
        # the same NaN; taking the branch explicitly means the suite can run with
        # `-W error` and this stays a value rather than becoming a failure.
        if not np.isfinite(iou).any():
            return float("nan"), iou
        return float(np.nanmean(iou)), iou

    def predicted(self) -> np.ndarray:
        """Predicted pixels per class: the column sums, against `support`'s row sums.

        Only interesting where the two disagree in kind. A class with zero support and
        non-zero prediction in one val set is emitting pure false positives there, and
        pooling carries them into a set where the class is real.
        """
        if self.mat is None:
            return np.zeros(self.n, dtype=np.int64)
        return self.mat.reshape(self.n, self.n).sum(0).cpu().numpy()

    def support(self) -> np.ndarray:
        """Ground-truth pixels per class: how much evidence each IoU above stands on.

        An IoU is reported identically whether it was computed over 61% of the labelled
        pixels or over 0.66% of them, and the second kind has cost this project a run.
        Measured on the ADE20K splits the retail-objects config evaluates: `wall` covers
        283 of 285 val images and 61.86% of labelled pixels, while `column` -- the class
        that taxonomy was created to separate out of `wall` -- covers 22 images and
        **0.66%**. It scored 0.40-0.51 there and then predicted 0.00% of pixels on four
        daytime store cameras. Its whole ep25->ep60 decay of 0.510 to 0.400 is a swing
        across 22 images.

        Row sums rather than the IoU denominator on purpose: `union` mixes in the model's
        false positives, so a class the model over-predicts would look better evidenced
        than it is. What is wanted here is how much of the class was there to find, which
        is a property of the data alone and therefore constant across epochs.
        """
        if self.mat is None:
            return np.zeros(self.n, dtype=np.int64)
        return self.mat.reshape(self.n, self.n).cpu().numpy().sum(1)


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


def _seg_metrics_per_dataset(seg_cms_by_ds, cfg, logger) -> dict:
    """The same segmentation metrics again, per validation dataset.

    **The pooled numbers alone cost this project a day.** `evaluate` accumulates one
    confusion matrix per head across every val set, so a class that cannot exist in one of
    them still collects false positives there, and they land in the same denominator as
    the true positives from the set where it can. Measured on
    `runs/hydranet_retail_objects_site_balanced`: `product` scored **0.583 on the store's
    own val frames and 0.000 on ADE20K's**, where its ground truth is necessarily zero and
    every predicted pixel is a false positive with nothing to offset it. Pooled, that
    reads as 0.05-0.35 and looks like a model failing to learn.

    Detection already solved this -- see the suffix convention in `_det_metrics`, and the
    comment there about a second dataset silently redefining the first one's number. This
    is the same fix on the segmentation side, and it keeps the unqualified keys untouched
    so every existing `metrics.jsonl` and `train.primary_metric` still means what it did.

    A run whose reason for existing is one dataset should select on that dataset:

        primary_metric: IoU/terrain/05_product/site_sam3
    """
    heads = {head for head, _ in seg_cms_by_ds}
    trav_names = ("blocked", "caution", "go")
    terrain_names = tuple(cfg["data"].get("terrain_classes", []))
    metrics: dict[str, float] = {}
    for head in sorted(heads):
        datasets = sorted(ds for h, ds in seg_cms_by_ds if h == head)
        if len(datasets) < 2:
            # One dataset: the pooled number already is the per-dataset one, and emitting
            # both would double every key for no information.
            continue
        names = trav_names if head == "traversability" else terrain_names
        for ds_name in datasets:
            cm = seg_cms_by_ds[(head, ds_name)]
            miou, per_class = cm.miou()
            metrics[f"{head}_mIoU/{ds_name}"] = miou
            support, predicted = cm.support(), cm.predicted()
            uncontested = []
            for i, iou in enumerate(per_class):
                cname = names[i] if i < len(names) else str(i)
                if np.isfinite(iou):
                    metrics[f"IoU/{head}/{i:02d}_{cname}/{ds_name}"] = float(iou)
                # Ground truth cannot contain it, yet the model paints it here. Every one
                # of those pixels is a false positive that no true positive in this set
                # can offset, and pooling carries them into another set's IoU.
                if support[i] == 0 and predicted[i] > 0:
                    uncontested.append(f"{cname} {int(predicted[i]):,}px")
            logger.info(f"[val] {head} mIoU on {ds_name} = {miou:.4f}")
            if uncontested:
                logger.warning(
                    f"[val] {ds_name} has no ground truth for {', '.join(uncontested)} "
                    f"but the model predicts them there. Those are false positives with "
                    f"nothing in this set to offset them, and the pooled "
                    f"IoU/{head}/... carries them into the set where the class does "
                    f"exist. Select and report on the per-dataset key instead."
                )
    return metrics


@torch.no_grad()
def _update_seg_heads(
    model, out, batch, sup, device, seg_cms, samples, images, seg_cms_by_ds=None, ds_name=""
) -> dict:
    """Fold one batch into the per-head confusion matrices.

    Returns `{head: (pred, target)}` for the heads this batch supervised, which the
    disagreement check needs and which the caller cannot recompute without redoing the
    argmax.
    """
    seg_preds: dict[str, tuple] = {}
    for head in model.seg_heads:
        if head not in sup or head not in batch["targets"]:
            continue
        if head not in seg_cms:
            seg_cms[head] = ConfusionMatrix(model.seg_heads[head].num_classes)
        pred = out[head].argmax(dim=1)
        tgt = batch["targets"][head].to(device)
        seg_cms[head].update(pred, tgt)
        if seg_cms_by_ds is not None:
            key = (head, ds_name)
            if key not in seg_cms_by_ds:
                seg_cms_by_ds[key] = ConfusionMatrix(model.seg_heads[head].num_classes)
            seg_cms_by_ds[key].update(pred, tgt)
        seg_preds[head] = (pred, tgt)
        if samples is not None and head not in samples:
            k = min(4, images.shape[0])
            samples[head] = (images[:k].cpu(), pred[:k].cpu(), tgt[:k].cpu())
    return seg_preds


def _collect_detections(model, out, batch, ds, images, results_for_ds) -> None:
    """Decode one batch's boxes and append them in COCO result form."""
    dets = model.det_head.decode(
        out["det_cls"],
        out["det_reg"],
        out["det_ctr"],
        score_thr=SCORE_THR_EVAL,
        nms_thr=0.6,
        img_size=images.shape[-2:],
    )
    # Map boxes back to original image coordinates for COCOeval. The dataset records the
    # exact scale and padding it applied; deriving it here from the frame size would be
    # wrong for every letterboxed image, because the padding is not part of the original
    # frame.
    for img_id, det, geom in zip(batch["image_ids"], dets, batch["geoms"], strict=True):
        boxes = invert_geom(det["boxes"].cpu().numpy(), geom)
        for box, score, label in zip(
            boxes, det["scores"].cpu().numpy(), det["labels"].cpu().numpy(), strict=True
        ):
            # A shared `det_vocab` gives the head channels this dataset's ground truth
            # has no category for -- `person` predicted on a site frame whose annotation
            # file holds only merchandise. `.get` drops those, and dropping is exactly
            # what COCOeval would do with them anyway, since `params.catIds` is already
            # restricted to `score_cat_ids`. The number is unchanged; the KeyError is not.
            #
            # What it means is worth carrying: with two sources, each class's mAP is
            # measured on the one dataset that labels it, so `person` is scored on COCO
            # and never on the store cameras it will be deployed against. Site person
            # boxes are the only thing that changes that.
            cat_id = ds.label_to_cat.get(int(label))
            if cat_id is None:
                continue
            results_for_ds.append(
                {
                    "image_id": int(img_id),
                    "category_id": cat_id,
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1]),
                    ],
                    "score": float(score),
                }
            )


# Below this share of labelled ground-truth pixels, a per-class IoU is called out as
# standing on too little to be a measurement. A tripwire, not a statistical threshold:
# it is set where it separates the one class known to have failed this way from every
# other sourced class in the same taxonomy. On the ADE20K val split the retail-objects
# run evaluated, `column` sits at 0.66% (22 of 285 images) and scored 0.40-0.51 while
# predicting 0.00% of pixels on store cameras; `person`, the next thinnest, is at 1.70%.
# Anything under this wants site confirmation before the number is repeated.
THIN_SUPPORT = 0.01


def _seg_metrics(seg_cms, cfg, logger) -> dict:
    """mIoU per head, plus every class separately, and what each class stands on.

    The per-class breakdown is not detail for its own sake: the safety-critical indoor
    classes (glass, stairs) are tiny, so a mean alone hides the fact that they never
    converged. `support/` is the same argument one step further -- a tiny class does not
    only converge badly, it also produces an IoU that looks exactly like a well-evidenced
    one in the log and in metrics.jsonl.
    """
    trav_names = ("blocked", "caution", "go")
    terrain_names = tuple(cfg["data"].get("terrain_classes", []))
    metrics: dict[str, float] = {}
    for head, cm in seg_cms.items():
        miou, per_class = cm.miou()
        metrics[f"{head}_mIoU"] = miou
        # How many classes that mean covers. Without it, a run on richer data looks like
        # a regression when it is really averaging over more, harder classes.
        metrics[f"{head}_mIoU_classes"] = float(np.isfinite(per_class).sum())
        support = cm.support()
        total = float(support.sum()) or 1.0
        names = trav_names if head == "traversability" else terrain_names
        thin = []
        for i, iou in enumerate(per_class):
            if not np.isfinite(iou):
                continue
            cname = names[i] if i < len(names) else str(i)
            share = float(support[i]) / total
            metrics[f"IoU/{head}/{i:02d}_{cname}"] = float(iou)
            # Constant across epochs -- it describes the split, not the model. Emitted
            # every epoch anyway so that an archived metrics.jsonl stays readable once
            # the dataset behind it is gone, which is the same reason runs fingerprint
            # their splits rather than trusting the path to still mean something.
            metrics[f"support/{head}/{i:02d}_{cname}"] = share
            # `void` is the ignore class and nobody trains it, so "confirm this on site"
            # is not advice anyone can act on. It reaches here at all because the model
            # sometimes *predicts* class 0 where no target has it: union > 0, so the IoU
            # is a finite 0.000 rather than NaN. That is worth keeping as a metric -- a
            # head hallucinating the ignore class is real -- but it is not a thin
            # measurement, and it fired on the live batch02 run as the only warning of
            # the epoch. Excluded by name rather than by index: the traversability head's
            # class 0 is `blocked`, which is a real class and must still be checked.
            if share < THIN_SUPPORT and cname != "void":
                thin.append(f"{cname} {iou:.3f} on {100 * share:.2f}%")
        logger.info(
            f"[val] {head} mIoU = {miou:.4f} | per-class IoU = "
            + " ".join(f"{x:.3f}" if np.isfinite(x) else "-" for x in per_class)
        )
        if thin:
            logger.warning(
                f"[val] {head}: {', '.join(thin)} of labelled pixels. A per-class IoU "
                f"over that little evidence is not a measurement of the class -- read it "
                f"as 'not measured' and confirm on site before reporting it."
            )
    return metrics


def _det_metrics(det_results, coco_gts: dict[str, Any], det_cat_ids, logger) -> dict:
    """COCO mAP per detection dataset."""
    metrics: dict[str, float] = {}
    for ds_name, results in det_results.items():
        suffix = "" if len(det_results) == 1 else f"/{ds_name}"
        if not results:
            # A detector that predicted nothing has an mAP, and it is zero. Skipping the
            # key instead says "not measured", and the difference is not cosmetic: it
            # killed `hydranet_retail_surfaces_seed7` at epoch 1 with
            # `primary_metric='detection_mAP' was not produced by validation`, because an
            # untrained head happened to clear no boxes on that seed while seed 42's
            # cleared six. A seed-variance experiment must not be destroyed by seed
            # variance in a quantity it is not measuring.
            metrics[f"detection_mAP{suffix}"] = 0.0
            metrics[f"detection_mAP50{suffix}"] = 0.0
            logger.info(
                f"[val] {ds_name} detection mAP = 0.0000 -- the model returned no boxes "
                f"at all on this set, which is a score rather than a missing measurement"
            )
            continue
        from pycocotools.cocoeval import COCOeval

        coco_gt = coco_gts[ds_name]
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        # Score only the categories this dataset trains on. Without it a model trained on
        # a subset is measured against all 80 and every absent class counts as a miss;
        # worse in the other direction, comparing a 15-class model's mAP to an 80-class
        # model's rewards the smaller one for a easier denominator rather than for being
        # better. Restricting both to the same categories is what makes them comparable.
        if det_cat_ids.get(ds_name):
            ev.params.catIds = list(det_cat_ids[ds_name])
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        # `suffix` is set at the top of the loop: one detection dataset keeps the
        # unqualified key, which is what train.primary_metric and every existing
        # metrics.jsonl refer to. A second one gets its own suffixed keys rather than
        # quietly redefining that number.
        metrics[f"detection_mAP{suffix}"] = float(ev.stats[0])
        metrics[f"detection_mAP50{suffix}"] = float(ev.stats[1])
        logger.info(
            f"[val] {ds_name} detection mAP = {ev.stats[0]:.4f}, mAP@50 = {ev.stats[1]:.4f}"
        )
    return metrics


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
    # The same matrices again, split by validation dataset. Pooling them is right for a
    # headline number and wrong for a class that only one of the datasets can contain --
    # see `_seg_metrics_per_dataset`, which is where that cost is written down.
    seg_cms_by_ds: dict[tuple[str, str], ConfusionMatrix] = {}
    # Detections are kept per dataset. They used to share one list and one `coco_gt`
    # that each detection dataset overwrote, so with two of them every box was scored
    # against the last one's ground truth -- a wrong mAP, silently, and only when
    # someone added a second detection source.
    det_results: dict[str, list] = {}
    # `Any`, not `object`: pycocotools ships no stubs, so the COCO handle genuinely has
    # no type here. Declaring `object` claimed more than was known and made every
    # `coco_gt.loadRes(...)` an error against a class nothing can describe.
    coco_gts: dict[str, Any] = {}
    # The categories each dataset actually trains on, so mAP is scored over those and
    # not over all 80 with the untrained ones counted as misses.
    det_cat_ids: dict[str, list] = {}
    disagree = disagree_total = 0
    if loaders is None:
        loaders = build_val_loaders(val_sets, cfg, device)

    # Read off the model rather than the config: a checkpoint does not record its
    # memory format, so a caller who loaded weights into an NCHW module gets NCHW here
    # and one training in NHWC gets NHWC, without either having to say so.
    memory_format = model_memory_format(model)

    for (name, loader), (_, ds) in zip(loaders, val_sets, strict=True):
        sup = ds.supervises
        trav_map = getattr(getattr(ds, "scheme", None), "trav", None)
        for batch in loader:
            images = batch["image"].to(device).contiguous(memory_format=memory_format)
            out = model(images)

            seg_preds = _update_seg_heads(
                model,
                out,
                batch,
                sup,
                device,
                seg_cms,
                samples,
                images,
                seg_cms_by_ds=seg_cms_by_ds,
                ds_name=name,
            )

            if trav_map and {"traversability", "terrain"} <= set(seg_preds):
                trav_pred, trav_tgt = seg_preds["traversability"]
                d, n = head_disagreement(
                    trav_pred, seg_preds["terrain"][0], trav_map, trav_tgt != 255
                )
                disagree += d
                disagree_total += n

            if model.det_head is not None and model.det_head_name in sup:
                coco_gts[name] = ds.coco
                det_cat_ids[name] = list(getattr(ds, "score_cat_ids", None) or [])
                _collect_detections(
                    model, out, batch, ds, images, det_results.setdefault(name, [])
                )

    metrics.update(_seg_metrics(seg_cms, cfg, logger))
    metrics.update(_seg_metrics_per_dataset(seg_cms_by_ds, cfg, logger))
    metrics.update(_det_metrics(det_results, coco_gts, det_cat_ids, logger))

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
