#!/usr/bin/env python3
"""Evaluate a trained crop-encoder checkpoint: per-attribute quality and embedding quality.

    python3 scripts/eval_attributes.py --checkpoint runs/rapv2_crop01/last.pt \
        --root datasets/RAP-v2/prepared --split val \
        --market datasets/Market-1501/Market-1501-v15.09.15

`train_attributes.py` logs `val_mean_accuracy` per epoch, and at the positive rates these
attributes actually have (`Hat` 1.15%, `Backpack` 1.38% on RAP v2) that number is
dominated by true negatives: a head that answers "no" to everything scores ~0.97 mean
accuracy while knowing nothing. This script reports what that number hides:

* **per attribute** -- precision / recall / F1 at threshold 0.5, plus AUROC, which is
  threshold-free and so separates "the model cannot rank positives above negatives" from
  "0.5 is the wrong operating point for a 1% attribute". Positive counts travel with every
  row, because an F1 on 23 positives and an F1 on 1,631 positives are not the same kind of
  evidence.
* **association** -- the same Market-1501 protocol `train_attributes.score_association`
  runs, imported from there rather than reimplemented so the number stays comparable by
  construction with the measured 0.0318 ImageNet floor and crop_encoder01's 0.0543.

AUROC is computed by the rank statistic (Mann-Whitney) in plain numpy: this repo keeps
scipy and sklearn out of the runtime dependency set, and ties get average ranks so a
constant-output channel scores 0.5 rather than an artifact of sort order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # for train_attributes, which owns the Market protocol

from train_attributes import score_association  # noqa: E402

from syncai_hydranet.data.attributes import ATTRIBUTES, PA100K  # noqa: E402
from syncai_hydranet.models.crop_encoder import (  # noqa: E402
    CropEncoder,
    load_crop_encoder,
)
from syncai_hydranet.utils.device import pick_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--checkpoint", required=True, help="crop-encoder .pt from train_attributes"
    )
    ap.add_argument("--root", default="datasets/RAP-v2/prepared", help="prepared parquet dir")
    ap.add_argument("--split", default="val", help="parquet split name (val, test, ...)")
    ap.add_argument(
        "--market",
        default=None,
        help="Market-1501 root; when given, also score the embedding on the ReID protocol",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", default=None, help="write the full result dict to this path")
    return ap


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney rank statistic, ties averaged.

    Returns nan when the split has no positives or no negatives -- an AUROC quoted on a
    class that is absent from the split would be a number about nothing.
    """
    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):  # average the ranks of tied scores
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


@torch.no_grad()
def score_attributes(model, loader, device) -> dict:
    """Per-attribute precision / recall / F1 at 0.5 plus AUROC, with positive counts."""
    model.eval()
    logits, targets = [], []
    for batch in loader:
        _, a = model(batch["image"].to(device, non_blocking=True))
        logits.append(a.float().cpu())
        targets.append(batch["labels"])
    p = torch.sigmoid(torch.cat(logits)).numpy()
    y = torch.cat(targets).numpy()
    pred = (p >= 0.5).astype(np.float32)
    out: dict = {"n": len(y), "mean_accuracy": float((pred == y).mean()), "attributes": {}}
    for i, name in enumerate(ATTRIBUTES):
        tp = float((pred[:, i] * y[:, i]).sum())
        fp = float((pred[:, i] * (1 - y[:, i])).sum())
        fn = float(((1 - pred[:, i]) * y[:, i]).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        out["attributes"][name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc(y[:, i], p[:, i]),
            "positives": int(y[:, i].sum()),
            "predicted_positives": int(pred[:, i].sum()),
        }
    return out


def load_encoder(path: str, device) -> CropEncoder:
    """The shared loader, with this script's attribute order as the contract.

    It used to build the model here and return it WITHOUT `.eval()`, so every number
    below was scored while BatchNorm was still updating from the batch being scored.
    """
    model, _ = load_crop_encoder(path, device, expect=ATTRIBUTES)
    return model


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = pick_device(None)
    model = load_encoder(args.checkpoint, device)

    ds = PA100K(args.root, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    print(f"{args.checkpoint} on {args.root}/{args.split}: {len(ds)} crops")
    result = score_attributes(model, loader, device)
    result["checkpoint"] = args.checkpoint
    result["root"] = args.root
    result["split"] = args.split

    print(f"mean accuracy {result['mean_accuracy']:.4f}  (dominated by negatives; see rows)")
    header = (
        f"{'attribute':<20s} {'pos':>6s} {'prec':>6s} {'recall':>6s} {'f1':>6s} {'auroc':>6s}"
    )
    print(header)
    print("-" * len(header))
    for name in ATTRIBUTES:
        m = result["attributes"][name]
        aur = f"{m['auroc']:.3f}" if not np.isnan(m["auroc"]) else "   nan"
        print(
            f"{name:<20s} {m['positives']:>6d} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f} {aur:>6s}"
        )

    if args.market:
        print(f"\nMarket-1501 protocol ({args.market}):")
        assoc = score_association(model, device, root=args.market)
        result["market"] = assoc
        print(
            f"mAP {assoc['mAP']:.4f}  rank1 {assoc['rank1']:.4f}  rank5 {assoc['rank5']:.4f}  "
            f"rank10 {assoc['rank10']:.4f}  ({assoc['queries_scored']} queries scored)"
        )
        print("floors: ImageNet resnet18 mAP 0.0318 / rank1 0.0962 (reid_metrics, 2026-08-17)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
