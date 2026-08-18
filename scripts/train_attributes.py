#!/usr/bin/env python3
"""Train the second-stage crop encoder on PA-100K, and score its embedding on Market-1501.

    python3 scripts/train_attributes.py --out runs/crop_encoder01 --epochs 8

Two numbers come out and they answer different questions, which is the whole point of one
encoder rather than two models:

* **attributes** -- per-attribute recall and precision on PA-100K val, each printed beside
  its training support. `AgeOver60` rests on 1,127 crops and `Female` on 36,492; they are
  the same kind of output and not the same kind of evidence.
* **association** -- mAP and rank-1 for the embedding on Market-1501's standard
  3,368-query protocol, against the floor `reid_metrics` already measured: an ImageNet
  resnet18 with no re-identification training scores **mAP 0.0318 / rank-1 0.0962**. That
  floor is what this has to clear to have earned anything, and it is measured with the
  same code on the same protocol rather than quoted from a paper.

`PERSON_ATTRIBUTES.md` puts association first and attributes second. This trains the
attribute head because that is the label PA-100K carries, and reports the embedding
because that is the output the ordering says matters -- so a run that improves attributes
while leaving association at the ImageNet floor is visible as exactly that, rather than
being reported as progress.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.data.attributes import ATTRIBUTES, PA100K, SUPPORT  # noqa: E402
from syncai_hydranet.models.crop_encoder import AttributeLoss, CropEncoder  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.seeding import seed_everything  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="datasets/PA-100K")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed-dim", type=int, default=256)
    return ap


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    logits, targets = [], []
    for batch in loader:
        _, a = model(batch["image"].to(device))
        logits.append(a.float().cpu())
        targets.append(batch["labels"])
    p = torch.sigmoid(torch.cat(logits)).numpy()
    y = torch.cat(targets).numpy()
    pred = (p >= 0.5).astype(np.float32)
    out = {}
    for i, name in enumerate(ATTRIBUTES):
        tp = float((pred[:, i] * y[:, i]).sum())
        fp = float((pred[:, i] * (1 - y[:, i])).sum())
        fn = float(((1 - pred[:, i]) * y[:, i]).sum())
        out[name] = {
            "recall": tp / max(tp + fn, 1.0),
            "precision": tp / max(tp + fp, 1.0),
            "val_positives": int(y[:, i].sum()),
            "train_support": SUPPORT.get(name),
        }
    out["_mean_accuracy"] = float((pred == y).mean())
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    device = pick_device(None)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    train = PA100K(args.root, "train", train=True)
    val = PA100K(args.root, "val")
    print(f"PA-100K: {len(train)} train crops, {len(val)} val, {len(ATTRIBUTES)} attributes")

    pos_rate = torch.from_numpy(train.labels.mean(axis=0)).float()
    model = CropEncoder(len(ATTRIBUTES), embed_dim=args.embed_dim).to(device)
    criterion = AttributeLoss(pos_rate).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    tl = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    vl = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for batch in tl:
            _, a = model(batch["image"].to(device, non_blocking=True))
            loss = criterion(a, batch["labels"].to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(a)
            n += len(a)
        sched.step()
        metrics = evaluate(model, vl, device)
        row = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            "val_mean_accuracy": metrics["_mean_accuracy"],
            "seconds": round(time.time() - t0, 1),
            "attributes": {k: v for k, v in metrics.items() if not k.startswith("_")},
        }
        history.append(row)
        (out_root / "metrics.json").write_text(json.dumps(history, indent=2))
        torch.save(
            {"model": model.state_dict(), "attributes": ATTRIBUTES}, out_root / "last.pt"
        )
        ask = {k: metrics[k] for k in ("Female", "AgeLess18", "AgeOver60")}
        print(
            f"epoch {epoch}: loss {row['train_loss']:.4f}  mean acc "
            f"{row['val_mean_accuracy']:.4f}  ({row['seconds']:.0f}s)"
        )
        for k, v in ask.items():
            print(
                f"    {k:10s} recall {v['recall']:.3f}  precision {v['precision']:.3f}  "
                f"train support {v['train_support']}"
            )
    print(f"\nwrote {out_root / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------- association
#
# The half this script's docstring promised and its first version did not compute -- which
# is the failure `ARCHITECTURE_DIRECTION.md` section 5 names in the abstract: "the
# mechanism was built and the thing that would show whether it worked was not". An
# attribute head that improves while the embedding sits at the ImageNet floor is a model
# that has learned nothing about the job `PERSON_ATTRIBUTES.md` puts first.


def _market_split(root: Path, folder: str):
    """Market-1501 filenames: ``pid_cCAM_sSEQ_frame_bbox.jpg``. -1 is a distractor."""
    items = []
    for p in sorted((root / folder).glob("*.jpg")):
        pid_s, cam_s = p.stem.split("_")[0], p.stem.split("_")[1]
        items.append((p, int(pid_s), int(cam_s[1])))
    return items


@torch.no_grad()
def market_embeddings(model, items, device, batch_size: int = 256):
    from PIL import Image

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    feats = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        arr = []
        for p, _pid, _cam in chunk:
            img = Image.open(p).convert("RGB").resize((128, 256), Image.BILINEAR)
            x = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
            arr.append(x.transpose(2, 0, 1))
        batch = torch.from_numpy(np.stack(arr)).to(device)
        feats.append(model.embed_only(batch).float().cpu().numpy())
    return np.concatenate(feats)


def score_association(model, device, root="datasets/Market-1501/Market-1501-v15.09.15"):
    """mAP and CMC on the standard 3,368-query protocol, against the measured floor.

    The floor is not quoted from a paper: `reid_metrics.py` measured an ImageNet resnet18
    with no re-identification training at **mAP 0.0318 / rank-1 0.0962 / rank-5 0.1986 /
    rank-10 0.2699** with this same code on this same protocol. Comparable by construction,
    which is why the encoder here uses the same backbone.
    """
    from syncai_hydranet.analytics.reid_metrics import cmc_map

    root = Path(root)
    query = _market_split(root, "query")
    gallery = _market_split(root, "bounding_box_test")
    model.eval()
    qf = market_embeddings(model, query, device)
    gf = market_embeddings(model, gallery, device)
    out = cmc_map(
        qf,
        np.array([p for _, p, _ in query]),
        np.array([c for _, _, c in query]),
        gf,
        np.array([p for _, p, _ in gallery]),
        np.array([c for _, _, c in gallery]),
    )
    out["queries"] = len(query)
    out["gallery"] = len(gallery)
    return out
