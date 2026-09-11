"""Cache frozen HydraNet features, then train a separate Trans10K glazing head.

The public validation split selects the checkpoint. It is not an untouched test set,
not StudioA footage, and not evidence that an opening's metric geometry is correct.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from syncai_bev3d.render_provenance import sha256
from syncai_hydranet.data.trans10k import GLAZING_CLASSES, glazing_labels
from syncai_hydranet.models.glazing import (
    GlazingEncoder,
    GlazingHead,
    confusion_scores,
    glazing_loss,
)
from syncai_hydranet.utils.visualize import preprocess


def cache(args):
    args.out.mkdir(parents=True, exist_ok=False)
    encoder = GlazingEncoder(args.config, args.encoder_checkpoint)
    sources = sorted(args.dataset.glob("*.parquet"))
    if not sources:
        raise ValueError("no parquet data found")
    provenance = {
        "schema": 1,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "encoder_checkpoint": str(args.encoder_checkpoint.resolve()),
        "encoder_sha256": sha256(args.encoder_checkpoint),
        "encoder_weights": "ema",
        "input_size": args.size,
        "classes": list(GLAZING_CLASSES),
        "sources": {str(p.resolve()): sha256(p) for p in sources},
        "duplicate_policy": "keep first content identity; ignore repeated images in all splits",
        "splits": {},
    }
    identities = {}
    for split, prefix in [("train", "train-"), ("validation", "validation-")]:
        paths = [p for p in sources if p.name.startswith(prefix)]
        total = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
        if not total:
            raise ValueError(f"missing {split} split")
        feature_map = target_map = None
        records = []
        start = time.monotonic()
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    index = len(records)
                    source = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                    raw_mask = Image.open(io.BytesIO(row["mask"]["bytes"]))
                    if source.size != raw_mask.size:
                        raise ValueError("image and mask dimensions disagree")
                    x, _canvas, region = preprocess(source, args.size)
                    left, top, width, height = region
                    # Resize RGB labels with nearest interpolation before palette decoding.
                    small = raw_mask.resize((width, height), Image.Resampling.NEAREST)
                    target = np.full(args.size, 255, np.uint8)
                    target[top : top + height, left : left + width] = glazing_labels(
                        np.asarray(small)
                    )
                    with torch.inference_mode():
                        features = encoder(x)[0].numpy().astype(np.float16)
                    if feature_map is None:
                        feature_map = np.lib.format.open_memmap(
                            args.out / f"{split}.features.npy",
                            mode="w+",
                            dtype=np.float16,
                            shape=(total, *features.shape),
                        )
                        target_map = np.lib.format.open_memmap(
                            args.out / f"{split}.labels.npy",
                            mode="w+",
                            dtype=np.uint8,
                            shape=(total, *features.shape[-2:]),
                        )
                    target_small = np.array(
                        Image.fromarray(target).resize(
                            features.shape[-2:][::-1], Image.Resampling.NEAREST
                        )
                    )
                    identity = hashlib.sha256(row["image"]["bytes"]).hexdigest()
                    duplicate = identities.get(identity)
                    if duplicate is not None:
                        target_small[:] = 255
                    else:
                        identities[identity] = {"split": split, "image": row["image"]["path"]}
                    feature_map[index] = features
                    assert target_map is not None
                    target_map[index] = target_small
                    records.append(
                        {
                            "image": row["image"]["path"],
                            "sha256": identity,
                            "source_size": source.size,
                            "region": region,
                            "excluded_duplicate_of": duplicate,
                        }
                    )
                    if index % 250 == 0:
                        print(
                            f"{split} {index}/{total}, {time.monotonic() - start:.1f}s",
                            flush=True,
                        )
        assert feature_map is not None and target_map is not None
        feature_map.flush()
        target_map.flush()
        pixels = np.bincount(np.asarray(target_map).ravel(), minlength=256)
        provenance["splits"][split] = {
            "count": total,
            "unique_images_used": sum(r["excluded_duplicate_of"] is None for r in records),
            "feature_shape": list(feature_map.shape),
            "class_pixels": pixels[:4].tolist(),
            "ignored_pixels": int(pixels[255]),
            "features_sha256": sha256(args.out / f"{split}.features.npy"),
            "labels_sha256": sha256(args.out / f"{split}.labels.npy"),
        }
        (args.out / f"{split}.images.json").write_text(json.dumps(records) + "\n")
        print("CACHED", split, total, flush=True)
    (args.out / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


def validate(head, features, labels, batch_size):
    head.eval()
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(
                np.array(features[start : start + batch_size], dtype=np.float32)
            )
            target = torch.from_numpy(
                np.array(labels[start : start + batch_size], dtype=np.int64)
            )
            prediction = head(x).argmax(1)
            valid = target != 255
            confusion += torch.bincount(
                (target[valid] * 4 + prediction[valid]).flatten(), minlength=16
            ).reshape(4, 4)
    return confusion_scores(confusion)


def train(args):
    manifest_path = args.cache / "manifest.json"
    spec = json.loads(manifest_path.read_text())
    if spec["classes"] != list(GLAZING_CLASSES):
        raise ValueError("cache class ordering differs")
    arrays = {}
    for split in ["train", "validation"]:
        for kind in ["features", "labels"]:
            path = args.cache / f"{split}.{kind}.npy"
            if sha256(path) != spec["splits"][split][f"{kind}_sha256"]:
                raise ValueError(f"changed feature cache: {path}")
            arrays[split, kind] = np.load(path, mmap_mode="r")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    head = GlazingHead(spec["splits"]["train"]["feature_shape"][1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    counts = torch.tensor(spec["splits"]["train"]["class_pixels"], dtype=torch.float32)
    weights = (counts.sum() / counts.clamp_min(1)).sqrt()
    weights = weights / weights[0]
    weights = weights.clamp(max=10)
    run = {
        "cache": str(args.cache.resolve()),
        "cache_manifest_sha256": sha256(manifest_path),
        "encoder": spec,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "class_weights": weights.tolist(),
        "selection": "public validation foreground mIoU at feature resolution",
        "limitations": "frozen encoder; no StudioA training; validation selects checkpoint; "
        "not an independent test or metric-scale validation",
    }
    (args.out / "config.json").write_text(json.dumps(run, indent=2) + "\n")
    best = -1
    usable = np.flatnonzero(np.any(arrays["train", "labels"] != 255, axis=(1, 2)))
    start = time.monotonic()
    for epoch in range(args.epochs + 1):
        losses = []
        if epoch:
            head.train()
            order = rng.permutation(usable)
            for offset in range(0, len(order), args.batch_size):
                indices = order[offset : offset + args.batch_size]
                x = torch.from_numpy(
                    np.array(arrays["train", "features"][indices], dtype=np.float32)
                )
                y = torch.from_numpy(
                    np.array(arrays["train", "labels"][indices], dtype=np.int64)
                )
                optimizer.zero_grad(set_to_none=True)
                loss = glazing_loss(head(x), y, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach()))
        metrics = validate(
            head,
            arrays["validation", "features"],
            arrays["validation", "labels"],
            args.batch_size,
        )
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else None,
            "seconds": time.monotonic() - start,
            **metrics,
        }
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if epoch and metrics["foreground_miou"] > best:
            best = metrics["foreground_miou"]
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "spec": run,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                args.out / "best.pt",
            )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "spec": run,
            "epoch": args.epochs,
            "metrics": metrics,
        },
        args.out / "last.pt",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("cache")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--encoder-checkpoint", type=Path, required=True)
    prepare.add_argument("--size", type=int, nargs=2, default=[256, 384])
    prepare.add_argument("--out", type=Path, required=True)
    fit = actions.add_parser("train")
    fit.add_argument("--cache", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=12)
    fit.add_argument("--batch-size", type=int, default=32)
    fit.add_argument("--lr", type=float, default=0.001)
    fit.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.action == "cache" and (min(args.size) < 64 or any(s % 32 for s in args.size)):
        parser.error("input size must be multiples of 32, at least 64")
    if args.action == "train" and (min(args.epochs, args.batch_size) < 1 or args.lr <= 0):
        parser.error("epochs, batch size and learning rate must be positive")
    torch.set_num_threads(args.threads)
    (cache if args.action == "cache" else train)(args)


if __name__ == "__main__":
    main()
