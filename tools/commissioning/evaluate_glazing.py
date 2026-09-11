"""Compare glazing checkpoints at original validation-image resolution.

The public validation set selects checkpoints; this is not independent test accuracy.
Check cache/source identities before scoring, and exclude the cache's repeated images.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.nn import functional as F

from syncai_bev3d.render_provenance import sha256
from syncai_hydranet.data.trans10k import GLAZING_CLASSES, glazing_labels
from syncai_hydranet.models.glazing import GlazingHead, confusion_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.out.exists():
        parser.error("output already exists")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    run = checkpoint["spec"]
    spec = run["encoder"]
    if sha256(args.cache / "manifest.json") != run["cache_manifest_sha256"]:
        raise ValueError("cache manifest differs from training")
    if spec["classes"] != list(GLAZING_CLASSES):
        raise ValueError("class ordering differs")
    path = args.cache / "validation.features.npy"
    if sha256(path) != spec["splits"]["validation"]["features_sha256"]:
        raise ValueError("validation features changed")
    sources = sorted(args.validation)
    for source in sources:
        if sha256(source) not in spec["sources"].values():
            raise ValueError("validation source differs from cache")
    features = np.load(path, mmap_mode="r")
    records = json.loads((args.cache / "validation.images.json").read_text())
    if len(features) != len(records):
        raise ValueError("cache image count differs")
    head = GlazingHead(features.shape[1]).eval()
    head.load_state_dict(checkpoint["state_dict"])
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    per_image = []
    index = 0
    for source in sources:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                record = records[index]
                if record["image"] != row["image"]["path"] or record["sha256"] != (
                    hashlib.sha256(row["image"]["bytes"]).hexdigest()
                ):
                    raise ValueError("validation record identity differs")
                index += 1
                if record["excluded_duplicate_of"] is not None:
                    continue
                truth = glazing_labels(np.array(Image.open(io.BytesIO(row["mask"]["bytes"]))))
                h, w = truth.shape
                if record["source_size"] != [w, h]:
                    raise ValueError("validation dimensions differ")
                target = torch.from_numpy(truth.astype(np.int64))
                valid = target != 255
                left, top, width, height = record["region"]
                with torch.inference_mode():
                    x = torch.from_numpy(np.array(features[index - 1 : index], np.float32))
                    logits = F.interpolate(
                        head(x), size=spec["input_size"], mode="bilinear", align_corners=False
                    )
                    logits = F.interpolate(
                        logits[:, :, top : top + height, left : left + width],
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    prediction = logits.argmax(1)[0]
                    confusion += torch.bincount(
                        target[valid] * 4 + prediction[valid], minlength=16
                    ).reshape(4, 4)
                door = (prediction == 1) & valid
                reference = (target == 1) & valid
                union = int((door | reference).sum())
                per_image.append(
                    {
                        "image": record["image"],
                        "door_iou": int((door & reference).sum()) / union if union else None,
                        "door_predicted_pixels": int(door.sum()),
                    }
                )
                if index % 100 == 0:
                    print(f"Scored {index}/{len(records)} source images", flush=True)
    if index != len(records):
        raise ValueError("validation sources are incomplete")
    result = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "selected_epoch": checkpoint["epoch"],
        "unique_validation_images": len(per_image),
        "input_size": spec["input_size"],
        "scoring_resolution": "original source image after inverse letterbox; argmax",
        "selection_limitation": "Validation selects checkpoints; not an independent test.",
        "trained": confusion_scores(confusion),
        "images": per_image,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "images"}), flush=True)


if __name__ == "__main__":
    main()
