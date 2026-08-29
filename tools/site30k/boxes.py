#!/usr/bin/env python3
"""Detection labels for the small batch: draw them, and write them as COCO.

The campaign taxonomy carries THREE detection classes, and they come from different
teachers, so this script keeps them apart instead of inventing one pipeline:

  person   Grounding DINO @0.35 on the frame -- the measured day/night threshold
           (PERSON_TRAIN_THR in campaign_site30k), NMS 0.55. Run here, not inherited:
           the batch driver only needed person PIXELS, so it never produced boxes.
  product  the batch's own SAM 3 product pixels, boxed per connected component. The
           source family is kept per box (laptop / tablet / phone / boxed_stock), which
           is what the campaign merges into one `product` category.
  pet      NOT DRAWN. The census (runs/site30k_qa/pet_census.json, 2477 boxes over 336
           frames) tops out at 0.5427 with p99 = 0.2018, and no threshold has been chosen
           yet -- so there is nothing to draw that would mean anything.

Usage: batch30_boxes.py <batch_dir> [out_dir]
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, "/home/paul/SyncAI-Lib-HydraNet/src")
sys.path.insert(0, "/home/paul/SyncAI-Lib-HydraNet/scripts")
spec = importlib.util.spec_from_file_location(
    "campaign", "/home/paul/SyncAI-Lib-HydraNet/scripts/campaign_site30k.py"
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

BATCH = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else BATCH / "boxes"
OUT.mkdir(parents=True, exist_ok=True)

PRODUCT_IDS = {7: "laptop", 8: "tablet", 9: "phone", 10: "boxed_stock"}
BOX_COLORS = {
    "person": (230, 25, 75),
    "laptop": (140, 60, 255),
    "tablet": (255, 0, 150),
    "phone": (255, 140, 0),
    "boxed_stock": (170, 110, 40),
}
MIN_PRODUCT_PX = 300

device = (
    M.torch.device if False else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
)
gd_proc, gd_model = M.load_gdino("IDEA-Research/grounding-dino-base", device)

coco = {
    "info": {
        "source": str(BATCH),
        "person": f"Grounding DINO @{M.PERSON_TRAIN_THR}, NMS {M.NMS_IOU}",
        "product": "SAM 3 product pixels of the batch mask, boxed per component",
        "pet": "not labelled: no threshold chosen (pet_census.json)",
    },
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "pet"},
        {"id": 3, "name": "product"},
    ],
    "images": [],
    "annotations": [],
}
aid = 0
report = []
for iid, mask_path in enumerate(sorted((BATCH / "masks").glob("*.png")), start=1):
    name = mask_path.stem
    img = Image.open(BATCH / "preview" / f"{name}_raw.jpg").convert("RGB")
    mask = np.asarray(Image.open(mask_path))
    coco["images"].append(
        {"id": iid, "file_name": f"{name}.jpg", "width": img.width, "height": img.height}
    )

    boxes = []
    raw = M.gdino_detect(gd_proc, gd_model, img, "person", M.SCORE_FLOOR, device)
    kept = M.gdino_nms(raw[raw[:, 4] >= M.PERSON_TRAIN_THR], M.NMS_IOU) if len(raw) else raw
    for b in kept:
        boxes.append(
            ("person", float(b[4]), [float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        )
        aid += 1
        coco["annotations"].append(
            {
                "id": aid,
                "image_id": iid,
                "category_id": 1,
                "score": float(b[4]),
                "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                "area": float((b[2] - b[0]) * (b[3] - b[1])),
                "iscrowd": 0,
            }
        )

    for cid, fam in PRODUCT_IDS.items():
        lb, n = ndimage.label(mask == cid)
        for k in range(1, n + 1):
            ys, xs = np.where(lb == k)
            if len(ys) < MIN_PRODUCT_PX:
                continue
            x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            boxes.append((fam, None, [x0, y0, x1, y1]))
            aid += 1
            coco["annotations"].append(
                {
                    "id": aid,
                    "image_id": iid,
                    "category_id": 3,
                    "family": fam,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": float(len(ys)),
                    "iscrowd": 0,
                }
            )

    vis = img.copy()
    d = ImageDraw.Draw(vis)
    for fam, score, (x0, y0, x1, y1) in boxes:
        col = BOX_COLORS[fam]
        d.rectangle([x0, y0, x1, y1], outline=col, width=3 if fam == "person" else 2)
        tag = f"{fam} {score:.2f}" if score is not None else fam
        d.rectangle([x0, max(y0 - 15, 0), x0 + 7.2 * len(tag), max(y0 - 15, 0) + 14], fill=col)
        d.text((x0 + 3, max(y0 - 15, 0) + 2), tag, fill=(255, 255, 255))
    counts = {}
    for fam, _s, _b in boxes:
        counts[fam] = counts.get(fam, 0) + 1
    head = f"{name}   " + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    d.rectangle([4, 4, 12 + 7.2 * len(head), 24], fill=(0, 0, 0))
    d.text((10, 8), head, fill=(255, 255, 255))
    vis.save(OUT / f"{name}_boxes.jpg", quality=92)
    report.append({"name": name, "counts": counts})
    print(f"  {name}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

(OUT / "instances_batch.json").write_text(json.dumps(coco, indent=1))
tot = {}
for r in report:
    for k, v in r["counts"].items():
        tot[k] = tot.get(k, 0) + v
print(f"\n{len(report)} frames, boxes: {tot}")
print(f"written to {OUT}")
