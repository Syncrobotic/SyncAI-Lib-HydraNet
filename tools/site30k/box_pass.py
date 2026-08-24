#!/usr/bin/env python3
"""Detection labels for the finished campaign, written in the SHIPPING vocabulary.

    python tools/site30k/box_pass.py --root datasets/site30k_v1
    python tools/site30k/box_pass.py --root datasets/site30k_v1 --limit 40 --preview 8

The campaign wrote masks and images and no boxes at all, so the detection head currently
gets nothing from 29,211 annotated frames. This closes that, and it is far cheaper than
the campaign was, for one reason: **the product boxes are already in the masks.**
`recipe.py` paints laptop/tablet/phone/boxed_stock as ids 7-10, so a product box is a
connected component of the mask, not a second SAM 3 pass. Only `person` needs a GPU
teacher, over jpgs that are already on disk -- no video decode anywhere in this script.

VOCABULARY -- the decision this script exists to get right
----------------------------------------------------------
The campaign's own detection taxonomy is two categories, `person` and `product`, with the
source family kept per box. Writing that verbatim would be a silent regression in the
business logic: `analytics/events/zones.py:346` defaults to
``classes=("boxed_stock", "device")`` and matches by NAME, so after a retrain on a
`product`-only vocabulary its wanted set is empty, `zone_stock_counts` returns 0 for every
frame, and the stock-removal alarm stops firing WITHOUT RAISING. So the families are
mapped back onto `retail_security`, which is what the deployed model speaks::

    person                      -> person        (vocab id 0)
    boxed_stock                 -> boxed_stock   (vocab id 2)
    laptop | tablet | phone     -> device        (vocab id 3, "demo units, handsets, laptops")
    bag                         -> not produced here; it keeps coming from batch02/03

`family` is written on every annotation regardless, so a later head that wants the finer
split can have it without re-running anything.

Two populations, deliberately
-----------------------------
`instances_<split>.json` holds `person` at PERSON_THRESHOLD (0.35, the measured day/night
gap) after NMS -- the file a training run reads. `instances_all_<split>.json` holds every
person box down to SCORE_FLOOR (0.10) with no NMS, so the score population stays visible
and a future threshold question can be answered without GPU time. The campaign is daylight
only, so `night_person`'s static veto is not applied here; a night tranche must not reuse
this script without it.

Resumable per image into `boxes_cache.jsonl`: a rerun re-reads the cache and only pays GPU
for frames it has never seen. The COCO files are assembled from the cache at the end, so a
kill mid-run costs the frames in flight and nothing else.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
sys.path.insert(0, str(ROOT / "src"))

from syncai_hydranet.data.teachers import boxes as B  # noqa: E402
from syncai_hydranet.data.teachers import gdino as G  # noqa: E402

SCORE_FLOOR = 0.10  # keep both score populations visible; see the module docstring
NMS_IOU = 0.55
MIN_PRODUCT_PX = 300  # same cut as tools/site30k/boxes.py used on the reviewed batch

# mask id -> the family the campaign painted
PRODUCT_IDS = {7: "laptop", 8: "tablet", 9: "phone", 10: "boxed_stock"}
# family -> the shipping `retail_security` category
FAMILY_TO_VOCAB = {
    "laptop": "device",
    "tablet": "device",
    "phone": "device",
    "boxed_stock": "boxed_stock",
}
# `retail_security` ids, from src/syncai_hydranet/data/label_maps_retail_security.py.
# COCO category ids are 1-based, so they are written as vocab_id + 1 and the name is what
# `det_vocab` actually matches on.
VOCAB_ID = {"person": 0, "bag": 1, "boxed_stock": 2, "device": 3}
CATEGORIES = [{"id": VOCAB_ID[n] + 1, "name": n} for n in ("person", "boxed_stock", "device")]
BOX_COLORS = {"person": (230, 25, 75), "boxed_stock": (170, 110, 40), "device": (140, 60, 255)}


def product_boxes(mask: np.ndarray) -> list[tuple[str, list[float], int]]:
    """(family, xyxy, area_px) per connected component of each product id."""
    out = []
    for cid, fam in PRODUCT_IDS.items():
        lab, n = ndimage.label(mask == cid)
        if not n:
            continue
        for sl, k in zip(ndimage.find_objects(lab), range(1, n + 1)):
            if sl is None:
                continue
            area = int((lab[sl] == k).sum())
            if area < MIN_PRODUCT_PX:
                continue
            ys, xs = sl
            out.append((fam, [float(xs.start), float(ys.start),
                              float(xs.stop - 1), float(ys.stop - 1)], area))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("datasets/site30k_v1"))
    ap.add_argument("--model", default=G.MODEL_ID)
    ap.add_argument("--limit", type=int, default=0, help="first N frames only, for a smoke pass")
    ap.add_argument("--preview", type=int, default=24, help="how many drawn frames to write")
    ap.add_argument("--preview-every", type=int, default=1200)
    args = ap.parse_args()

    root = args.root
    splits = json.loads((root / "splits.json").read_text())
    split_of = {cam: s for s, cams in splits["cameras"].items() for cam in cams}
    masks = sorted((root / "masks").glob("*.png"))
    if args.limit:
        masks = masks[:: max(1, len(masks) // args.limit)][: args.limit]

    cache_path = root / "boxes_cache.jsonl"
    cached: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cached[rec["name"]] = rec
    print(f"{len(masks)} frames, {len(cached)} already in the cache")

    todo = [m for m in masks if m.stem not in cached]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = model = None
    if todo:
        print(f"loading {args.model} on {device}")
        proc, model = G.load_gdino(args.model, device)

    prev_dir = root / "preview_boxes"
    prev_dir.mkdir(exist_ok=True)
    written = 0
    t0 = time.time()
    with cache_path.open("a") as cache_fh:
        for i, mpath in enumerate(todo):
            name = mpath.stem
            img = Image.open(root / "images" / f"{name}.jpg").convert("RGB")
            mask = np.asarray(Image.open(mpath))
            raw = G.detect(proc, model, img, "person", SCORE_FLOOR, device)
            strong = B.nms(raw[raw[:, 4] >= G.PERSON_THRESHOLD], NMS_IOU) if len(raw) else raw
            rec = {
                "name": name,
                "w": img.width,
                "h": img.height,
                "person": [[float(v) for v in b] for b in strong],
                "person_all": [[float(v) for v in b] for b in raw],
                "product": [[fam, box, area] for fam, box, area in product_boxes(mask)],
            }
            cache_fh.write(json.dumps(rec) + "\n")
            cache_fh.flush()
            cached[name] = rec
            written += 1
            if written <= args.preview or i % args.preview_every == 0:
                draw_preview(img, rec, prev_dir / f"{name}_boxes.jpg")
            if written % 200 == 0:
                rate = written / (time.time() - t0)
                left = (len(todo) - written) / max(rate, 1e-6) / 3600
                print(f"  [{written}/{len(todo)}] {rate:.1f} fps, eta {left:.2f} h", flush=True)

    assemble(root, masks, cached, split_of)
    return 0


def draw_preview(img: Image.Image, rec: dict, out: Path) -> None:
    vis = img.copy()
    d = ImageDraw.Draw(vis)
    items = [("person", b[:4], b[4]) for b in rec["person"]]
    items += [(FAMILY_TO_VOCAB[fam], box, None) for fam, box, _a in rec["product"]]
    for cls, (x0, y0, x1, y1), score in items:
        col = BOX_COLORS[cls]
        d.rectangle([x0, y0, x1, y1], outline=col, width=3 if cls == "person" else 2)
        tag = f"{cls} {score:.2f}" if score is not None else cls
        d.rectangle([x0, max(y0 - 15, 0), x0 + 7.2 * len(tag), max(y0 - 15, 0) + 14], fill=col)
        d.text((x0 + 3, max(y0 - 15, 0) + 2), tag, fill=(255, 255, 255))
    counts = Counter(c for c, _b, _s in items)
    head = f"{rec['name']}   " + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    d.rectangle([4, 4, 12 + 7.2 * len(head), 24], fill=(0, 0, 0))
    d.text((10, 8), head, fill=(255, 255, 255))
    vis.save(out, quality=90)


def assemble(root: Path, masks, cached: dict, split_of: dict) -> None:
    """Write instances_<split>.json (trainable) and instances_all_<split>.json (population)."""
    info = {
        "source": str(root),
        "person": f"Grounding DINO {G.MODEL_ID} @{G.PERSON_THRESHOLD}, NMS {NMS_IOU}",
        "product": (f"connected components of the campaign mask ids {sorted(PRODUCT_IDS)}, "
                    f">= {MIN_PRODUCT_PX} px; families mapped to the retail_security vocab"),
        "family_to_vocab": FAMILY_TO_VOCAB,
        "note": "teacher opinion, not ground truth. Daylight only: no night static veto applied.",
    }
    per_split: dict[str, dict] = {}
    tally: dict[str, Counter] = {}
    for m in masks:
        rec = cached.get(m.stem)
        if rec is None:
            continue
        camera = m.stem.split("__")[0]
        split = split_of.get(camera)
        if split is None:
            continue
        d = per_split.setdefault(split, {"info": info, "categories": CATEGORIES,
                                         "images": [], "annotations": [],
                                         "annotations_all": []})
        t = tally.setdefault(split, Counter())
        iid = len(d["images"]) + 1
        d["images"].append({"id": iid, "file_name": f"{rec['name']}.jpg",
                            "width": rec["w"], "height": rec["h"]})
        for b in rec["person"]:
            add(d["annotations"], iid, "person", b[:4], score=b[4])
            t["person"] += 1
        for fam, box, area in rec["product"]:
            add(d["annotations"], iid, FAMILY_TO_VOCAB[fam], box, family=fam, area=area)
            t[FAMILY_TO_VOCAB[fam]] += 1
        for b in rec["person_all"]:
            add(d["annotations_all"], iid, "person", b[:4], score=b[4])

    ann_dir = root / "annotations"
    ann_dir.mkdir(exist_ok=True)
    for split, d in sorted(per_split.items()):
        allann = d.pop("annotations_all")
        (ann_dir / f"instances_{split}.json").write_text(json.dumps(d))
        (ann_dir / f"instances_all_{split}.json").write_text(
            json.dumps({"info": info, "categories": CATEGORIES,
                        "images": d["images"], "annotations": allann}))
        n_img = len(d["images"])
        empty = n_img - len({a["image_id"] for a in d["annotations"]})
        print(f"  {split:5s} {n_img:6d} images  {len(d['annotations']):7d} boxes  "
              f"{dict(sorted(tally[split].items()))}  ({empty} images with no box)")
    print(f"wrote instances_<split>.json and instances_all_<split>.json under {ann_dir}")


def add(into: list, image_id: int, cls: str, xyxy, score=None, family=None, area=None) -> None:
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    rec = {"id": len(into) + 1, "image_id": image_id, "category_id": VOCAB_ID[cls] + 1,
           "bbox": [x0, y0, x1 - x0, y1 - y0],
           "area": float(area if area is not None else (x1 - x0) * (y1 - y0)), "iscrowd": 0}
    if score is not None:
        rec["score"] = float(score)
    if family is not None:
        rec["family"] = family
    into.append(rec)


if __name__ == "__main__":
    raise SystemExit(main())
