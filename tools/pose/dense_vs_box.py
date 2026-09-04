"""What is in the people the dense head finds and the box head does not?

Measured 2026-08-26: the trunk segments `person` at IoU 0.885 while the detection head
scores mAP@50 0.302, and on one clip the dense map carried 1.30x as many person regions
as the box head returned boxes. That gap is either recall the detector is leaving on the
floor, or it is segmentation noise -- and the difference decides whether the dense head
can supervise the box head at all.

So this crops every dense `person` region the box head did not cover and lays them out
as a contact sheet, because the only honest way to answer "is that a person" is to look
at it. It also crops a matched sample as a control: a sheet of misses with nothing to
compare against invites the eye to see people in anything person-shaped.

Usage:
  uv run python tools/pose/dense_vs_box.py --checkpoint PATH [--cameras a,b,c]
      [--frames 60] [--stride 15]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from syncai_hydranet.analytics.tracker import iou_pair
from syncai_hydranet.data.video import frames as decode_frames
from syncai_hydranet.shipped import load_model
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
MIN_BLOB_PX = 300  # in the letterboxed map; a shopper at the far wall is bigger than this
COVER_IOU = 0.3
# A blob is not a person. An arm resting on a counter is segmented apart from the torso
# it belongs to, and against the torso's own box it scores almost no IoU while sitting
# entirely inside it -- counted as a miss, it inflates the recall gap with people the
# detector already found. Containment separates the two: mostly-inside an existing box
# is a fragment of a detected person, far from every box is a genuine miss.
FRAGMENT_CONTAINMENT = 0.5


def blob_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < MIN_BLOB_PX:
            continue
        out.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return out


def containment(blob, box) -> float:
    """Fraction of `blob`'s area that lies inside `box`."""
    x0 = max(blob[0], box[0])
    y0 = max(blob[1], box[1])
    x1 = min(blob[2], box[2])
    y1 = min(blob[3], box[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = (blob[2] - blob[0]) * (blob[3] - blob[1])
    return inter / area if area > 0 else 0.0


def sheet(
    crops: list[Image.Image], cols: int, cell: tuple[int, int], title: str
) -> Image.Image:
    rows = max(1, (len(crops) + cols - 1) // cols)
    w, h = cell
    img = Image.new("RGB", (cols * w, rows * h + 26), (16, 20, 27))
    d = ImageDraw.Draw(img)
    d.text((8, 6), title, fill=(255, 255, 255))
    for i, c in enumerate(crops):
        c = c.copy()
        c.thumbnail((w - 6, h - 6))
        x = (i % cols) * w + (w - c.width) // 2
        y = 26 + (i // cols) * h + (h - c.height) // 2
        img.paste(c, (x, y))
        d.text(((i % cols) * w + 4, 26 + (i // cols) * h + 2), str(i), fill=(255, 190, 60))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/hydranet_retail_pose01.yaml")
    ap.add_argument(
        "--cameras",
        default="Taichung-cam10,Taichung-cam01,Taichung-cam04,Tao-Hsin-cam03",
    )
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--out", default="runs/dense_vs_box")
    args = ap.parse_args()

    model, cfg, device = load_model(args.config, args.checkpoint, validate=False)
    size = cfg["data"]["input_size"]
    det_names = list(cfg["model"]["heads"]["detection"]["classes"])
    person = det_names.index("person")
    seg_person = list(cfg["data"]["terrain_classes"]).index("person")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    missed, fragments, matched, stats = [], [], [], []
    for cam in [c.strip() for c in args.cameras.split(",") if c.strip()]:
        clips = sorted((ROOT / "datasets/studioa_clips" / cam).glob("archive_*11*.mp4"))
        if not clips:
            print(f"  {cam}: no clip")
            continue
        n_blob = n_box = n_miss = n_frag = n_conf = 0
        for n, fr in enumerate(decode_frames(str(clips[0]), 1920, 1080, 5.0)):
            if n >= args.frames * args.stride:
                break
            if n % args.stride:
                continue
            img = Image.fromarray(fr)
            x, _c, region = preprocess(img, size)
            with torch.no_grad():
                res = model.predict(x.to(device), score_thr=args.score_thr)
            x0, y0, cw, ch = region
            det = res.get("detection", [{}])[0]
            boxes = []
            if det and len(det.get("boxes", [])):
                b = det["boxes"].cpu().numpy()
                lab = det["labels"].cpu().numpy()
                boxes = [tuple(v) for v in b[lab == person]]
            seg = res["terrain"][0].cpu().numpy()
            blobs = blob_boxes(seg[y0 : y0 + ch, x0 : x0 + cw] == seg_person)
            n_blob += len(blobs)
            n_box += len(boxes)
            # a box the dense head backs: the cheap confirmation this whole measurement
            # is looking for -- no training, and the confirming head scores person at
            # IoU 0.885 against this one's mAP@50 0.302
            for bx in boxes:
                bl_all = [(b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0) for b in blobs]
                if max((containment(bl, bx) for bl in bl_all), default=0.0) >= 0.25:
                    n_conf += 1
            scale = 1920.0 / cw
            for bb in blobs:
                # blob is in the cropped map; boxes are in letterboxed input space
                bl = (bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0)
                best = max((iou_pair(bl, bx) for bx in boxes), default=0.0)
                cont = max((containment(bl, bx) for bx in boxes), default=0.0)
                src = [int((v - (x0 if i % 2 == 0 else y0)) * scale) for i, v in enumerate(bl)]
                pad = int(0.12 * max(src[2] - src[0], src[3] - src[1]) + 8)
                crop = img.crop(
                    (max(0, src[0] - pad), max(0, src[1] - pad),
                     min(1920, src[2] + pad), min(1080, src[3] + pad))
                )  # fmt: skip
                if best >= COVER_IOU:
                    matched.append(crop)
                elif cont >= FRAGMENT_CONTAINMENT:
                    n_frag += 1
                    fragments.append(crop)
                else:
                    n_miss += 1
                    missed.append(crop)
        stats.append(
            {"camera": cam, "blobs": n_blob, "boxes": n_box,
             "dense_confirmed": n_conf, "fragments": n_frag, "uncovered": n_miss}
        )  # fmt: skip
        print(
            f"  {cam}: {n_blob} dense regions, {n_box} boxes, "
            f"{n_frag} fragments of a detected person, {n_miss} genuinely uncovered"
        )

    rng = np.random.default_rng(0)
    if len(matched) > 48:
        matched = [matched[i] for i in rng.choice(len(matched), 48, replace=False)]
    if missed:
        sheet(
            missed[:96], 12, (150, 200), f"UNCOVERED by the box head - {len(missed)} total"
        ).save(out / "uncovered.png")
    if matched:
        sheet(matched, 12, (150, 200), "CONTROL - dense regions the box head did cover").save(
            out / "covered_control.png"
        )
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    tb = sum(s["blobs"] for s in stats)
    tx = sum(s["boxes"] for s in stats)
    tm = sum(s["uncovered"] for s in stats)
    tf = sum(s["fragments"] for s in stats)
    print(
        f"\ntotal: {tb} dense regions, {tx} boxes | {tf} fragments "
        f"({tf / max(tb, 1):.0%}), {tm} genuinely uncovered ({tm / max(tb, 1):.0%})"
    )
    tc = sum(s["dense_confirmed"] for s in stats)
    print(f"recall left on the floor: {tm} against {tx} found = {tm / max(tx, 1):+.0%}")
    print(f"boxes the dense head backs: {tc} of {tx} ({tc / max(tx, 1):.0%})")
    print(f"wrote {out}/uncovered.png, fragments.png and covered_control.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
