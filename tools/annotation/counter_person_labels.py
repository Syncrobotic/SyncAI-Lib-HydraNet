#!/usr/bin/env python3
"""Person labels the teacher did not give: SAM 3 on the crowded window of every camera, at x2.

    python3 tools/annotation/counter_person_labels.py --root datasets/site30k_v1 \\
        --out datasets/site30k_counter_v1 --splits train val test

docs/PLAN.md 7a.41 measured the ceiling of the site person labels on 2026-09-10: Grounding
DINO, the source of every site person box, labels **11 of the 15-17 people** at a counter
huddle at the 0.35 the labels were cut at, and the student trained on them (person01)
returns 8-11 -- it learned the teacher's recall. SAM 3 on the same frames, on the counter
region upscaled x2, returns one clean box per person including the back row (median 13
at >= 0.3, 12 at >= 0.5). This tool makes that the dataset: the same images, the same
non-person labels, the person labels re-cut at a lower base threshold, and SAM 3's boxes
added wherever the frame is crowded.

---------------------------------------------------------------------------
WHAT IS RE-LABELLED, AND HOW THE WINDOW IS FOUND WITHOUT A HUMAN

SAM 3 at x2 over the full 1920x1080 costs a call per image and the corpus is 29,214
images; the labels that are missing are the ones in huddles, and huddles happen where
the counter is. Neither `camera.json`'s zone kinds (every fixture ships as `display`,
PLAN 2.1e) nor a human says where that is, so the window is read off the labels that
exist: the densest ``--window`` (960x600 by default, which upscales x2 onto the network
canvas without hitting its height) of person-box centres over the camera's own images.
Only images with at least ``--min-persons`` existing person boxes inside that window are
sent to SAM 3 -- a quiet frame's labels were never the problem -- which is 5,498 of
16,146 training images on the first fleet.

---------------------------------------------------------------------------
THE MERGE, STATED BECAUSE EVERY PART IS A CHOICE

* Non-person labels are copied verbatim. `boxed_stock` and `device` come from the
  campaign masks (`info.product`), not from Grounding DINO, and this tool has no opinion
  on them.
* Person labels start from `instances_all_<split>.json` -- Grounding DINO down to its
  0.10 floor -- cut at ``--base-thr`` (0.25: the threshold the ceiling probe found the
  back row at, 14.7 a frame against 11.1 at 0.35). On every image, crowded or not, so one
  threshold governs the whole split.
* Inside the window of a crowded image, SAM 3's boxes at ``--min-score`` (0.50, the
  score `scripts/sam3_person_boxes.py` measured its day person at) are mapped back to
  frame pixels and **added where they do not overlap an existing person box at
  ``--merge-iou``**; where they do, the existing box stays -- SAM 3 is the second opinion
  on presence, not on geometry. Its own duplicates are removed by NMS first.
* Every annotation carries ``source`` (`gdino` or `sam3`) and ``score``, so a training
  run, an audit or a re-cut can tell the two teachers apart later.

What survives is two teachers' opinion and not ground truth; thirty human-boxed frames
are the ruler both are scored against before any epoch runs (PLAN 7a.41). SAM 3's known
failure -- hanging packets as people on IR night frames -- does not arise here: the
crowded-window trigger needs four Grounding DINO people already present, and Grounding
DINO returns none on an empty night frame (`scripts/gdino_person_boxes.py`).

---------------------------------------------------------------------------
OUTPUT

``--out`` becomes a COCO root the training loader reads unchanged: ``<split>/`` is a
symlink to the source root's split directory (the images are not copied), and
``annotations/instances_<split>.json`` is the merged file. ``report.json`` carries the
window per camera and the counts, and ``preview/<camera>_<split>.jpg`` a contact sheet
of crowded frames with the window in yellow, Grounding DINO's kept boxes in blue and
SAM 3's additions in red -- the thing to look at before the dataset is used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syncai_bev3d.teachers.boxes import boxes_from_masks, nms  # noqa: E402
from syncai_hydranet.analytics.tracker import iou  # noqa: E402

PERSON_CAT = 1


# ------------------------------------------------------------------ pure parts


def densest_window(
    centres: np.ndarray, frame_wh: tuple[int, int], window_wh: tuple[int, int], cell: int = 20
) -> tuple[int, int, int, int]:
    """The window holding the most person-box centres, as (x0, y0, x1, y1) frame pixels.

    A 2-D histogram of centres at ``cell`` resolution, box-filtered to the window's
    size, argmax. With no centres at all the window sits at the frame's centre: a
    camera with no labelled people has no huddle to find and gets the default.
    """
    fw, fh = frame_wh
    ww, wh = min(window_wh[0], fw), min(window_wh[1], fh)
    nx, ny = max(1, fw // cell), max(1, fh // cell)
    if len(centres) == 0:
        x0, y0 = (fw - ww) // 2, (fh - wh) // 2
        return x0, y0, x0 + ww, y0 + wh
    hist, _, _ = np.histogram2d(
        centres[:, 1], centres[:, 0], bins=[ny, nx], range=[[0, fh], [0, fw]]
    )
    kx, ky = max(1, ww // cell), max(1, wh // cell)
    summed = ndimage.uniform_filter(hist, size=(ky, kx), mode="constant") * (kx * ky)
    # Window centres must keep the window inside the frame.
    cy_min, cy_max = ky // 2, ny - (ky - ky // 2)
    cx_min, cx_max = kx // 2, nx - (kx - kx // 2)
    sub = summed[cy_min : cy_max + 1, cx_min : cx_max + 1]
    iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
    cy, cx = (iy + cy_min + 0.5) * cell, (ix + cx_min + 0.5) * cell
    x0 = round(min(max(cx - ww / 2, 0), fw - ww))
    y0 = round(min(max(cy - wh / 2, 0), fh - wh))
    return x0, y0, x0 + ww, y0 + wh


def in_window(boxes_xyxy: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray:
    """Boxes whose centre lies inside the window."""
    if len(boxes_xyxy) == 0:
        return np.zeros(0, dtype=bool)
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    x0, y0, x1, y1 = window
    return (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)


def merge_additions(
    existing_xyxy: np.ndarray, sam3_xyxys: np.ndarray, merge_iou: float, nms_iou: float = 0.55
) -> np.ndarray:
    """SAM 3 boxes (N, 5 xyxy+score) that add a person: NMS'd, then not on an existing one."""
    if len(sam3_xyxys) == 0:
        return np.zeros((0, 5))
    cand = nms(sam3_xyxys, nms_iou)  # returns the kept boxes, input order preserved
    if len(existing_xyxy) == 0:
        return cand
    ov = iou(cand[:, :4], existing_xyxy)
    return cand[ov.max(axis=1) < merge_iou]


def to_frame(boxes_crop: np.ndarray, window, upscale: float) -> np.ndarray:
    """Boxes on the upscaled crop -> frame pixels, clipped to the window."""
    if len(boxes_crop) == 0:
        return np.zeros((0, 5))
    x0, y0, x1, y1 = window
    out = boxes_crop.copy()
    out[:, :4] = out[:, :4] / upscale + np.array([x0, y0, x0, y0])
    out[:, [0, 2]] = out[:, [0, 2]].clip(x0, x1)
    out[:, [1, 3]] = out[:, [1, 3]].clip(y0, y1)
    ok = (out[:, 2] - out[:, 0] >= 4) & (out[:, 3] - out[:, 1] >= 8)
    return out[ok]


def xywh_to_xyxy(b) -> list[float]:
    x, y, w, h = b
    return [float(x), float(y), float(x + w), float(y + h)]


# ------------------------------------------------------------------ the run


def load_split(root: Path, split: str) -> dict:
    """`instances_all_<split>.json` -- Grounding DINO down to its floor -- or the cut file."""
    p = root / "annotations" / f"instances_all_{split}.json"
    if not p.is_file():
        p = root / "annotations" / f"instances_{split}.json"
    return json.loads(p.read_text())


def camera_of(file_name: str) -> str:
    return file_name.split("__")[0]


def preview_sheet(rows: list[tuple[Image.Image, np.ndarray, np.ndarray, tuple]], path: Path):
    tiles = []
    for img, kept, added, window in rows:
        im = img.copy()
        d = ImageDraw.Draw(im)
        d.rectangle(list(window), outline=(255, 220, 0), width=4)
        for b in kept:
            d.rectangle(list(b[:4]), outline=(0, 130, 200), width=3)
        for b in added:
            d.rectangle(list(b[:4]), outline=(230, 25, 75), width=4)
            d.text((b[0] + 3, b[1] + 3), f"{b[4]:.2f}", fill=(230, 25, 75))
        tiles.append(im.resize((640, 360)))
    cols = 2
    n = len(tiles)
    sheet = Image.new("RGB", (640 * cols, 360 * ((n + cols - 1) // cols)), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 640, (i // cols) * 360))
    sheet.save(path, quality=85)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=ROOT / "datasets/site30k_v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument(
        "--cameras", nargs="*", default=None, help="default: every camera in the split"
    )
    ap.add_argument("--window", default="960x600", help="the crowded window, WxH frame px")
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument(
        "--min-persons",
        type=int,
        default=4,
        help="existing >=0.35 people inside the window to trigger SAM 3",
    )
    ap.add_argument(
        "--base-thr",
        type=float,
        default=0.25,
        help="existing person boxes kept at or above this",
    )
    ap.add_argument("--min-score", type=float, default=0.50, help="SAM 3's person score floor")
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument("--model-id", default="facebook/sam3")
    ap.add_argument(
        "--limit", type=int, default=0, help="crowded images per camera per split; 0 = all"
    )
    ap.add_argument(
        "--preview", type=int, default=8, help="crowded frames per camera on the sheet"
    )
    args = ap.parse_args()
    ww, wh = (int(v) for v in args.window.lower().split("x"))

    from syncai_bev3d.teachers.sam3 import load_sam3, segment
    from syncai_hydranet.utils.device import pick_device

    device = str(pick_device())
    proc, model = load_sam3(args.model_id, device)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "annotations").mkdir(exist_ok=True)
    (args.out / "preview").mkdir(exist_ok=True)
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    report: dict = {"settings": settings, "device": device, "cameras": {}}
    t0 = time.perf_counter()
    for split in args.splits:
        link = args.out / split
        if not link.exists():
            link.symlink_to((args.root / split).resolve())
        coco = load_split(args.root, split)
        imgs = {i["id"]: i for i in coco["images"]}
        by_img: dict[int, list] = defaultdict(list)
        for a in coco["annotations"]:
            by_img[a["image_id"]].append(a)
        cams = sorted({camera_of(i["file_name"]) for i in imgs.values()})
        if args.cameras:
            cams = [c for c in cams if c in set(args.cameras)]
        out_anns: list[dict] = []
        out_imgs: list[dict] = []
        ann_id = 1
        for cam in cams:
            cam_imgs = [i for i in imgs.values() if camera_of(i["file_name"]) == cam]
            fw, fh = cam_imgs[0]["width"], cam_imgs[0]["height"]
            centres = np.array(
                [
                    [a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2]
                    for i in cam_imgs
                    for a in by_img[i["id"]]
                    if a["category_id"] == PERSON_CAT and a.get("score", 1.0) >= 0.35
                ]
            ).reshape(-1, 2)
            window = densest_window(centres, (fw, fh), (ww, wh))
            stats = Counter()
            previews: list = []
            done = 0
            for info in sorted(cam_imgs, key=lambda i: i["file_name"]):
                anns = by_img[info["id"]]
                persons = [a for a in anns if a["category_id"] == PERSON_CAT]
                others = [a for a in anns if a["category_id"] != PERSON_CAT]
                kept = [a for a in persons if a.get("score", 1.0) >= args.base_thr]
                kept_xyxy = np.array([xywh_to_xyxy(a["bbox"]) for a in kept]).reshape(-1, 4)
                strong = np.array(
                    [xywh_to_xyxy(a["bbox"]) for a in persons if a.get("score", 1.0) >= 0.35]
                ).reshape(-1, 4)
                crowded = int(in_window(strong, window).sum()) >= args.min_persons
                added = np.zeros((0, 5))
                if crowded and (not args.limit or done < args.limit):
                    img = Image.open(args.root / split / info["file_name"]).convert("RGB")
                    x0, y0, x1, y1 = window
                    crop = img.crop((x0, y0, x1, y1))
                    if args.upscale != 1.0:
                        crop = crop.resize(
                            (int(crop.width * args.upscale), int(crop.height * args.upscale)),
                            Image.Resampling.LANCZOS,
                        )
                    sam = boxes_from_masks(
                        segment(proc, model, crop, "person", args.min_score, device)
                    )
                    sam = to_frame(sam, window, args.upscale)
                    added = merge_additions(kept_xyxy, sam, args.merge_iou)
                    stats["sam3_frames"] += 1
                    stats["sam3_boxes"] += len(sam)
                    stats["added"] += len(added)
                    done += 1
                    if len(previews) < args.preview:
                        previews.append((img, kept_xyxy, added, window))
                stats["images"] += 1
                stats["crowded"] += int(crowded)
                stats["gdino_kept"] += len(kept)
                stats["gdino_dropped_under_base"] += len(persons) - len(kept)
                out_imgs.append(dict(info))
                for a in others:
                    out_anns.append({**a, "id": ann_id, "source": a.get("source", "campaign")})
                    ann_id += 1
                for a in kept:
                    out_anns.append({**a, "id": ann_id, "source": "gdino"})
                    ann_id += 1
                for b in added:
                    out_anns.append(
                        {
                            "id": ann_id,
                            "image_id": info["id"],
                            "category_id": PERSON_CAT,
                            "bbox": [
                                float(b[0]),
                                float(b[1]),
                                float(b[2] - b[0]),
                                float(b[3] - b[1]),
                            ],
                            "area": float((b[2] - b[0]) * (b[3] - b[1])),
                            "iscrowd": 0,
                            "score": float(b[4]),
                            "source": "sam3",
                        }
                    )
                    ann_id += 1
            if previews:
                preview_sheet(previews, args.out / "preview" / f"{cam}_{split}.jpg")
            report["cameras"].setdefault(cam, {})[split] = {"window": list(window), **stats}
            print(
                f"{split:5s} {cam:16s} window {window}  images {stats['images']}  crowded "
                f"{stats['crowded']}  sam3 frames {stats['sam3_frames']}  added "
                f"{stats['added']}  gdino kept {stats['gdino_kept']} (dropped under "
                f"{args.base_thr}: {stats['gdino_dropped_under_base']})",
                flush=True,
            )
        out = {
            "info": {
                **coco.get("info", {}),
                "counter_labels": (
                    f"person: Grounding DINO cut at {args.base_thr} plus SAM 3 {args.model_id} "
                    f"person @{args.min_score} on the densest {args.window} window upscaled "
                    f"x{args.upscale}, on images with >= {args.min_persons} existing people in "
                    f"it; additions where IoU < {args.merge_iou} against an existing person "
                    "box. Two teachers' opinion, not ground truth (PLAN 7a.41)."
                ),
                "source_root": str(args.root),
            },
            "categories": coco["categories"],
            "images": out_imgs,
            "annotations": out_anns,
        }
        (args.out / "annotations" / f"instances_{split}.json").write_text(json.dumps(out))
    report["seconds"] = round(time.perf_counter() - t0, 1)
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.out} in {report['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
