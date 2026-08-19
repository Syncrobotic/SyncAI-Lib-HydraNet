#!/usr/bin/env python3
"""Score person pseudo-boxes against a camera's static mask, and draw them.

    python3 scripts/static_person_filter.py \
        --coco datasets/retail_person_batch01/annotations/instances_all.json \
        --static datasets/studioa_static --out runs/static_person_filter

`sam3_person_boxes.py` already carries a static gate and it is **off by default, because
the test failed it**: on Kaohsiung-cam04 it dropped 21 of 103 boxes and all three examined
were people -- two staff at their stations and a customer at the counter. Its own docstring
names the reason and the fix:

> recurrence does not separate a hanging packet from a person who is standing still.
> Sub-pixel immobility would, and measuring it needs consecutive frames rather than the
> 37-second spacing used here.

`static_plates.py` measures exactly that: over ~150 samples of the clip, how many times each
pixel deviated by more than eight of its own noise floors. A member of staff standing at one
station still breathes and shifts, so their pixels move by 40-100x that floor; a packet on a
pegboard moves by 6-18x, which is IR compression noise on high-frequency texture and not
motion at all. The two populations are an order of magnitude apart, which is what makes the
cut possible -- and getting there took three wrong versions of the mask, each of which is
recorded in that file because each was wrong in a different and instructive way.

So this is the discriminator that session identified and could not measure, and the first
thing to establish is **not** how many boxes it removes -- it is whether it repeats the old
gate's mistake. Scored against every one of Kaohsiung-cam04's 103 daylight boxes, it does
not; scored against only one of them, an earlier version looked fine and would have deleted
78. **Validate a gate against the whole population it can delete, never a spot check.**

That is why this script does not filter anything. It scores every box, writes the
distribution, and renders each frame with its boxes coloured by score, so the decision to
cut and the place to cut are made by looking rather than by a default.

**No threshold is baked in.** The old gate's `--static-share` had a default and a rationale
and was still wrong. Per ARCHITECTURE_DIRECTION rule 2, a cut belongs in a measured gap
between two populations; `--drop-above` exists so that once the gap is measured, the number
can be written down where it was measured. Left unset, nothing is dropped and only the
report is written.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_name(file_name: str) -> tuple[str, str, int]:
    """`Kaohsiung-cam04__archive_20260816-112757_20260816-113301__00374.jpg`
    -> `("Kaohsiung-cam04", "20260816-112757", 374)`.

    The middle field is the clip stem and its first timestamp is the slot key
    `static_plates.py` writes -- which is **UTC**, while the timestamp burned into the
    frame is store-local (+8). Nothing here needs the local hour, and it is not derived,
    precisely so that no reader is tempted to treat the slot key as one.
    """
    camera, session, frame = file_name.rsplit(".", 1)[0].split("__")
    return camera, session.split("_")[1], int(frame)


def static_share(mask: np.ndarray, box, img_w: int, img_h: int) -> tuple[float, int]:
    """Fraction of a box's area that never changed, plus the pixel count behind it.

    The count is returned because rule 1 applies to this number too: a share of 1.00 over
    nine mask pixels is a box smaller than the mask's resolution, not a confident verdict.
    """
    mh, mw = mask.shape
    x, y, w, h = box
    x0 = int(np.floor(x * mw / img_w))
    y0 = int(np.floor(y * mh / img_h))
    x1 = int(np.ceil((x + w) * mw / img_w))
    y1 = int(np.ceil((y + h) * mh / img_h))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(max(x1, x0 + 1), mw), min(max(y1, y0 + 1), mh)
    patch = mask[y0:y1, x0:x1]
    return (float(patch.mean()) if patch.size else 0.0), int(patch.size)


def load_mask(static_root: Path, camera: str, slot: str, which: str) -> np.ndarray | None:
    name = "static_all.png" if which == "all" else f"static_{slot}.png"
    path = static_root / camera / name
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("L")) > 127


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--images", type=Path, default=None, help="default: --coco/../../images")
    ap.add_argument("--static", type=Path, default=Path("datasets/studioa_static"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--mask",
        choices=["slot", "all"],
        default="slot",
        help="slot: the clip's own mask -- a person in THIS clip makes their own pixels "
        "dynamic in it. all: the intersection over every slot, which is the stronger "
        "claim and the one that survives a person who stood still for a whole clip",
    )
    ap.add_argument(
        "--drop-above",
        type=float,
        default=None,
        help="static share above which a box is called false. Still unset by default -- the "
        "old gate had a default and a rationale and was wrong anyway -- but the gap has "
        "now been measured and 0.5 sits in it. On Taichung-cam09's night IR clip, 229 SAM 3 "
        "`person` boxes on an empty closed shop score a median 0.936 and 225 are above 0.5; "
        "on Kaohsiung-cam04's daylight clip, 103 boxes of real shoppers and staff score a "
        "median 0.133 and NONE is above 0.5. Both sets were looked at, not just counted",
    )
    ap.add_argument("--only", nargs="*", default=[], help="camera names")
    ap.add_argument("--draw", type=int, default=12, help="how many frames to render")
    args = ap.parse_args(argv)

    coco = json.loads(args.coco.read_text())
    _images = {im["id"]: im for im in coco["images"]}
    per_image = defaultdict(list)
    for ann in coco["annotations"]:
        per_image[ann["image_id"]].append(ann)
    img_root = args.images or args.coco.parent.parent / "images"

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "frames").mkdir(exist_ok=True)

    rows: list[dict] = []
    missing: set[str] = set()
    masks: dict[tuple[str, str], np.ndarray | None] = {}
    for im in coco["images"]:
        camera, slot, frame_idx = parse_name(im["file_name"])
        if args.only and camera not in args.only:
            continue
        key = (camera, slot)
        if key not in masks:
            masks[key] = load_mask(args.static, camera, slot, args.mask)
            if masks[key] is None:
                missing.add(f"{camera}/{slot}")
        mask = masks[key]
        if mask is None:
            continue
        for ann in per_image[im["id"]]:
            share, npix = static_share(mask, ann["bbox"], im["width"], im["height"])
            rows.append(
                {
                    "camera": camera,
                    "slot": slot,
                    "frame": frame_idx,
                    "file_name": im["file_name"],
                    "bbox": [round(v, 1) for v in ann["bbox"]],
                    "score": ann.get("score"),
                    "static_share": round(share, 4),
                    "mask_pixels": npix,
                }
            )

    if missing:
        print(f"no static mask for {len(missing)} (camera, slot): {sorted(missing)[:6]}")
    if not rows:
        print("no boxes scored -- is the static batch finished for these cameras?")
        return 1

    shares = np.array([r["static_share"] for r in rows])
    print(f"\n{len(rows)} boxes over {len({r['file_name'] for r in rows})} frames")
    print("static_share 分佈:")
    for q in (5, 10, 25, 50, 75, 90, 95, 99, 100):
        print(f"   p{q:<3d} {np.percentile(shares, q):.3f}")
    hist, edges = np.histogram(shares, bins=10, range=(0, 1))
    print("  直方圖 (0..1, 10 bins):")
    for h, lo in zip(hist, edges[:-1], strict=False):
        print(f"   {lo:.1f}-{lo + 0.1:.1f}  {h:5d}  {'#' * min(60, h)}")

    # Render frames, boxes coloured by share. Highest-share frames first: those are the
    # ones a cut would act on, so those are the ones worth a human's eye.
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_frame[r["file_name"]].append(r)
    order = sorted(by_frame, key=lambda f: -max(r["static_share"] for r in by_frame[f]))
    drawn = 0
    for file_name in order[: args.draw]:
        src = img_root / file_name
        if not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        draw = ImageDraw.Draw(img)
        for r in by_frame[file_name]:
            x, y, w, h = r["bbox"]
            s = r["static_share"]
            colour = (
                (240, 60, 60) if s >= 0.9 else (250, 200, 40) if s >= 0.5 else (60, 220, 90)
            )
            draw.rectangle([x, y, x + w, y + h], outline=colour, width=4)
            draw.text((x + 4, y + 4), f"static {s:.2f}", fill=colour)
        img.save(args.out / "frames" / file_name.replace(".jpg", ".png"))
        drawn += 1

    report = {
        "coco": str(args.coco),
        "mask": args.mask,
        "boxes": len(rows),
        "percentiles": {
            f"p{q}": round(float(np.percentile(shares, q)), 4) for q in (5, 25, 50, 75, 95, 100)
        },
        "per_camera": {
            cam: {
                "boxes": sum(1 for r in rows if r["camera"] == cam),
                "median_share": round(
                    float(np.median([r["static_share"] for r in rows if r["camera"] == cam])), 4
                ),
                "over_0.9": sum(
                    1 for r in rows if r["camera"] == cam and r["static_share"] >= 0.9
                ),
            }
            for cam in sorted({r["camera"] for r in rows})
        },
        "drop_above": args.drop_above,
        "would_drop": (
            sum(1 for r in rows if r["static_share"] >= args.drop_above)
            if args.drop_above is not None
            else None
        ),
        "frames_rendered": drawn,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (args.out / "boxes.json").write_text(json.dumps(rows, indent=1) + "\n")
    print("\n每台相機:")
    for cam, v in report["per_camera"].items():
        print(
            f"   {cam:20s} {v['boxes']:4d} boxes  中位 {v['median_share']:.3f}"
            f"  >=0.9: {v['over_0.9']}"
        )
    if args.drop_above is not None:
        print(f"\n--drop-above {args.drop_above}: 會丟掉 {report['would_drop']} / {len(rows)}")
    print(f"\n{drawn} 張圖 -> {args.out}/frames  (紅 >=0.9, 黃 >=0.5, 綠 <0.5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
