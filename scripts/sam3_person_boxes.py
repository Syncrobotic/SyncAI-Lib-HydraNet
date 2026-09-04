#!/usr/bin/env python3
"""Site `person` boxes from SAM 3, without a human pass and without the failure that needs one.

    python3 scripts/sam3_person_boxes.py --out datasets/retail_person_batch01 \\
        --frames 12 datasets/studioa_clips/Taichung-cam*/archive_*.mp4

`person` is the class every security event is keyed on and the only one this project has
no site annotation for, so it is measured on COCO val and nowhere else. This closes that
with the labour budget available, which is none.

---------------------------------------------------------------------------
THE FAILURE THIS EXISTS TO AVOID, MEASURED 2026-08-18 BY ANOTHER SESSION

On Taichung-cam09 at 23:59, store closed, IR monochrome, empty frame, SAM 3's `person`
prompt returns **14 instances at score >= 0.5 -- every one an accessory packet hanging on
the pegboard wall**, at 0.55-0.60. Nine of 48 cameras return something for `person` on the
midnight frames. `person` sits on LAYER_PERSON and overwrites everything beneath it, and
`sam3_prompts.py` already says a false person is the one false positive nothing downstream
can outvote.

**And the fixed-camera consensus vote makes it worse, not better.** The packets are in the
same pixels in every frame, so they agree perfectly; voting keeps exactly what is wrong.

Two gates instead, and both are measurements rather than filename guesses:

**1. Daylight only.** `person` peaks at 0.984 by day and 0.703 at night, while static
controls are unchanged by IR (`display table` 0.977 -> 0.980). So IR does not degrade SAM 3
in general, it degrades this class. A frame is rejected when it is too dark or when its
channels are too close together to be colour -- an IR frame is monochrome by construction,
so the test is a property of the pixels and not of the clock.

**2. Static rejection -- built, tested, and OFF BY DEFAULT because the test failed it.**
The theory was consensus inverted: a shopper moves, a packet on a pegboard does not, so a
box recurring in the same place is furniture. Run on Kaohsiung-cam04 it dropped 21 of 103
boxes, and drawing them showed all three in the sampled frame were **people** -- two
staff standing at their stations and a customer at the counter. It removes exactly the
person a loitering alarm exists for.

The discriminator was wrong rather than the idea: recurrence does not separate a hanging
packet from a person who is standing still. Sub-pixel immobility would, and measuring it
needs consecutive frames rather than the 37-second spacing used here. Until that is
measured the gate stays off (`--static-share 0`), because a gate that deletes staff is
worse than the failure it was aimed at -- and the failure it was aimed at is a *night*
phenomenon that gate 1 already excludes.

What survives is still SAM 3's opinion and not ground truth. `split.json`'s
`test_provenance` says the same thing about the merchandise boxes, and every number
downstream of this file inherits it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from syncai_bev3d.teachers.boxes import (  # noqa: E402
    boxes_from_masks,
    drop_static,
)
from syncai_bev3d.teachers.photometry import (  # noqa: E402
    MIN_CHROMA,
    MIN_LUMA,
    is_daylight,
)
from syncai_bev3d.teachers.sam3 import load_sam3, segment  # noqa: E402
from syncai_hydranet.data.video import frames, probe  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="facebook/sam3")
    ap.add_argument("--frames", type=int, default=12, help="frames sampled per clip")
    ap.add_argument("--min-score", type=float, default=0.50)
    # Daylight gate. Both are properties of the pixels, and both defaults are measured
    # over all 48 cameras -- `syncai_bev3d/teachers/photometry.py` carries the distribution and
    # why 1.0 sits where it does.
    ap.add_argument("--min-luma", type=float, default=MIN_LUMA)
    ap.add_argument("--min-chroma", type=float, default=MIN_CHROMA)
    # Static gate. A box recurring at IoU >= 0.7 in this share of frames is furniture.
    ap.add_argument("--static-iou", type=float, default=0.7)
    # 0 disables. Off by default -- see the docstring: it was measured removing people.
    ap.add_argument("--static-share", type=float, default=0.0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = str(pick_device())
    proc, model = load_sam3(args.model_id, device)
    out_root = Path(args.out)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)

    images, annotations, report = [], [], []
    img_id = ann_id = 1
    for clip in args.clips:
        camera = Path(clip).parent.name
        w, h, _ = probe(clip)
        picked, per_frame, skipped = [], [], 0
        every = max(int(300 * 5 / max(args.frames, 1)), 1)
        for i, frame in enumerate(frames(clip, w, h, 5.0)):
            if i % every:
                continue
            if not is_daylight(frame, args.min_luma, args.min_chroma):
                skipped += 1
                continue
            img = Image.fromarray(frame)
            per_frame.append(
                boxes_from_masks(segment(proc, model, img, "person", args.min_score, device))
            )
            picked.append((i, frame))
            if len(picked) >= args.frames:
                break
        if not picked:
            report.append(
                {
                    "camera": camera,
                    "session": Path(clip).stem,
                    "kept": 0,
                    "dropped_static": 0,
                    "frames_skipped_not_daylight": skipped,
                    "note": "no daylight frame in this clip",
                }
            )
            continue
        if args.static_share > 0:
            kept, dropped = drop_static(per_frame, args.static_iou, args.static_share)
        else:
            kept, dropped = per_frame, [np.zeros((0, 5))] * len(per_frame)
        n_dropped = sum(len(b) for b in dropped)
        n_kept = 0
        (out_root / "dropped").mkdir(parents=True, exist_ok=True)
        for (idx, _f), db in zip(picked, dropped, strict=True):
            if len(db):
                np.save(out_root / "dropped" / f"{camera}__{idx:05d}.npy", db)
        for (idx, frame), boxes in zip(picked, kept, strict=True):
            name = f"{camera}__{Path(clip).stem}__{idx:05d}.jpg"
            Image.fromarray(frame).save(out_root / "images" / name, quality=92)
            images.append({"id": img_id, "file_name": name, "width": w, "height": h})
            for b in boxes:
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [
                            float(b[0]),
                            float(b[1]),
                            float(b[2] - b[0]),
                            float(b[3] - b[1]),
                        ],
                        "area": float((b[2] - b[0]) * (b[3] - b[1])),
                        "iscrowd": 0,
                        "score": float(b[4]),
                    }
                )
                ann_id += 1
                n_kept += 1
            img_id += 1
        report.append(
            {
                "camera": camera,
                "session": Path(clip).stem,
                "frames": len(picked),
                "kept": n_kept,
                "dropped_static": n_dropped,
                "frames_skipped_not_daylight": skipped,
            }
        )
        print(
            f"{camera:18s} {len(picked):3d} daylight frames  {n_kept:4d} person boxes kept  "
            f"{n_dropped:4d} dropped as static  {skipped:3d} frames not daylight"
        )

    coco = {
        "info": {
            "source": "SAM 3 `person` prompt, daylight-gated and static-rejected",
            "provenance": "PRE-LABEL, NOT GROUND TRUTH -- no human pass",
            "why_the_gates": "sam3 person returns hanging accessory packets as people on "
            "IR night frames; 14 on one empty Taichung-cam09 frame",
            "min_score": args.min_score,
        },
        "categories": [{"id": 1, "name": "person"}],
        "images": images,
        "annotations": annotations,
    }
    (out_root / "annotations" / "instances_all.json").write_text(json.dumps(coco))
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    total_drop = sum(r["dropped_static"] for r in report)
    print(
        f"\n{len(annotations)} person boxes over {len(images)} daylight frames; "
        f"{total_drop} dropped as static. Wrote {out_root}"
    )
    print("SAM 3's opinion, not ground truth. Every number downstream inherits that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
