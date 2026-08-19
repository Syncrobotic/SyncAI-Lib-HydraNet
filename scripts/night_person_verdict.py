#!/usr/bin/env python3
"""Does GDINO's 0.35 hold at night on every camera, or only on the one it was measured on?

    python3 scripts/night_person_verdict.py \\
        --gdino runs/gdino_night_fleet --sam3 runs/sam3_night_fleet \\
        --static runs/static_gdino_night_fleet --out runs/night_person_verdict

THE CAVEAT THIS EXISTS TO CLOSE

`person`'s teacher was swapped to Grounding DINO at 0.35 on 2026-08-19, and the night half
of that decision rested on **one empty-store clip on one camera**: Taichung-cam09's IR
midnight frames, where SAM 3 returned 229 false people at its own 0.50 default and GDINO
topped out at 0.326. The journal entry that recorded the swap named the gap in the same
breath -- nine of 48 cameras returned SAM 3 night hallucinations, so 0.35 had to be
re-measured across the fleet before "works at night" could be a claim rather than a hope.

This joins three measurements over the same 12 midnight frames from each of the 42 live
cameras, and the frames really are the same: `gdino_person_boxes.py` and
`sam3_person_boxes.py` share a sampling rule to the line (`every = 300*5/frames`, decode at
5 fps, break at `--frames`), so running the second with its daylight gate disabled puts both
teachers on identical pixels without a new sampler to disagree.

  1. **GDINO** at floor 0.10, so both score populations stay visible.
  2. **SAM 3** at its shipped 0.50, which is the failure being replaced -- and which also
     recovers the identity of those nine cameras. The number was written down; the names
     were not, the same way sweep C's per-camera output was not. This writes them down.
  3. **Static share** against the ~00:00 plates in `datasets/studioa_static`, which exist
     for all 42 cameras and are cut from these very clips, so the reference illumination
     matches the frame. A packet on a pegboard scores near 1.00; a person who is standing
     still still breathes and scores near 0.

WHAT A VERDICT MEANS HERE

`0.35 holds` on a camera when no box on its empty midnight frames reaches 0.35. A camera
that fails is not automatically a counter-example to the teacher swap: a cleaner or a
restocker at midnight is a **real person**, and a threshold that finds them is working. The
static share is what separates the two, and a camera whose high-scoring box is *not* static
is reported as `person present` rather than as a failure. Anything that scores above 0.35
**and** is static is the first real counter-example and is named as such.

The output is a table, not a decision. A blocky high score on an empty store gets opened at
native resolution before it is called anything -- the two candidate `column` cameras of the
same day both looked plausible in a number and were wrong on sight.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# The ladder the swap was argued on, plus 0.30 -- the journal quotes an 18x separation
# there and the threshold ladder in `gdino_person_boxes.py` skips it.
CUTS = (0.25, 0.30, 0.35, 0.50)

# Above this a box's pixels never moved across ~150 samples of the clip: furniture, or
# stock hanging on it. Measured, not chosen: the day population sits at 0.133 median and
# the night [0.25, 0.35) population at 0.890, and this sits in the gap between them.
STATIC_IS_FURNITURE = 0.60


def camera_of(file_name: str) -> str:
    return file_name.split("__")[0]


def load_gdino(root: Path) -> tuple[dict, dict]:
    """Per-camera score arrays, and the per-image lookup the static join needs."""
    coco = json.loads((root / "annotations" / "instances_all.json").read_text())
    images = {im["id"]: im for im in coco["images"]}
    per_cam: dict[str, list] = defaultdict(list)
    for a in coco["annotations"]:
        im = images[a["image_id"]]
        per_cam[camera_of(im["file_name"])].append(
            {"score": a["score"], "file": im["file_name"], "bbox": a["bbox"], "id": a["id"]}
        )
    frames_per_cam: dict[str, int] = defaultdict(int)
    for im in images.values():
        frames_per_cam[camera_of(im["file_name"])] += 1
    return per_cam, frames_per_cam


def load_sam3(root: Path) -> dict[str, int]:
    """SAM 3 person boxes per camera at its own 0.50 -- the hallucination count."""
    path = root / "annotations" / "instances_all.json"
    if not path.exists():
        return {}
    coco = json.loads(path.read_text())
    images = {im["id"]: im for im in coco["images"]}
    per_cam: dict[str, int] = defaultdict(int)
    for a in coco["annotations"]:
        per_cam[camera_of(images[a["image_id"]]["file_name"])] += 1
    return per_cam


def box_key(file_name: str, bbox) -> tuple:
    """`static_person_filter.py` writes no annotation id, so boxes join on identity.

    Rounding to the pixel is deliberate: both sides read the same COCO file, so the
    coordinates are bit-identical and the rounding only guards against a float repr
    difference, never against two genuinely different boxes colliding.
    """
    return (file_name, tuple(round(float(v), 1) for v in bbox))


def load_static(root: Path | None) -> dict[tuple, float]:
    """(frame, box) -> static share, from `static_person_filter.py`'s boxes.json."""
    if root is None or not (root / "boxes.json").exists():
        return {}
    return {
        box_key(b["file_name"], b["bbox"]): b["static_share"]
        for b in json.loads((root / "boxes.json").read_text())
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gdino", type=Path, required=True)
    ap.add_argument("--sam3", type=Path, default=None)
    ap.add_argument("--static", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decision-thr", type=float, default=0.35)
    args = ap.parse_args(argv)

    per_cam, frames_per_cam = load_gdino(args.gdino)
    sam3 = load_sam3(args.sam3) if args.sam3 else {}
    static = load_static(args.static)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for cam in sorted(frames_per_cam):
        boxes = per_cam.get(cam, [])
        scores = np.array([b["score"] for b in boxes], dtype=float)
        over = [b for b in boxes if b["score"] >= args.decision_thr]
        shares = [static.get(box_key(b["file"], b["bbox"])) for b in over]
        known = [s for s in shares if s is not None]
        rows.append(
            {
                "camera": cam,
                "frames": frames_per_cam[cam],
                "gdino_max": round(float(scores.max()), 4) if len(scores) else 0.0,
                "gdino_n": {f"{c:.2f}": int((scores >= c).sum()) for c in CUTS},
                "sam3_at_0.50": sam3.get(cam, 0),
                "over_thr": len(over),
                "over_thr_static_median": round(float(np.median(known)), 3) if known else None,
                "over_thr_static_known": len(known),
                "verdict": verdict(over, known),
            }
        )

    (args.out / "verdict.json").write_text(json.dumps(rows, indent=1))
    write_report(rows, args.out, args.decision_thr)
    return 0


def verdict(over: list, shares: list[float]) -> str:
    """Three outcomes, and the middle one is the reason this is not a pass/fail count."""
    if not over:
        return "holds"
    if shares and float(np.median(shares)) >= STATIC_IS_FURNITURE:
        return "COUNTER-EXAMPLE"  # above threshold and it never moved: not a person
    if shares:
        return "person present"  # above threshold and it moves: the threshold is working
    return "review"  # above threshold, no static reference to judge it by


def write_report(rows: list[dict], out: Path, thr: float) -> None:
    holds = [r for r in rows if r["verdict"] == "holds"]
    people = [r for r in rows if r["verdict"] == "person present"]
    counter = [r for r in rows if r["verdict"] == "COUNTER-EXAMPLE"]
    review = [r for r in rows if r["verdict"] == "review"]
    sam3_hits = [r for r in rows if r["sam3_at_0.50"] > 0]

    lines = [
        f"# Does {thr} hold at night, fleet-wide?",
        "",
        f"**{len(holds)} of {len(rows)} cameras: no box on an empty midnight frame reaches "
        f"{thr}.**",
        "",
        f"- `holds` — {len(holds)}",
        f"- `person present` — {len(people)} (over threshold, but the box moves: a real "
        "person at midnight is not a failure of the threshold)",
        f"- **`COUNTER-EXAMPLE` — {len(counter)}** (over threshold and static: the box is "
        "furniture or stock, and this is what would break the swap)",
        f"- `review` — {len(review)} (over threshold with no static plate to judge by)",
        "",
        f"SAM 3 at its own 0.50 returns something on **{len(sam3_hits)} of {len(rows)}** "
        "cameras over the same frames. The journal recorded that count and not the names; "
        "they are in the table below.",
        "",
        "| camera | frames | GDINO max | >=0.25 | >=0.30 | >=0.35 | >=0.50 "
        "| SAM 3 @0.50 | static med | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (-r["gdino_max"], r["camera"])):
        n = r["gdino_n"]
        med = r["over_thr_static_median"]
        sm = "--" if med is None else f"{med:.2f}"
        lines.append(
            f"| {r['camera']} | {r['frames']} | {r['gdino_max']:.3f} "
            f"| {n['0.25']} | {n['0.30']} | {n['0.35']} | {n['0.50']} "
            f"| {r['sam3_at_0.50']} | {sm} | {r['verdict']} |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"\nwrote {out}/REPORT.md and {out}/verdict.json")


if __name__ == "__main__":
    raise SystemExit(main())
