#!/usr/bin/env python3
"""Recover a fixed camera's height and pitch from the people standing under it.

    python3 scripts/fit_camera_from_people.py --config configs/hydranet_retail_cctv.yaml \\
        --checkpoint runs/hydranet_retail_cctv/best.pt --input assets/clip.mp4

`bev_video.py` needs a camera height and a down-pitch, and its defaults (1.5 m, 15 deg)
describe a robot's own camera. Pointed at ceiling CCTV they are simply wrong, and wrong
in the way this project keeps warning about: every distance in the panel comes out scaled
by a smooth factor and nothing looks broken.

**What replaces the guess.** A standing adult is about 1.70 m. Each detection gives a
foot point on the floor and a head point above it, so a candidate (height, pitch) implies
a height for every person seen. The right pose is the one where those implied heights
*agree with each other* -- that is what pins the pitch, because a wrong pitch makes
people near the top of the frame measure differently from people near the bottom -- and
then the pose that puts their median at 1.70 m pins the camera height, because height
scales the whole scene uniformly.

So the output is a fit to data, not a nicer guess. What it inherits:

* **1.70 m is still an assumption**, just a far better grounded one than 1.5 m of camera
  height. Every distance carries whatever error is in it, proportionally.
* **The vertical field of view is not fitted.** It trades off against pitch, and one
  clip's people do not separate them reliably. It stays an input, and the residual is
  printed so a bad one is visible rather than absorbed.
* Detections whose feet or head leave the frame are dropped: a cropped box measures the
  crop, not the person.

**Measured limitation, 2026-08-14.** On `archive_20260802-125220` the residual falls
*monotonically* with the vfov given to it -- 4.9% at 55 deg, 3.0% at 90, 1.1% at 140,
0.7% at 150 -- and the fitted pose slides with it (20 deg / 1.87 m up to 67 deg /
1.69 m). A pinhole model with a free field of view is absorbing the lens distortion, so
"lowest residual" selects the most extreme lens rather than the real one. At a fixed
vfov the pitch is only weakly pinned as well: 20 to 35 deg all sit within 0.5% of each
other while the implied camera height runs 1.87 m to 2.98 m.

So this recovers a *family* of poses, not a pose. What is genuinely constrained is the
shape of the floor and the ordering of ranges; the absolute scale is not. One tape
measure on site, or one object of known size lying on the floor, collapses the family
immediately and is worth more than any amount of fitting here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from live_view_orin import COCO_NAMES  # noqa: E402

from syncai_hydranet.cli.infer_video import frames, probe  # noqa: E402  # isort: skip
from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.geometry.bev import box_extents  # noqa: E402
from syncai_hydranet.geometry.ground import Camera, GroundPlane  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights  # noqa: E402
from syncai_hydranet.utils.visualize import preprocess  # noqa: E402

ADULT_M = 1.70


def collect(args, device) -> np.ndarray:
    """Person boxes, in the letterboxed mask's coordinates, one row per detection."""
    cfg = load_config(args.config, args.set)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), args.weights))
    size = cfg["data"]["input_size"]
    src_w, src_h, _ = probe(args.input)
    person = COCO_NAMES.index("person")

    keep, shape = [], None
    step = max(int(args.stride), 1)
    for i, frame in enumerate(frames(args.input, src_w, src_h, args.fps)):
        if i % step:
            continue
        x, _, region = preprocess(Image.fromarray(frame), size)
        with torch.no_grad():
            out = model.predict(x.to(device), score_thr=args.score_thr)
        x0, y0, cw, ch = region
        shape = (ch, cw)
        det = out.get("detection", [{}])[0]
        if not det or not len(det.get("boxes", [])):
            continue
        b = det["boxes"].cpu().numpy() - np.array([x0, y0, x0, y0], dtype=np.float32)
        lab = det["labels"].cpu().numpy()
        sc = det["scores"].cpu().numpy()
        for box, la, s in zip(b, lab, sc, strict=True):
            if la != person or s < args.person_score:
                continue
            bw, bh = box[2] - box[0], box[3] - box[1]
            # A box touching an edge is a crop, and a crop is not a person.
            m = args.margin_px
            if box[0] < m or box[1] < m or box[2] > cw - m or box[3] > ch - m:
                continue
            if bh <= 0 or not (1.4 <= bh / max(bw, 1e-6) <= 6.0):
                continue
            keep.append(box)
        if args.max_boxes and len(keep) >= args.max_boxes:
            break
    if shape is None or not keep:
        raise SystemExit("no usable person detections; lower --person-score or --stride")
    return np.asarray(keep, dtype=float), shape


def fit(boxes: np.ndarray, shape, vfov: float, heights, pitches):
    cam = Camera.from_vfov(shape[0], shape[1], vfov)
    best = None
    for pitch in pitches:
        plane_unit = GroundPlane(height=1.0, pitch=np.radians(pitch))
        ext = box_extents(boxes, cam, plane_unit)
        hs = ext[:, 1]
        hs = hs[np.isfinite(hs) & (hs > 0)]
        if len(hs) < 8:
            continue
        # Heights scale linearly with camera height, so the *relative* spread is a
        # function of pitch alone -- which is what makes this a one-dimensional search
        # per pitch instead of a two-dimensional one.
        med = float(np.median(hs))
        spread = float(np.median(np.abs(hs - med))) / max(med, 1e-9)
        cam_h = ADULT_M / med  # the height that puts the median person at 1.70 m
        if not (heights[0] <= cam_h <= heights[-1]):
            continue
        if best is None or spread < best[0]:
            best = (spread, float(pitch), float(cam_h), len(hs))
    if best is None:
        raise SystemExit("no pitch produced a usable fit within the allowed height range")
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--input", required=True)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--stride", type=int, default=15, help="use every Nth sampled frame")
    ap.add_argument("--score-thr", type=float, default=0.15)
    ap.add_argument("--person-score", type=float, default=0.30)
    ap.add_argument("--margin-px", type=float, default=6.0)
    ap.add_argument("--max-boxes", type=int, default=600)
    ap.add_argument("--vfov", type=float, default=55.0)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boxes, shape = collect(a, device)
    spread, pitch, cam_h, n = fit(
        boxes, shape, a.vfov, heights=(1.0, 8.0), pitches=np.arange(5.0, 80.1, 0.5)
    )
    print(f"{len(boxes)} person boxes kept, {n} usable")
    print(f"  vfov      {a.vfov:.0f} deg   (input, not fitted)")
    print(f"  pitch     {pitch:.1f} deg down")
    print(f"  height    {cam_h:.2f} m")
    print(f"  residual  {100 * spread:.1f}% median deviation of implied person heights")
    if spread > 0.20:
        print("  !! that spread is large: the fit is weak and the scale should not be quoted")
    print()
    print(f"  --camera-height {cam_h:.2f} --pitch {pitch:.1f} --vfov {a.vfov:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
