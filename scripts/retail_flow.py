#!/usr/bin/env python3
"""People-flow analytics over site clips: tracks, dwell and a floor heatmap.

    python3 scripts/retail_flow.py --config configs/hydranet_retail_objects.yaml \\
        --checkpoint runs/hydranet_retail_objects/best.pt --fps 5 \\
        --out runs/flow01 /path/to/cam*.mp4

Stage one of the plan in the session that commissioned it: **zone dwell and heatmaps on
the existing angled cameras**, not footfall. Those are different products and only one
of them is buildable from this hardware.

Footfall -- "how many people entered" -- is a line-crossing count and wants an overhead
camera at the door. Commercial counters use nadir units for a reason that is geometric
rather than algorithmic: two shoppers walking abreast overlap in an angled view and are
two separate blobs from directly above. Asking these cameras for footfall means fighting
occlusion forever to reach a number a EUR 400 sensor gets right by construction.

Dwell and heatmaps are the opposite case. They need identity only *within one camera's
view for a few seconds*, never across the store, and they are distributions rather than
counts -- so an error moves a percentile instead of accumulating into the headline. Both
are numbers retail pays for.

**Every figure this writes is provisional until a manual audit.** The industry accepts
+/-5% for counting; nothing here has been checked against a hand count, because this
project has no hand-labelled site data at all. Sample an hour, count it by eye, compare.
That is the acceptance test and there is no substitute for it.

The camera pose is per-camera and it is the largest source of error in the metric
output. docs/journal/2026-08-14 records what happens when it is guessed: a pinhole fit
absorbed the lens's barrel distortion and put 142 detected people at 1.0-1.2 m tall,
and at the wrong pitch one image column spanned 0.07 to 3543 m of floor. Fit the lens
first, then the pose -- `scripts/fit_camera_from_people.py`, and prefer floor tiles to
people.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.analytics import Tracker, dwell_table, track_ground_path  # noqa: E402
from syncai_hydranet.analytics.dwell import ground_map  # noqa: E402
from syncai_hydranet.analytics.tracker import SIMPLIFICATIONS  # noqa: E402
from syncai_hydranet.cli.infer_video import frames, probe  # noqa: E402
from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.data.coco_subsets import COCO_NAMES  # noqa: E402
from syncai_hydranet.data.transforms import invert_geom  # noqa: E402
from syncai_hydranet.geometry.ground import Camera, GroundPlane  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import preprocess  # noqa: E402

PERSON = COCO_NAMES.index("person")


def to_source_pixels(boxes: np.ndarray, region, src_w: int, src_h: int) -> np.ndarray:
    """Model-canvas boxes -> pixels of the frame that was actually filmed.

    `preprocess` letterboxes, so the canvas is content plus grey bars and a plain
    `src_w / model_w` scale is wrong by the padding. Getting it wrong does not look
    wrong: every box lands somewhere plausible, shifted by the bar width, and the ground
    projection then reports a floor position that is confidently a metre off.

    The arithmetic already exists as `invert_geom`, which the evaluator uses to put COCO
    boxes back on the original image; this only translates between the two ways the
    project describes the same letterbox. `region` is (x0, y0, content_w, content_h)
    from `preprocess`; `geom` is (sx, sy, px, py) with `out = (box - p) / s`. A second
    copy of the arithmetic would be a second chance to get a sign wrong.
    """
    if not len(boxes):
        return np.zeros((0, 4))
    x0, y0, cw, ch = region
    return invert_geom(np.asarray(boxes, dtype=float), (cw / src_w, ch / src_h, x0, y0))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=5.0, help="sampling rate for tracking")
    ap.add_argument("--max-frames", type=int, default=0, help="0 means the whole clip")
    ap.add_argument(
        "--score-thr",
        type=float,
        default=0.20,
        help="0.20 rather than the 0.30 viewing default: measured on Taichung-cam01, "
        "person detections went 62 -> 72 -> 112 as the cut fell 0.35 -> 0.25 -> 0.15, "
        "and a tracker cannot associate what was never detected",
    )
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--iou", type=float, default=0.3)
    # Pose. Per camera, and the defaults are Taichung-cam01's corrected fit from
    # docs/journal/2026-08-14 -- right for that camera and an assumption for any other.
    ap.add_argument("--camera-height", type=float, default=2.38, metavar="M")
    ap.add_argument("--pitch", type=float, default=50.2, metavar="DEG")
    # 70.4, not 55. The comment above says these defaults are Taichung-cam01's corrected
    # fit and two of the three were: the tile-grid fit in
    # `fit_camera_from_people.py`'s docstring reads "70.4 deg vfov and 50.2 deg pitch",
    # and quotes `f 766` for the result. 55 deg on a 1080-row frame is f 1037 -- 35%
    # longer -- so every metre this script printed under its own defaults was wrong by
    # that factor, with nothing in the output looking wrong. The stale value is the one
    # the fit replaced, left behind when the other two were updated.
    ap.add_argument("--vfov", type=float, default=70.4, metavar="DEG")
    ap.add_argument("--cell", type=float, default=0.25, metavar="M")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), "ema"))
    size = cfg["data"]["input_size"]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("tracker simplifications, which bound every number below:")
    for s in SIMPLIFICATIONS:
        print(f"  - {s}")
    print(
        f"\npose: height {args.camera_height} m, pitch {args.pitch} deg, "
        f"vfov {args.vfov} deg -- per camera, and defaults are Taichung-cam01's"
    )

    report = {"clips": [], "settings": vars(args) | {"simplifications": list(SIMPLIFICATIONS)}}
    for clip in args.clips:
        session = Path(clip).stem
        src_w, src_h, _ = probe(clip)
        cam = Camera.from_vfov(src_h, src_w, args.vfov)
        plane = GroundPlane(height=args.camera_height, pitch=np.deg2rad(args.pitch))

        tracker = Tracker(iou_threshold=args.iou, max_age=args.max_age, min_hits=args.min_hits)
        n = 0
        for frame in frames(clip, src_w, src_h, args.fps):
            x, _, region = preprocess(Image.fromarray(frame), size)
            with torch.no_grad():
                res = model.predict(x.to(device), score_thr=args.score_thr)
            det = res["detection"][0]
            if len(det.get("labels", [])):
                lab = det["labels"].cpu().numpy()
                box = det["boxes"].cpu().numpy()[lab == PERSON]
                box = to_source_pixels(box, region, src_w, src_h)
            else:
                box = np.zeros((0, 4))
            tracker.update(box, n)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break

        tracks = tracker.finished()
        rows = dwell_table(tracks, fps=args.fps, last_frame=n - 1)
        paths = [track_ground_path(t, cam, plane) for t in tracks]
        gmap = ground_map(paths, cell_m=args.cell)

        complete = [r for r in rows if not r["truncated"]]
        dwells = [r["dwell_s"] for r in complete]
        entry = {
            "session": session,
            "frames": n,
            "tracks": len(rows),
            "complete_tracks": len(complete),
            "median_dwell_s": float(np.median(dwells)) if dwells else None,
            "p90_dwell_s": float(np.percentile(dwells, 90)) if dwells else None,
            "visited_m2": round(gmap.visited_m2, 2),
            "busiest_cells_xz_count": gmap.busiest(3),
            "rows": rows,
        }
        report["clips"].append(entry)
        np.save(out_root / f"{session}_heatmap.npy", gmap.cells)
        med = f"{entry['median_dwell_s']:.1f}s" if dwells else "--"
        print(
            f"{session}: {n} frames, {len(rows)} tracks "
            f"({len(complete)} complete), median dwell {med}, "
            f"floor visited {entry['visited_m2']:.1f} m2"
        )

    (out_root / "flow.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out_root / 'flow.json'}")
    print(
        "\nNot validated. Sample an hour of footage, count it by eye, and compare before\n"
        "any of these numbers reaches a business decision -- the industry bar for counting\n"
        "is +/-5% and nothing here has been checked against a hand count."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
