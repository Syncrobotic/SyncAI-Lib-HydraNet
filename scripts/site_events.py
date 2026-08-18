#!/usr/bin/env python3
"""Security events on a real store clip: the whole L0 -> L1 -> L2 chain, once, end to end.

    python3 scripts/site_events.py --config runs/hydranet_retail_cctv/config.yaml \\
        --checkpoint runs/hydranet_retail_cctv/best.pt \\
        --out runs/site_events01 datasets/studioa_clips/Taichung-cam01/archive_*.mp4

`analytics/events.py` has tests and had never seen a store. What stood between the two was
never the code: it was a **camera pose**, because every zone in that module is a polygon in
metres on the floor and there was no fitted pose on disk for any camera.

---------------------------------------------------------------------------
THE POSE, AND WHY IT IS TAICHUNG-CAM01

`fit_camera_from_people.py` records a tile-grid fit for exactly one camera, and records why
it should be trusted over the people-based fit its own filename promises: a manufactured
grid has no posture floor, while 142 detections of staff bent over counters and customers
on stools measured 1.0-1.2 m tall and drove every pinhole parameter to its search boundary.

    k1 = -0.225   vfov 70.4 deg   pitch 50.2 deg   height 2.38 m

**Order matters and the lens comes first.** Undistorted, those tiles give 70.4/50.2; fitted
as a pinhole they give 92.5/65.4, which is the fit that measured people at a metre tall.
So this script undistorts before it projects, and `Pose.rejection` gates the result --
the check that caught a "passing" pose whose horizon sat below the frame, i.e. a ceiling
camera fitted as though it looked up.

Defaults are that camera's. **On any other camera they are an assumption**, and the honest
symptom of a wrong one is not an error -- it is a floor that is smoothly the wrong size.

---------------------------------------------------------------------------
THE ZONE IS DERIVED FROM THE FOOTAGE, NOT DRAWN BY ME

A zone invented at a desk would make every event below a statement about my invention. So
the zone is the **busiest square metre of floor in this clip**: foot points are projected,
binned at 0.25 m, and a square is placed on the peak. That is where shoppers actually
stand, which on these cameras is a counter or a display table.

It is still a demonstration and not a product zone. A real one is drawn once per camera by
whoever owns the store's rules, which is the whole reason `Zone` takes metres and a
threshold rather than being learned.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
sys.path.insert(0, str(HERE))

from retail_flow import PERSON, to_source_pixels  # noqa: E402

from syncai_hydranet.analytics import events as ev  # noqa: E402
from syncai_hydranet.analytics.dwell import track_ground_path  # noqa: E402
from syncai_hydranet.analytics.tracker import SIMPLIFICATIONS, Tracker  # noqa: E402
from syncai_hydranet.cli.infer_video import frames, probe  # noqa: E402
from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.geometry.calibrate import Pose, undistort_points  # noqa: E402
from syncai_hydranet.geometry.ground import Camera, GroundPlane  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import preprocess  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--score-thr", type=float, default=0.20)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--iou", type=float, default=0.3)
    # Taichung-cam01's tile-grid fit. Per camera, and an assumption anywhere else.
    ap.add_argument("--k1", type=float, default=-0.225)
    ap.add_argument("--vfov", type=float, default=70.4, metavar="DEG")
    ap.add_argument("--pitch", type=float, default=50.2, metavar="DEG")
    ap.add_argument("--camera-height", type=float, default=2.38, metavar="M")
    # Store rules, which is why they are arguments. None is measured.
    ap.add_argument("--zone-side", type=float, default=2.0, metavar="M")
    ap.add_argument("--max-occupancy", type=int, default=2)
    ap.add_argument("--loiter-seconds", type=float, default=45.0)
    ap.add_argument("--run-speed", type=float, default=2.5, metavar="MPS")
    return ap


def undistort_boxes(boxes: np.ndarray, k1: float, w: int, h: int) -> np.ndarray:
    """Undo the lens on box corners, so the ground projection sees a pinhole camera.

    Corners rather than the foot point alone, and then the axis-aligned hull: the tracker
    associates on IoU and would otherwise compare undistorted foot points against distorted
    boxes. The hull is an approximation of a shape that is no longer a rectangle after a
    radial map; at these radii it moves an edge by a pixel or two, which is far below the
    error in the pose itself and does not pretend otherwise.
    """
    if not len(boxes):
        return boxes
    centre = (w / 2.0, h / 2.0)
    radius = math.hypot(h, w) / 2.0
    pts = boxes.reshape(-1, 2, 2).reshape(-1, 2)
    out = undistort_points(pts, k1, centre, radius).reshape(-1, 2, 2)
    return np.concatenate([out.min(axis=1), out.max(axis=1)], axis=1)


def track_clip(clip: str, model, size, device, args):
    src_w, src_h, _ = probe(clip)
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
            box = undistort_boxes(box, args.k1, src_w, src_h)
        else:
            box = np.zeros((0, 4))
        tracker.update(box, n)
        n += 1
    return tracker.finished(), n, src_w, src_h


def busiest_zone(tracks, cam, plane, side: float) -> tuple[ev.Zone, dict]:
    """A square on the floor's busiest 0.25 m cell. Returns the zone and what backs it."""
    pts = np.concatenate([track_ground_path(t, cam, plane) for t in tracks]) if tracks else None
    if pts is None or not len(pts):
        raise SystemExit("no tracks, so no floor to place a zone on")
    pts = pts[np.isfinite(pts).all(axis=1)]
    cell = 0.25
    keys = np.floor(pts / cell).astype(int)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    peak = uniq[counts.argmax()] * cell + cell / 2
    half = side / 2
    poly = np.array(
        [
            [peak[0] - half, peak[1] - half],
            [peak[0] + half, peak[1] - half],
            [peak[0] + half, peak[1] + half],
            [peak[0] - half, peak[1] + half],
        ]
    )
    support = {
        "foot_points_projected": len(pts),
        "peak_cell_track_frames": int(counts.max()),
        "peak_centre_m": [round(float(v), 2) for v in peak],
        "share_of_all_foot_points": round(float(counts.max() / len(pts)), 4),
    }
    return ev.Zone("busiest_floor", poly, max_occupancy=None, loiter_seconds=None), support


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), "ema"))
    size = cfg["data"]["input_size"]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("tracker simplifications, which bound every row below:")
    for s in SIMPLIFICATIONS:
        print(f"  - {s}")

    report = {"settings": vars(args), "clips": []}
    for clip in args.clips:
        camera = Path(clip).parent.name
        tracks, n_frames, w, h = track_clip(clip, model, size, device, args)

        # The gate before any metre is printed. A pose whose horizon lands inside the
        # frame manufactures floor; one below it has the camera looking up.
        horizon = h / 2 - (h / 2) / math.tan(math.radians(args.vfov) / 2) * math.tan(
            math.radians(args.pitch)
        )
        pose = Pose(args.k1, args.vfov, args.pitch, args.camera_height, horizon, h, w)
        print(f"\n{camera}: {pose.summary()}")
        if pose.rejection:
            print("  refusing to project; no events for this clip")
            continue

        cam = Camera.from_vfov(h, w, args.vfov)
        plane = GroundPlane(height=args.camera_height, pitch=math.radians(args.pitch))
        zone, support = busiest_zone(tracks, cam, plane, args.zone_side)
        zone_occ = ev.Zone(zone.name, zone.polygon, max_occupancy=args.max_occupancy)
        zone_loiter = ev.Zone(zone.name, zone.polygon, loiter_seconds=args.loiter_seconds)

        rows = []
        rows += ev.zone_events([*tracks], [zone_loiter], cam, plane, args.fps, camera)
        rows += ev.occupancy_events(tracks, zone_occ, cam, plane, args.fps, camera)
        rows += ev.speed_events(
            tracks, cam, plane, args.fps, camera, max_speed_mps=args.run_speed
        )
        rows += ev.crowd_events(tracks, cam, plane, args.fps, camera)
        rows.sort(key=lambda e: e.frame_start)

        lengths = [len(t.frames) for t in tracks]
        entry = {
            "camera": camera,
            "session": Path(clip).stem,
            "frames": n_frames,
            "tracks": len(tracks),
            "median_track_frames": float(np.median(lengths)) if lengths else None,
            "pose": pose.summary().splitlines()[0],
            "zone_m": [[round(float(x), 2) for x in p] for p in zone.polygon],
            "zone_support": support,
            "events": [e.as_row() for e in rows],
        }
        report["clips"].append(entry)
        counts: dict[str, int] = {}
        for e in rows:
            counts[e.type] = counts.get(e.type, 0) + 1
        print(f"  {len(tracks)} tracks, zone at {support['peak_centre_m']} m")
        print(f"  events: {counts or 'none'}")
        (out_root / "events.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_root / 'events.json'}")
    print("Every metre here rests on one camera's tile-grid fit. Tape-measure it before use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
