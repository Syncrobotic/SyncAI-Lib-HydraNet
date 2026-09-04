#!/usr/bin/env python3
"""Who stood where, for how long, on eight commissioned cameras -- L1 end to end.

    python3 scripts/site_journeys.py --out runs/journeys01
    python3 scripts/site_journeys.py --cameras Taichung-cam01 --frames 300

The first fleet-wide run of the layer PLAN section 2.3.1 types and `analytics/journey.py`
consumes: detection -> tracks -> the vector space in metres -> visits to named zones with
durations. Every part of it is measured elsewhere; this is the first time they run
together on real footage from more than one camera.

The eight clips are the ones the 2026-08-26 threshold sweep and the pose-event pass used,
so a number here sits beside those rather than beside a new sample.

---------------------------------------------------------------------------
WHAT BOUNDS EVERY ROW, AND IT IS NOT THE THRESHOLDS

**A journey is a track's, not a customer's.** `analytics/events` records what person
tracks are worth on this footage: a 4.6-minute clip fragments into 1,234 tracks. One
shopper walking A -> B -> C and re-identified twice on the way produces three journeys,
each true and none of them the visit. So a dwell here is a *lower* bound on a visit, in a
direction that does not average out, and the fleet totals below are sums of fragments.
Association is PLAN step 5 and `reid_metrics.py` is where it will be shown to have worked.

**A foot point behind a counter is the counter's distance.** `Track.foot` states it: the
box bottom sits on the fixture edge when the feet are occluded, and fixtures are where
shoppers stand. That biases a position toward the far side of every counter -- which, for
a zone that *is* the floor beside a counter, can move a shopper into or out of it.

**Taichung-cam10's metres are 1.21x too large** (a known, undecided item: re-pinning
changes every metre already reported for that camera). Its zone areas, path lengths and
speeds are inflated by that factor; its *durations* are not.

**Zones come from SAM 3, and their kind is deliberately not decided.** Every zone ships as
`display` named `fixture_NN`; which fixture is the till is a fact about the store and a
shape-shaped prompt does not know it. So this reports "time at fixture 6", never "time at
the till".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from syncai_hydranet.analytics import events as ev
from syncai_hydranet.analytics.clip_tracks import track_clip
from syncai_hydranet.analytics.delivery import report_settings
from syncai_hydranet.analytics.journey import journeys
from syncai_hydranet.analytics.tracker import SIMPLIFICATIONS, Tracker
from syncai_hydranet.analytics.world import world_frames
from syncai_hydranet.data.video import frames, probe
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.serving.camera import BIRTH_REF
from syncai_hydranet.shipped import load_model
from syncai_hydranet.utils.device import pick_device
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path(__file__).resolve().parent.parent


COMMISSIONED = ROOT / "runs/commission01"

# The eight clips every fleet measurement of 2026-08-26 ran on. Read from
# `configs/sweep_clips.json` rather than written here, and rather than imported from a
# neighbouring script: what is shared is a corpus *selection*, which is data, and
# `tests/test_scripts_are_not_libraries.py` refuses one script importing another for the
# reason `clip_tracks.py` records -- four copies of one loop that stopped agreeing. A
# dataset path has no business in the wheel either, so the file is the right home for it.
_SWEEP = json.loads((ROOT / "configs/sweep_clips.json").read_text())
CLIPS = ROOT / _SWEEP["corpus_root"]
SWEEP_CLIPS: dict[str, str] = _SWEEP["clips"]


def run_camera(camera: str, model, cfg, device, args) -> dict:
    cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
    zones = ev.zones_from_camera(cam_file, kinds=["display", "till", "premium_shelf"])
    clip = CLIPS / camera / SWEEP_CLIPS[camera]
    if not clip.exists():
        raise SystemExit(f"{camera}: {clip} is missing")

    tracker = Tracker(iou_threshold=args.iou, max_age=args.max_age, min_hits=args.min_hits)
    out = track_clip(
        str(clip),
        model,
        cfg["data"]["input_size"],
        device,
        tracker,
        frames=frames,
        preprocess=preprocess,
        probe=probe,
        fps=args.fps,
        score_thr=args.score_thr,
        # Stated per camera from its own commissioning, never defaulted: `clip_tracks`
        # exists because four scripts disagreed about exactly this.
        k1=cam_file.lens.k1 if cam_file.lens else None,
        max_frames=args.frames,
    )
    # The boxes are in the decoded stream's pixels and several cameras are calibrated on
    # half that frame. Stating it is the difference between metres and confident nonsense.
    wf = world_frames(
        out.tracks,
        cam_file,
        name="person",
        fps=args.fps,
        source_size_px=(out.src_w, out.src_h),
    )
    js = journeys(wf, zones, fps=args.fps, min_seconds=args.min_seconds)

    per_zone: dict[str, dict] = {}
    for j in js:
        for v in j.visits:
            row = per_zone.setdefault(v.zone, {"visits": 0, "seconds": 0.0, "tracks": set()})
            row["visits"] += 1
            row["seconds"] += v.seconds or 0.0
            row["tracks"].add(j.track_id)
    for row in per_zone.values():
        row["tracks"] = len(row["tracks"])
        row["seconds"] = round(row["seconds"], 1)

    return {
        "camera": camera,
        "clip": clip.name,
        "frames": out.frames,
        "detections": out.detections,
        "tracks": len(out.tracks),
        "zones_available": len(zones),
        "zones_entered": len(per_zone),
        "journeys": [j.as_row() for j in js],
        "per_zone": dict(sorted(per_zone.items(), key=lambda kv: -kv[1]["seconds"])),
        "tracks_in_no_zone": sum(1 for j in js if not j.visits),
        "path_m_median": round(float(np.median([j.path_m for j in js])), 2) if js else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/journeys01"))
    ap.add_argument("--cameras", nargs="*", default=sorted(SWEEP_CLIPS))
    ap.add_argument("--config", default="configs/hydranet_retail_pose01.yaml")
    ap.add_argument(
        "--checkpoint",
        default="runs/hydranet_retail_security_b03_cw_xl-20260825-162131/last.pt",
    )
    ap.add_argument("--frames", type=int, default=900, help="900 at 5 fps is three minutes")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--score-thr", type=float, default=BIRTH_REF, help="the shipped birth edge")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--min-seconds", type=float, default=1.0, help="visit hysteresis")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(None)
    model, cfg, _ = load_model(args.config, args.checkpoint, device=device)

    fleet = []
    for camera in args.cameras:
        row = run_camera(camera, model, cfg, device, args)
        (args.out / f"{camera}.journeys.json").write_text(json.dumps(row, indent=1) + "\n")
        top = list(row["per_zone"].items())[:3]
        print(
            f"{camera:<18} {row['tracks']:>3} tracks  {row['zones_entered']:>2}/"
            f"{row['zones_available']:<2} zones entered  "
            f"{row['tracks_in_no_zone']:>2} in no zone  "
            + "  ".join(f"{z}={d['seconds']}s" for z, d in top)
        )
        fleet.append({k: v for k, v in row.items() if k != "journeys"})

    (args.out / "fleet.json").write_text(
        json.dumps(
            {
                "settings": report_settings(args, tracker_simplifications=SIMPLIFICATIONS),
                "cameras": fleet,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}/fleet.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
