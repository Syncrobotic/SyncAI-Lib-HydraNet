#!/usr/bin/env python3
"""Why do tracks end? The fragmentation question, asked without a single label.

    python3 scripts/track_endings.py --out runs/endings01

`runs/journeys01` measured the consequence: 197 visits with a median of 3.2 s, and eight
in 24 minutes lasting 30 s or more. The cause is that tracks are shorter than visits, and
`reid_metrics.idf1` / `id_switches` cannot say why -- both need ground-truth tracks, and
its own header records that a labelled site clip does not exist. Labelling one is exactly
the human cost this project refuses to pay by default (PLAN section 4).

**A track's own ending is evidence, and it is free.** When a track dies, one of three
things is true, and they are distinguishable from the detections alone:

* **exit** -- its last box was against a frame edge. The person left the view; the track
  was supposed to end.
* **lost** -- a detection appears near where it was, within `--gap` frames, and no live
  track claimed it. The tracker dropped a person who was still there. This is
  fragmentation, and it is the number that has to come down.
* **gone** -- no detection near it either. The detector lost the person: an occlusion, a
  fixture, a bad pose. A tracker cannot fix this one.

`tracker._match` already ran this on one clip and recorded the shape -- "48 of that clip's
76 tracks ended with a detection available" -- which is what made "match against the
predicted box **or** the last observed one" worth trying. This is that measurement over
eight cameras, so the next tracker change has a fleet baseline rather than one clip.

The classification is deliberately generous to the tracker: a detection counts as "near"
only within `--near` metres of the dead track's last foot point, measured **on the floor**
rather than in pixels, because a pixel radius means a different distance at 3 m and at
12 m and would call every distant reappearance a fragmentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from syncai_hydranet.analytics.clip_tracks import track_clip  # noqa: E402
from syncai_hydranet.analytics.delivery import report_settings  # noqa: E402
from syncai_hydranet.analytics.tracker import Tracker  # noqa: E402
from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.data.video import frames, probe  # noqa: E402
from syncai_hydranet.geometry.camera_json import CameraFile  # noqa: E402
from syncai_hydranet.geometry.ground import pixel_to_ground, undistort_points  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import preprocess  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from site_journeys import CLIPS, COMMISSIONED, SWEEP_CLIPS  # noqa: E402


class Recording(Tracker):
    """A `Tracker` that keeps every frame's detections, so endings can be explained.

    Subclassed rather than wrapped because `track_clip` calls `update` and `finished` and
    nothing else; a wrapper would have to forward both and would drift the first time the
    loop learned a third method.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen: dict[int, np.ndarray] = {}

    def update(self, boxes, frame_idx, keypoints=None, scores=None):
        self.seen[int(frame_idx)] = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
        return super().update(boxes, frame_idx, keypoints=keypoints, scores=scores)


def foot_m(boxes: np.ndarray, cam_file: CameraFile, src) -> np.ndarray:
    """Box bottom-centres to floor metres, through this camera's own calibration."""
    if not len(boxes):
        return np.zeros((0, 2))
    pts = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
    w, h = cam_file.image_size_px
    pts = pts * np.array([w / src[0], h / src[1]])
    if cam_file.lens is not None:
        lens = cam_file.lens
        pts = undistort_points(pts, lens.k1, lens.centre_px, lens.radius_px)
    x, z = pixel_to_ground(pts[:, 0], pts[:, 1], cam_file.camera, cam_file.plane)
    return np.stack([x, z], axis=1)


def classify(track, tracker: Recording, cam_file, src, args) -> str:
    last_frame = track.frames[-1]
    box = np.asarray(track.boxes[-1], dtype=float)
    w, h = src
    edge = (
        box[0] <= args.edge_px
        or box[1] <= args.edge_px
        or box[2] >= w - args.edge_px
        or box[3] >= h - args.edge_px
    )
    if edge:
        return "exit"
    here = foot_m(box[None], cam_file, src)[0]
    if not np.isfinite(here).all():
        return "gone"
    for f in range(last_frame + 1, last_frame + 1 + args.gap):
        boxes = tracker.seen.get(f)
        if boxes is None or not len(boxes):
            continue
        d = np.linalg.norm(foot_m(boxes, cam_file, src) - here, axis=1)
        if np.isfinite(d).any() and float(np.nanmin(d)) <= args.near:
            return "lost"
    return "gone"


def run_camera(camera: str, model, cfg, device, args) -> dict:
    cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
    tracker = Recording(iou_threshold=args.iou, max_age=args.max_age, min_hits=args.min_hits)
    out = track_clip(
        str(CLIPS / camera / SWEEP_CLIPS[camera]),
        model,
        cfg["data"]["input_size"],
        device,
        tracker,
        frames=frames,
        preprocess=preprocess,
        probe=probe,
        fps=args.fps,
        score_thr=args.score_thr,
        k1=cam_file.lens.k1 if cam_file.lens else None,
        max_frames=args.frames,
    )
    src = (out.src_w, out.src_h)
    counts = {"exit": 0, "lost": 0, "gone": 0}
    lengths: dict[str, list[int]] = {k: [] for k in counts}
    for t in out.tracks:
        if not t.frames:
            continue
        # A track still alive at the last frame has not ended; its ending is unobserved.
        if t.frames[-1] >= out.frames - 1 - args.max_age:
            continue
        why = classify(t, tracker, cam_file, src, args)
        counts[why] += 1
        lengths[why].append(len(t.frames))
    return {
        "camera": camera,
        "tracks": len(out.tracks),
        "endings_judged": sum(counts.values()),
        "counts": counts,
        "median_length": {k: (int(np.median(v)) if v else None) for k, v in lengths.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/endings01"))
    ap.add_argument("--cameras", nargs="*", default=sorted(SWEEP_CLIPS))
    ap.add_argument("--config", default="configs/hydranet_retail_pose01.yaml")
    ap.add_argument(
        "--checkpoint",
        default="runs/hydranet_retail_security_b03_cw_xl-20260825-162131/last.pt",
    )
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--gap", type=int, default=10, help="frames to look for a reappearance")
    ap.add_argument("--near", type=float, default=1.0, metavar="M")
    ap.add_argument("--edge-px", type=float, default=8.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(None)
    cfg = load_config(args.config)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), "ema"))

    fleet, tot = [], {"exit": 0, "lost": 0, "gone": 0}
    for camera in args.cameras:
        row = run_camera(camera, model, cfg, device, args)
        for k, v in row["counts"].items():
            tot[k] += v
        fleet.append(row)
        c = row["counts"]
        print(
            f"{camera:<18} {row['endings_judged']:>3} endings   "
            f"exit {c['exit']:>3}   lost {c['lost']:>3}   gone {c['gone']:>3}"
        )
    n = max(sum(tot.values()), 1)
    print(
        f"\nfleet: {n} endings   "
        + "   ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in tot.items())
    )
    (args.out / "fleet.json").write_text(
        json.dumps(
            {"settings": report_settings(args), "cameras": fleet, "total": tot}, indent=1
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
