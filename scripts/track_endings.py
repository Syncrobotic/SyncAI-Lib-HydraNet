#!/usr/bin/env python3
"""Why do tracks end? One answer holds, and one does not -- both are recorded.

    python3 scripts/track_endings.py --out runs/endings03

`runs/journeys01` measured the consequence: 197 visits, a median of 3.2 s, and eight in 24
minutes lasting 30 s or more. The cause is that tracks are shorter than visits, and
`reid_metrics.idf1`/`id_switches` cannot say why -- both need ground-truth tracks, and
that module's own header records that no labelled site clip exists. Labelling one is the
human cost PLAN section 4 refuses by default. A track's own ending is evidence and it is
free.

---------------------------------------------------------------------------
WHAT THIS MEASURES RELIABLY: WHERE THE TRACK DIED

If the last box sits against a frame edge the person walked out and the track was supposed
to end (**exit**). Otherwise the track died in the middle of the view, and a shopper does
not vanish from a shop floor. That split does not depend on any matching rule, and it did
not move across three runs: **80 of 191 endings are exits (42%), so 58% die mid-view.**

Per camera it is not uniform and that is the finding:

    Tao-Hsin-cam03      35 endings    83% died mid-view
    Kaohsiung-cam04     85 endings    78%
    Taichung-cam01      13 endings    31%
    Taichung-cam11       6 endings    17%
    Taichung-cam10      14 endings    14%
    Taichung-cam04      30 endings    13%

Taichung-cam04 loses thirteen percent of its tracks mid-view; Tao-Hsin-cam03 loses
eighty-three. The tracker is the same tracker. Two cameras carry this, and one of them has
had an open person-score investigation since PLAN step 2.

---------------------------------------------------------------------------
WHAT THIS DOES NOT MEASURE, AND THE THREE RUNS THAT ESTABLISHED THAT

A mid-view death is either the tracker dropping someone the detector still sees
(**lost**) or the detector losing them with nothing to reacquire (**gone**). Separating
those needs a rule for "the same person reappeared", and position alone does not carry one:

    flat 1.0 m radius          exit 80   lost  72   gone 39
    flat 1.0 m radius (rerun)  exit 80   lost  72   gone 39
    0.35 m per frame of gap    exit 80   lost 101   gone 10

`exit` is identical every time. The other two trade places wholesale, because both rules
are wrong in ways that only showed up in the detail. The radius matched the **neighbouring
shopper**: its "reappearances" sat a median 0.71 m away one frame later, which is 3.5 m/s,
and on a camera carrying 87 tracks in three minutes there is always somebody within a
metre. The speed limit fixed that at gap 1 and broke the far end -- 0.35 m per frame is
1.05 m at gap 3 and 3.5 m at gap 10, wider than the radius it replaced, across most of a
small shop.

**The honest conclusion is that this question is not decidable from geometry.** Telling
"the same person, one metre on" from "a different person, one metre away" is what an
appearance model is for; `reid_metrics.cmc_map` is the metric for one and PLAN step 5 is
where it belongs. `lost` and `gone` are still reported, and `lost_detail` carries every
gap, IoU and floor distance, so a later rule can be tried against the same endings -- but
nothing should be concluded from that split until something can tell two shoppers apart.
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
from syncai_hydranet.analytics.tracker import Tracker, iou  # noqa: E402
from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.data.video import frames, probe  # noqa: E402
from syncai_hydranet.geometry.camera_json import CameraFile  # noqa: E402
from syncai_hydranet.geometry.ground import pixel_to_ground, undistort_points  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import preprocess  # noqa: E402

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


def classify(track, tracker: Recording, cam_file, src, args) -> dict:
    """The ending, plus what a `lost` one costs to recover.

    A `lost` ending is two different failures wearing one label, and they have different
    fixes. If the person reappears on the **next** frame with a box that overlaps the dead
    track, the tracker had an easy match and did not take it -- an association failure,
    and the association logic is the thing to change. If the reappearance is six or more
    frames later, the track was killed by `max_age`, which at 5 fps is one second of
    coasting: the detector lost the person for longer than the tracker is willing to wait,
    and no association rule reaches that. So `gap` and `iou` come back with the verdict.
    """
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
        return {"why": "exit"}
    here = foot_m(box[None], cam_file, src)[0]
    if not np.isfinite(here).all():
        return {"why": "gone"}
    for f in range(last_frame + 1, last_frame + 1 + args.gap):
        boxes = tracker.seen.get(f)
        if boxes is None or not len(boxes):
            continue
        d = np.linalg.norm(foot_m(boxes, cam_file, src) - here, axis=1)
        # The speed limit, not a radius: how far this person could have walked since
        # their last observation. See the module docstring for what a radius did instead.
        if not np.isfinite(d).any() or float(np.nanmin(d)) > args.speed * (f - last_frame):
            continue
        j = int(np.nanargmin(d))
        return {
            "why": "lost",
            # Frames from the last observation to the reappearance. `max_age` is the
            # tracker's patience; a gap larger than it was never recoverable by matching.
            "gap": f - last_frame,
            "iou": round(float(iou(box[None], boxes[j : j + 1])[0, 0]), 3),
            "metres": round(float(d[j]), 2),
        }
    return {"why": "gone"}


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
    lost: list[dict] = []
    lengths: dict[str, list[int]] = {k: [] for k in counts}
    for t in out.tracks:
        if not t.frames:
            continue
        # A track still alive at the last frame has not ended; its ending is unobserved.
        if t.frames[-1] >= out.frames - 1 - args.max_age:
            continue
        v = classify(t, tracker, cam_file, src, args)
        counts[v["why"]] += 1
        lengths[v["why"]].append(len(t.frames))
        if v["why"] == "lost":
            lost.append({"track_id": t.track_id, **{k: v[k] for k in ("gap", "iou", "metres")}})
    return {
        "camera": camera,
        "tracks": len(out.tracks),
        "endings_judged": sum(counts.values()),
        "counts": counts,
        "median_length": {k: (int(np.median(v)) if v else None) for k, v in lengths.items()},
        # Every lost ending, so the two failures wearing that one label stay countable:
        # a detection available on the next frame that the tracker did not match, against
        # a detector outage longer than `max_age` that no matching rule could reach.
        "lost_detail": lost,
        "max_age": args.max_age,
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
    ap.add_argument(
        "--speed",
        type=float,
        default=0.35,
        metavar="M_PER_FRAME",
        help="how far a person may have walked between frames; 0.35 at 5 fps is 1.75 m/s",
    )
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
