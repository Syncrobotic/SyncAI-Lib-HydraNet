#!/usr/bin/env python3
"""Step 6: the L3 event log for one **commissioned** camera, and the alert rows it files.

    python3 scripts/step6_events.py --camera Taichung-cam01 \\
        --loiter-seconds 8 --out runs/step6_01

docs/PLAN.md step 6 asks for "an event log readable against its video", and section 9.6
records what was missing: nothing anywhere wrote an alert row. `serving/dispositions.py`
is a complete schema -- append-only JSONL, operator verdict joined by `alert_id` -- with
no store on disk, and section 4.5 calls that store the human test set this project's
success number comes out of. This is the runner that starts filling it.

---------------------------------------------------------------------------
WHAT THIS IS NOT: `scripts/site_events.py`

That script ran the same chain in 2026-08 and it is still the right thing to read for the
pose argument. Two differences, and both are the reason this file exists:

* **Its camera pose was fitted by hand for one camera** and its zone was *derived from the
  footage* -- "the busiest square metre of floor in this clip", which it labels a
  demonstration rather than a product zone. This one reads a commissioned `camera.json`:
  a fitted pose, a lens, and the **71 service zones across 8 cameras** that
  `tools/commissioning/service_zones.py` produced, where a zone is the floor *beside* a
  fixture because a footprint is a region no shopper can stand in.
* **It filed no alerts.** Every event here goes through `dispositions.record_alert`, which
  is what makes an event gradeable months later: the row carries `basis`/`value`/
  `threshold`, the frame it can be pulled from, the checkpoint and commit that raised it,
  and the hash of the calibration the geometry ran under.

---------------------------------------------------------------------------
THE POLICY IS NOT IN THE CAMERA FILE, AND THAT IS ON PURPOSE

`events.zones_from_camera` bridges the two `Zone` types and carries **geometry only** --
every policy field comes back at its default, so a commissioned camera on its own fires
nothing at all. docs/PLAN.md section 5 is why: "four minutes is loitering" is an argument
a store manager changes on a Tuesday, not a property of the mount.

So the thresholds come from a **policy file** (`--policy`, `analytics/policy.py`), which
the serving path reads too, so the offline log and the live one fire on the same numbers.
The default file, `configs/policy/demo.yaml`, holds the demonstration values this script
carried as argument defaults until 2026-09-10 -- **not a store's rules**, and the file
says so in its `provenance`. A real policy arrives from whoever owns the store's rules,
as a second file.

---------------------------------------------------------------------------
TWO PIXEL-FRAME TRAPS, BOTH LIVE, BOTH SILENT

1. **`track_clip` returns boxes in the decoded stream's pixels and the commissioned
   intrinsics are fitted on half that** on several cameras (Taichung-cam01: 960x540
   against 1920x1080). `zone_events` projects through `dwell.track_ground_path`, which
   takes `(cam, plane)` and so has nowhere to put a source size. Feeding it stream pixels
   returns metres -- the 2026-08-26 failure `analytics/world.py` was written after, where
   three shoppers came back several metres outside the walkable polygon with no NaN
   anywhere. So the tracks are rescaled to the calibrated frame **once, here**, before any
   of them reaches an event function.

2. **These tracks must never be handed to `world.world_frame`.** `track_clip` undistorts
   upstream (its `k1` is keyword-only with no default precisely so a caller cannot
   forget), which is the precondition `track_ground_path` documents and relies on --
   while `world_frame` undistorts *internally*, from raw stream pixels. The same tracks
   through both paths get the lens applied twice. The two contracts are genuinely
   different and neither is wrong; this runner picks the event layer's, because the event
   layer is what step 6 is about.

---------------------------------------------------------------------------
THE TRACKER RUNS THE SHIPPED BAND, AND THE FIRST FLEET RUN DID NOT

`runs/step6_fleet01` (2026-09-09, the 263 rows) tracked at a single threshold: decode at
0.35, birth at 0.35, nothing survives below it. The serving path had already moved off
that point -- `serving/camera.py` ships `person` at birth 0.35 / keep 0.20 and
`serve_pilot.py` builds its tracker from the pair -- and the survival band was measured
on its own before that (`runs/band_probe01`, PLAN 7.11): 202 tracks -> 109 over the same
eight clips, coasted fraction 0.0643 -> 0.0167, 91% of what bytetrack's band-plus-Kalman
recovers with no filter and no change to association. So the L3 log was being written by
a tracker one operating point behind the one that ships, and `occupancy_events` says in
its own docstring what that costs: it counts tracks, so it over-counts by exactly the
fragmentation rate.

Now the default: decode at `--keep-thr` so the low boxes reach the tracker, birth at
`--score-thr` inside it. `--single-threshold` reproduces the first run's tracker, and it
exists so the two logs can be read against each other rather than so anyone runs it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from syncai_hydranet import shipped  # noqa: E402
from syncai_hydranet.analytics import events as ev  # noqa: E402
from syncai_hydranet.analytics.clip_tracks import track_clip  # noqa: E402
from syncai_hydranet.analytics.delivery import report_settings  # noqa: E402
from syncai_hydranet.analytics.policy import load_policy  # noqa: E402
from syncai_hydranet.analytics.tracker import Track, Tracker  # noqa: E402
from syncai_hydranet.data.video import frames, probe  # noqa: E402
from syncai_hydranet.geometry.camera_json import CameraFile  # noqa: E402
from syncai_hydranet.geometry.ground import (  # noqa: E402
    FrameBounds,
    distort_points,
    height_above_floor_m,
    pixel_to_ground,
)
from syncai_hydranet.serving import dispositions as dp  # noqa: E402
from syncai_hydranet.serving.camera import BIRTH_REF, KEEP_REF  # noqa: E402
from syncai_hydranet.utils.visualize import preprocess  # noqa: E402

DEFAULT_COMMISSION = ROOT / "runs/commission01"
DEFAULT_POLICY = ROOT / "configs/policy/demo.yaml"


def to_calibrated(tracks: list[Track], src_w: int, src_h: int, cam_file: CameraFile):
    """Stream-pixel tracks -> the frame the intrinsics were fitted on. See trap 1 above.

    A uniform scale, because that is all the two frames differ by -- and it is applied to
    `velocity` as well as the boxes, since `speed_events` reads it and a half-scale
    velocity is a shopper walking at half speed, which is a threshold crossing that
    silently does not happen.
    """
    w, h = cam_file.image_size_px
    s = np.array([w / float(src_w), h / float(src_h)])
    if not np.allclose(s, s[0]):
        raise SystemExit(
            f"{cam_file.camera_id}: the stream is {src_w}x{src_h} and the calibration is "
            f"{w}x{h}, which is not a uniform scale. A non-uniform one is a different "
            "aspect ratio, and rescaling across it would silently distort every metre."
        )
    box_s = np.concatenate([s, s])
    return [
        replace(
            t,
            box=np.asarray(t.box, float) * box_s,
            boxes=[np.asarray(b, float) * box_s for b in t.boxes],
            velocity=np.asarray(t.velocity, float) * s,
        )
        for t in tracks
    ]


def observations(tracks, cam_file: CameraFile, camera: str, clip: str) -> dict:
    """Per observed track-frame, the quantities the near-boundary decision needs.

    The open question this exists to answer, from the 2026-09-09 review of this script's
    own first two alerts: a person whose feet are outside the frame still gets a box, the
    detector ends it somewhere plausible **above** the frame edge -- measured 13-30 px on
    Taichung-cam01 -- and its foot point lands on the nearest floor the camera can see. A
    pixel-margin edge gate does not catch that, and the buffer that would has no
    defensible value until this distribution exists across the fleet.

    So: `bottom_gap` is the distance from the box bottom to the frame bottom **in the raw
    frame**, which means the foot point is put back through the lens first -- the boxes
    here are undistorted and the frame edge is a fact about the distorted frame. The
    fraction is carried beside the pixels because cameras are calibrated at different
    resolutions and only the fraction is comparable.
    """
    lens = cam_file.lens
    _, h = cam_file.image_size_px
    rows: dict[str, list] = {k: [] for k in
        ("track_id", "frame", "bottom_gap_px", "bottom_gap_frac", "x_m", "z_m",
         "height_m", "score")}  # fmt: skip
    for t in tracks:
        if not t.boxes:
            continue
        b = np.stack(t.boxes)
        u, v = (b[:, 0] + b[:, 2]) / 2, b[:, 3]
        raw = np.stack([u, v], -1)
        if lens is not None and abs(lens.k1) > 1e-12:
            raw = distort_points(raw, lens.k1, lens.centre_px, lens.radius_px)
        gap = h - raw[:, 1]
        x, z = pixel_to_ground(u, v, cam_file.camera, cam_file.plane)
        cam, plane = cam_file.camera, cam_file.plane
        hh = np.array([
            height_above_floor_m(float(xi), float(zi), float(vt), cam, plane)
            for xi, zi, vt in zip(x, z, b[:, 1], strict=True)
        ])  # fmt: skip
        scores = list(t.scores) + [np.nan] * (len(b) - len(t.scores))
        rows["track_id"] += [t.track_id] * len(b)
        rows["frame"] += list(map(int, t.frames))
        rows["bottom_gap_px"] += list(map(float, gap))
        rows["bottom_gap_frac"] += list(map(float, gap / h))
        rows["x_m"] += list(map(float, x))
        rows["z_m"] += list(map(float, z))
        rows["height_m"] += list(map(float, hh))
        rows["score"] += list(map(float, scores[: len(b)]))
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["camera"] = np.array([camera] * len(out["frame"]))
    out["clip"] = np.array([Path(clip).name] * len(out["frame"]))
    return out


def run_clip(camera, cam_file, clip, model, size, device, args, policy) -> dict:
    """One camera, one clip: tracks, events, alert rows, and the observation table."""
    # The band: boxes down to `keep_thr` reach the tracker and may continue a track; only
    # a box at `score_thr` may start one. `single_threshold` is the first fleet run's
    # tracker, kept for reading the two logs against each other -- see the header.
    band = not args.single_threshold
    tracker = Tracker(
        iou_threshold=args.iou,
        max_age=args.max_age,
        min_hits=args.min_hits,
        birth_thr=args.score_thr if band else None,
    )
    out = track_clip(
        clip, model, size, device, tracker,
        frames=frames, preprocess=preprocess, probe=probe,
        fps=args.fps, score_thr=args.keep_thr if band else args.score_thr,
        max_frames=args.max_frames, k1=cam_file.lens.k1,
    )  # fmt: skip
    tracks = to_calibrated(out.tracks, out.src_w, out.src_h, cam_file)
    zones = policy.zones_for(cam_file)

    bounds = None
    if not args.no_edge_gate:
        bounds = FrameBounds(
            *cam_file.image_size_px,
            k1=cam_file.lens.k1,
            centre_px=cam_file.lens.centre_px,
            radius_px=cam_file.lens.radius_px,
        )
    events = ev.zone_events(
        tracks, zones, cam_file.camera, cam_file.plane, args.fps, camera,
        min_seconds=policy.min_seconds, bounds=bounds,
    )  # fmt: skip
    for zone in zones:
        if zone.max_occupancy is not None:
            events += ev.occupancy_events(
                tracks, zone, cam_file.camera, cam_file.plane, args.fps, camera,
                bounds=bounds,
            )  # fmt: skip
    return {
        "camera": camera,
        "clip": Path(clip).name,
        "frames_read": out.frames,
        "detections": out.detections,
        "tracks": len(tracks),
        "events": events,
        "obs": observations(tracks, cam_file, camera, clip),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", required=True, help='a camera name, or "all"')
    ap.add_argument("--clip", default=None, help="default: the camera's first clip")
    ap.add_argument("--all-clips", action="store_true", help="every clip on disk, per camera")
    ap.add_argument("--camera-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = the whole clip")
    # The shipped birth edge, named rather than restated: `test_shared_constants`
    # holds every `--score-thr` default to one of the three operating points.
    ap.add_argument("--score-thr", type=float, default=BIRTH_REF)
    # The shipped keep edge. A box between the two may continue a track and may not
    # start one; `serving/camera.py` names both edges for the reason it gives there.
    ap.add_argument("--keep-thr", type=float, default=KEEP_REF)
    ap.add_argument(
        "--single-threshold",
        action="store_true",
        help="decode and birth both at --score-thr, no survival band: the tracker "
        "`runs/step6_fleet01` was written with, for comparison and not for use",
    )
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help="the store policy file (analytics/policy.py); the default is the "
        "demonstration policy, which says so in its provenance",
    )
    ap.add_argument(
        "--no-edge-gate",
        action="store_true",
        help="keep foot points the frame bottom decided; the state before the gate",
    )
    args = ap.parse_args()
    if not 0.0 < args.keep_thr <= args.score_thr <= 1.0:
        raise SystemExit(
            f"need 0 < --keep-thr <= --score-thr <= 1, got keep {args.keep_thr} and "
            f"birth {args.score_thr}: a keep edge above the birth edge is a band that "
            "admits nothing"
        )

    policy = load_policy(args.policy)

    # ---- what to run --------------------------------------------------------------
    if args.camera == "all":
        if args.camera_file or args.clip:
            raise SystemExit("--camera all takes neither --camera-file nor --clip")
        cams = sorted(
            p.name[: -len(".camera.json")] for p in DEFAULT_COMMISSION.glob("*.camera.json")
        )
    else:
        cams = [args.camera]

    jobs = []
    for cam in cams:
        cam_path = args.camera_file or (DEFAULT_COMMISSION / f"{cam}.camera.json")
        if not cam_path.is_file():
            raise SystemExit(f"{cam_path} does not exist -- is {cam} commissioned?")
        cam_file = CameraFile.load(cam_path)
        if cam_file.lens is None:
            # Not skipped quietly: `track_clip`'s k1 is keyword-only with no default so a
            # caller cannot forget the lens, and guessing one here would defeat that.
            print(f"  skipping {cam}: {cam_path.name} has no lens, so k1 cannot be stated")
            continue
        found = (
            [Path(args.clip)]
            if args.clip
            else sorted((ROOT / f"datasets/studioa_clips/{cam}").glob("*.mp4"))
        )
        if not found:
            print(f"  skipping {cam}: no clips on disk")
            continue
        for clip in found if (args.all_clips or args.clip) else found[:1]:
            jobs.append((cam, cam_path, cam_file, str(clip)))
    if not jobs:
        raise SystemExit("nothing to run")

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = shipped.for_detection()
    model, cfg, device = shipped.load_model(shipped.SHIPPED_CONFIG, checkpoint, validate=False)
    size = cfg["data"]["input_size"]
    # `model_identity` takes the CHECKPOINT first and the config second. Reversed, it
    # records the config path as the checkpoint and hashes the checkpoint as the config,
    # and the row reads as a perfectly ordinary one -- caught here only by reading the
    # written store back. Both paths are repo-relative because the row is long-lived and
    # `analytics/delivery.py` names the cost of the absolute form: it carries an operator,
    # a home directory and a checkout into a file that looks like output.
    model_id = dp.model_identity(checkpoint.resolve().relative_to(ROOT), shipped.SHIPPED_CONFIG)

    # ---- run ----------------------------------------------------------------------
    rows, obs_parts, all_events = [], [], []
    t0 = time.perf_counter()
    for n, (cam, cam_path, cam_file, clip) in enumerate(jobs, 1):
        print(f"[{n}/{len(jobs)}] {cam}  {Path(clip).name}", flush=True)
        r = run_clip(cam, cam_file, clip, model, size, device, args, policy)
        records = [
            dp.record_alert(args.out / "dispositions", e, model=model_id,
                            calib=cam_path, clip=Path(clip).resolve().relative_to(ROOT))
            for e in r["events"]
        ]  # fmt: skip
        obs_parts.append(r.pop("obs"))
        events = r.pop("events")
        all_events += [(cam, e) for e in events]
        rows.append({**r, "events": [e.as_row() for e in events],
                     "alerts_filed": [x.alert_id for x in records]})  # fmt: skip
        for e in events:
            print(f"      {e.type:16s} {e.zone or '-':16s} tracks {e.track_ids} "
                  f"{e.frame_start / args.fps:7.1f}-{e.frame_end / args.fps:7.1f}s  "
                  f"value {e.value:.2f} vs {e.threshold}")  # fmt: skip
    elapsed = time.perf_counter() - t0

    # ---- what came out ------------------------------------------------------------
    obs = {k: np.concatenate([o[k] for o in obs_parts]) for k in obs_parts[0]}
    # A `**dict[str, ndarray]` splat could in principle supply `allow_pickle`, so the
    # checker unifies every value against its `bool`. The code is right and the
    # alternative -- naming ten arrays by hand -- is the copy this dict exists to avoid.
    np.savez_compressed(
        args.out / "observations.npz",
        **obs,  # ty: ignore[invalid-argument-type]
    )

    by_camera: dict[str, dict] = {}
    for r in rows:
        c = by_camera.setdefault(r["camera"], {"clips": 0, "frames": 0, "detections": 0,
                                               "tracks": 0, "events": 0})  # fmt: skip
        c["clips"] += 1
        c["frames"] += r["frames_read"]
        c["detections"] += r["detections"]
        c["tracks"] += r["tracks"]
        c["events"] += len(r["events"])
    kinds: dict[str, int] = {}
    for _, e in all_events:
        kinds[e.type] = kinds.get(e.type, 0) + 1

    fleet = {
        **report_settings(
            args,
            policy={
                "store": policy.store,
                "provenance": policy.provenance,
                "min_seconds": policy.min_seconds,
                "open_hours": policy.open_hours,
                "zones": {
                    k: {
                        f: getattr(r, f)
                        for f in ("loiter_seconds", "max_occupancy", "restricted")
                    }
                    for k, r in policy.rules.items()
                },
            },
        ),
        "seconds": round(elapsed, 1),
        "clips": len(jobs),
        "cameras": len(by_camera),
        "footage_seconds": round(sum(r["frames_read"] for r in rows) / args.fps, 1),
        "events_by_type": kinds,
        "per_camera": by_camera,
        "clip_reports": rows,
    }
    (args.out / "fleet.json").write_text(json.dumps(fleet, indent=2, default=str) + "\n")

    print(f"\n{len(jobs)} clips over {len(by_camera)} cameras in {elapsed / 60:.1f} min")
    print(f"{'camera':18s} {'clips':>5} {'minutes':>8} {'dets':>7} {'tracks':>7} {'events':>7}")
    for cam, c in sorted(by_camera.items()):
        print(f"{cam:18s} {c['clips']:5d} {c['frames'] / args.fps / 60:8.1f} "
              f"{c['detections']:7d} {c['tracks']:7d} {c['events']:7d}")  # fmt: skip
    print(f"\nevents by type: {kinds or '{}'}")
    print(f"observations:   {len(obs['frame'])} rows -> {args.out / 'observations.npz'}")
    print(f"alerts:         {sum(len(r['alerts_filed']) for r in rows)} "
          f"in {args.out / 'dispositions'}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
