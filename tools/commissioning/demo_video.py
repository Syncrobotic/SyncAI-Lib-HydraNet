"""The three-minute demo: detections and tracks on the left, the mesh scene walked
through on the right.

The right panel is the standing instruction: the solid-furniture scene
(`scene_mesh.build_scene_regular`), not the flat ribbon panel -- with a mesh figure
placed at every confirmed track's floor position, so a shopper's motion is visible in
metres in the same picture that shows the furniture they walk between. The camera.json
false-positive polygons are applied to the detections before tracking, which is those
polygons doing their production job for the first time. The floor also carries the
LIVE dwell field: amber tiles that accumulate occupancy-seconds as the video plays,
`heatmap3d.py`'s exact construction through `syncai_bev3d.heatmap` (walkable-clipped,
p99-scaled), disabled with --no-heatmap.

A figure standing behind a 2 m shelf is *correctly* invisible from any one fixed
viewpoint -- three eye positions were rendered and the far shopper hid behind the same
shelf in all three. So each figure is drawn twice: once in depth order, and once as a
translucent x-ray pass over the top. Occlusion still reads, but a tracked person is
never absent from the panel that exists to show tracked people.

Known limit of the fixed 3/4 view, and it is perspective rather than a bug: a floor
position *behind* a waist-high counter projects onto that counter's top face, and the
figure is genuinely nearer the eye than those faces, so it reads as standing on the
counter. Rendered at ghost alpha 0 it reads the same way. Disambiguating it wants
per-pixel depth or a drop-line to the floor; `<camera>.demo_tracks.json` carries the
metre position of every figure meanwhile, which is what settles such a question.

Usage:
  uv run python tools/commissioning/demo_video.py <camera> [--clip PATH]
      [--frames 900] [--fps 5] [--checkpoint last.pt] [--score-thr BIRTH_REF]

Writes assets/dev/demo_<camera>_<stamp>.mp4 and a stable assets/dev/demo_<camera>.mp4
beside it (both ignored wholesale -- customer footage), plus three sample frames for the
frame-check. `assets/` itself holds only results; a render is working material, and the
result is the figure `demo_gif.py` cuts from it. The mp4 is written to `.part` and
renamed on success, so a killed render leaves no file rather than a truncated one that
ffprobe cannot open.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from syncai_bev3d import scene_mesh
from syncai_bev3d.figures import (
    CUSTOMER_COLOR,
    PLACE_MARGIN_M,
    PLUMB_W_M,
    STAFF_COLOR,
    TRACK_COLORS,
    VEL_FLOOR_MS,
    VEL_SECONDS_SHOWN,
    VEL_WINDOW_S,
    _torso_crop,
    box_iou,
    content_crop,
    facing_wedge,
    in_fp_zone,
    lift_fronto_parallel,
    point_in_walkable,
    track_colour,
    track_states,
    velocity_arrow,
    walkable_bounds,
)
from syncai_bev3d.heatmap import DEFAULT_CELL_M as HEAT_CELL_M
from syncai_bev3d.heatmap import LEVELS as HEAT_LEVELS
from syncai_bev3d.heatmap import heat_colour, tile_specs
from syncai_bev3d.heatmap import normalise as heat_normalise
from syncai_bev3d.meshes import (
    Placement,
    box,
    extrude,
    ground_disc,
    human,
    human_posed,
    place,
)
from syncai_bev3d.shading import draw_scene
from syncai_hydranet.analytics.staff import (
    StaffModel,
    require_camera,
    track_staff,
)
from syncai_hydranet.analytics.tracker import Tracker
from syncai_hydranet.config import load_config
from syncai_hydranet.data.video import frames as decode_frames
from syncai_hydranet.data.video import probe as probe_video
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.serving.camera import BIRTH_REF
from syncai_hydranet.shipped import SHIPPED_RUN
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.device import pick_device
from syncai_hydranet.utils.face_blur import (
    BLUR_THR,
    blur_region,
    plate_person_boxes,
)
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")


def _display_verdict(track):
    """The figure-colour verdict: floor 3 = the tracker's confirmation delay, so the
    verdict exists at a figure's first drawn frame and the green->blue pop measured at
    46 per 900 frames goes to zero. See `track_staff` for the measured cost."""
    return track_staff(track, min_observations=3)


# The run the tools ship from, named once in `syncai_hydranet.shipped`. Six files
# used to carry their own copy of this string and the best run was in none of them.
RUN = SHIPPED_RUN


def _records_pass(
    args, clip, src_w, src_h, size, model, device, cf, person_label, rng, staff_model
):
    """Every frame's model outputs the render loop consumes, recorded for replay.

    The pre-pass a chunked render needs, `heads_video`'s pattern with one improvement
    this tool's shape allows: the record carries the BLUR boxes and (when posed) the
    keypoints too, so a render worker needs no model at all -- its whole job is
    decode, blur, draw, encode. **Nothing is rounded**: boxes replay into the tracker,
    the EMA and the stature medians, and a rounded box was measured to move one frame
    in twenty-four under a lossless encoder.

    Staff probabilities are recorded here, not left to the workers: a verdict is
    accumulated over a track's whole life, and they are read off the RAW frame because
    the face blur covers the torso band the classifier reads.
    """
    lo, hi = rng if rng else (0, args.frames)
    out = []
    for n, frame in enumerate(decode_frames(str(clip), src_w, src_h, args.fps)):
        if n >= hi:
            break
        if n < lo:
            continue
        img = Image.fromarray(frame)
        x, _canvas, region = preprocess(img, size)
        with torch.no_grad():
            res = model.predict(x.to(device), score_thr=min(BLUR_THR, args.score_thr))
        det = res.get("detection", [{}])[0]
        boxes = np.zeros((0, 4), np.float32)
        kps = None
        blur: list[list[float]] = []
        if det and len(det.get("boxes", [])):
            b_all = det["boxes"].cpu().numpy()
            lab = det["labels"].cpu().numpy()
            sc = det["scores"].cpu().numpy()
            x0, y0, cw, _ch = region
            b_all = (b_all - np.array([x0, y0, x0, y0])) * (src_w / cw)
            person = lab == person_label
            blur = [list(map(float, v)) for v in b_all[person & (sc >= BLUR_THR)]]
            b = b_all[person & (sc >= args.score_thr)]
            keep = [
                i
                for i, bb in enumerate(b)
                if not in_fp_zone(cf, (bb[0] + bb[2]) / 4, (bb[1] + bb[3]) / 4)
            ]
            boxes = b[keep]
            pose_rows = res.get("pose", [None])[0]
            if pose_rows is not None and args.posed_figures:
                kp = pose_rows.cpu().numpy()[person & (sc >= args.score_thr)][keep].copy()
                kp[:, :, 0] = (kp[:, :, 0] - x0) * (src_w / cw)
                kp[:, :, 1] = (kp[:, :, 1] - y0) * (src_w / cw)
                kps = np.asarray(kp, np.float64).tolist()
            n_dropped = len(b) - len(keep)
        else:
            n_dropped = 0
        sp = (
            None
            if staff_model is None
            else [
                float(staff_model.probability_of_crop(_torso_crop(frame, bb))) for bb in boxes
            ]
        )
        out.append(
            {
                "boxes": np.asarray(boxes, np.float64).tolist(),
                "staff": sp,
                "blur": blur,
                "kps": kps,
                "n_det": int(len(boxes) + n_dropped),
                "n_fp": int(n_dropped),
            }
        )
        if len(out) % 200 == 0:
            print(f"  records {lo + len(out)}/{hi}", flush=True)
    return out


def _write_demo_tracks(args, camera, clip, staff_model, n_frames, positions) -> None:
    """The track log that makes the right-hand panel checkable instead of merely
    convincing: every figure in the video has a frame, an id and a floor position in
    metres here. Written by the single-process path and the chunk orchestrator alike --
    one function, so the two records cannot drift."""
    log_path = ROOT / f"runs/commission01/{camera}.demo_tracks.json"
    log_path.write_text(
        json.dumps(
            {
                # **Every argument, wholesale, and not a hand-picked list.** This record
                # held eight fields against a parser of ten, and on 2026-08-29 that cost
                # a figure: a second session re-cut `demo_Taichung-cam10` without
                # `--staff-colours` -- undoing `65a6b78`, which made it two-colour on
                # purpose -- and nothing in the verdict named the flag, so the audit, the
                # blur check and the staleness guard all passed on a figure that had
                # silently changed. `--metre-scale` was missing on the same principle and
                # is worse: `positions` below are in metres and two runs at 1.0 and
                # 0.8824 produce different numbers under identical-looking provenance.
                # An allowlist is broken again the day flag eleven lands;
                # `tests/test_figures_are_audited.py` reads the parser and checks this.
                "args": vars(args),
                # And what the arguments RESOLVED to, because a path is not identity:
                # `runs/` is gitignored and regenerable, so the same model file can be
                # refitted at the same path with a different accuracy and no record
                # would move. The figure's own legend prints this number.
                "staff_model": None
                if staff_model is None
                else {
                    "path": args.staff_colours,
                    "sha256": hashlib.sha256(Path(args.staff_colours).read_bytes()).hexdigest(),
                    "accuracy": staff_model.accuracy,
                    "held_out": staff_model.held_out,
                    "held_out_n": staff_model.held_out_n,
                    "min_accuracy_required": args.staff_min_accuracy,
                },
                "camera": camera,
                "clip": clip.name,
                "checkpoint": args.checkpoint,
                "score_thr": args.score_thr,
                # The threshold the faces were blurred at, recorded because an auditor
                # has to reconstruct the blur set THIS render applied. Reading the
                # current constant instead answers "would today's threshold have covered
                # them", which is a different and more flattering question -- and the
                # constant moved on 2026-08-28 (0.10 -> 0.07), so the two now differ.
                "blur_score_thr": BLUR_THR,
                "blur_faces": not args.no_blur,
                "fps": args.fps,
                "frames": n_frames,
                "positions": positions,
            },
            indent=2,
        )
        + "\n"
    )


def _worker_cmd(args, camera):
    """This script re-invoked with the caller's own flags, minus the fan-out ones."""
    cmd = [sys.executable, str(Path(__file__).resolve()), camera,
           "--frames", str(args.frames), "--fps", str(args.fps),
           "--checkpoint", args.checkpoint, "--score-thr", str(args.score_thr),
           "--metre-scale", str(args.metre_scale)]  # fmt: skip
    if args.clip:
        cmd += ["--clip", args.clip]
    if args.no_blur:
        cmd.append("--no-blur")
    if args.posed_figures:
        cmd.append("--posed-figures")
    if args.staff_colours:
        cmd += ["--staff-colours", args.staff_colours]
        if args.staff_min_accuracy is not None:
            cmd += ["--staff-min-accuracy", str(args.staff_min_accuracy)]
    return cmd


def _render_in_chunks(args, camera, clip, cf, bounds, staff_model) -> int:
    """Fan the frames out over processes, then join the segments back into one file.

    `heads_video`'s orchestration shape: a fanned-out records pass (each worker runs
    the model over a frame range), then render workers that REPLAY the track state
    from the merged record and draw only their own chunk -- no model loaded, which is
    what makes a demo render worker cheap enough that the fan-out is nearly linear.
    """
    # The decoded size the recorded boxes are in, so the replay converts to calibrated
    # pixels the same way the recording pass did.
    src_w, src_h, _ = probe_video(str(clip))
    t0 = time.time()
    per = math.ceil(args.frames / args.workers)
    edges = [(i, min(i + per, args.frames)) for i in range(0, args.frames, per)]
    dev = ROOT / "assets/dev"
    r_paths, procs = [], []
    for k, (lo, hi) in enumerate(edges):
        rp = dev / f"_demo_{camera}_{os.getpid()}_rec{k:02d}.json"
        r_paths.append(rp)
        procs.append(
            subprocess.Popen(
                [*_worker_cmd(args, camera), "--chunk", f"{lo}:{hi}", "--records-only", str(rp)]
            )
        )
    if any(pr.wait() for pr in procs):
        for rp in r_paths:
            rp.unlink(missing_ok=True)
        print(f"{camera}: a records pass failed; nothing written", file=sys.stderr)
        return 1
    records = []
    for rp in r_paths:
        records += json.loads(rp.read_text())
        rp.unlink(missing_ok=True)
    rec_path = dev / f"_demo_{camera}_{os.getpid()}.records.json"
    rec_path.write_text(json.dumps(records))
    print(f"  records pass: {len(records)} frames in {time.time() - t0:.0f} s", flush=True)

    segs, procs = [], []
    for k, (lo, hi) in enumerate(edges):
        seg = dev / f"_demo_{camera}_{os.getpid()}_seg{k:02d}.mp4"
        segs.append(seg)
        procs.append(
            subprocess.Popen(
                [
                    *_worker_cmd(args, camera),
                    "--chunk",
                    f"{lo}:{hi}",
                    "--chunk-out",
                    str(seg),
                    "--records",
                    str(rec_path),
                ]
            )
        )
    codes = [pr.wait() for pr in procs]
    if any(codes):
        rec_path.unlink(missing_ok=True)
        for seg in segs:
            seg.unlink(missing_ok=True)
        print(f"{camera}: a chunk failed with {codes}; nothing written", file=sys.stderr)
        return 1

    stamp = time.strftime("%Y%m%d-%H%M%S")
    final = dev / f"demo_{camera}_{stamp}.mp4"
    listing = dev / f"_demo_{camera}_{os.getpid()}.concat.txt"
    listing.write_text("".join(f"file '{s}'\n" for s in segs))
    rc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(final)],
    ).returncode  # fmt: skip
    listing.unlink(missing_ok=True)
    for seg in segs:
        seg.unlink(missing_ok=True)
    if rc != 0:
        rec_path.unlink(missing_ok=True)
        print(f"{camera}: concat failed", file=sys.stderr)
        return 1
    rec_path.unlink(missing_ok=True)

    # The sidecar and summary come from replaying the records through the same
    # `track_states` the workers ran -- one input, one function, so the record cannot
    # disagree with the picture.
    # staff-memory on in BOTH the render and its replay, or a chunked render's
    # colours diverge from a single-process one. 15/0.2 measured 2026-09-03:
    # warm-up colour pops 66->46, zero contamination in 15 decidable pairs.
    tracker = Tracker(staff_memory_gap=15, staff_memory_iou=0.2)
    state: dict = {
        "smoothed": {},
        "statures": {},
        "history": {},
        "last_heading": {},
        "n_skipped": 0,
    }
    vel_window = max(1, round(VEL_WINDOW_S * args.fps))
    positions, seen_ids = [], set()
    verdict_of = None if staff_model is None else _display_verdict
    for n, rec in enumerate(records):
        b = np.asarray(rec["boxes"], np.float32).reshape(-1, 4)
        sp = None if rec["staff"] is None else np.asarray(rec["staff"], float)
        tracks = [t for t in tracker.update(b, n, staff_scores=sp) if t.hits >= 3]
        for t in tracks:
            seen_ids.add(t.track_id)
        for st in track_states(
            tracks,
            n,
            cf,
            state,
            vel_window,
            args.metre_scale,
            args.fps,
            bounds,
            verdict_of,
            source_size_px=(src_w, src_h),
        ):
            positions.append(
                {
                    "frame": n,
                    "track_id": int(st.track_id),
                    "x_m": round(st.x_m, 3),
                    "z_m": round(st.z_m, 3),
                    "in_walkable": bool(point_in_walkable(cf, st.sm[0], st.sm[1])),
                    "speed_ms": round(st.speed, 3),
                    "stature_m": round(st.stature, 3),
                    "stature_frame_m": (
                        None if not np.isfinite(st.h_one) else round(st.h_one, 3)
                    ),
                    "heading_rad": None if st.heading is None else round(st.heading, 4),
                    "staff": st.verdict,
                }
            )
    latest = dev / f"demo_{camera}.mp4"
    if args.no_blur:
        print(
            f"  --no-blur: {latest.name} left pointing at the last blurred render. "
            f"This file is unpublishable; it is at {final.name} only."
        )
    else:
        latest.write_bytes(final.read_bytes())
    _write_demo_tracks(args, camera, clip, staff_model, len(records), positions)
    print(f"wrote {final} ({len(records)} frames @ {args.fps} fps)")
    if not args.no_blur:
        print(f"  newest also at {latest}")
    n_placed = len(positions)
    print(
        f"  person detections {sum(r['n_det'] for r in records)}, dropped by FP zone "
        f"{sum(r['n_fp'] for r in records)}; track placements {n_placed}, outside "
        f"walkable+{PLACE_MARGIN_M:g}m {state['n_skipped']}; "
        f"{sum(p['in_walkable'] for p in positions)} of {n_placed} inside the walkable "
        f"polygon; {len(seen_ids)} distinct track ids; stature median "
        f"{np.median([p['stature_m'] for p in positions]):.2f} m"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("camera")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=float, default=5.0)
    # last.pt (epoch 60) not best.pt: best.pt is selected on terrain_mIoU and lands on
    # epoch 9, the weakest person detector in the run (site_boxes mAP 0.054 vs 0.122).
    ap.add_argument("--checkpoint", default="last.pt")
    # the model scores people 0.34-0.59 on this camera; 0.5 keeps almost nothing and
    # the tracker's 3-hit confirmation then never fires. Measured, not guessed.
    ap.add_argument("--score-thr", type=float, default=BIRTH_REF)
    # A uniform scale on the whole reconstruction -- scene, zone, people and eye alike --
    # so the rendered image is unchanged and only the units move. 1.0 leaves camera.json's
    # metres exactly as commissioned; 0.8824 is what this camera's own shoppers imply
    # (median recovered stature 1.93 m against the fleet's 1.702 m person).
    ap.add_argument("--metre-scale", type=float, default=1.0)
    # Default ON, and the flag exists to be refused rather than to be convenient: this
    # tool reads `datasets/studioa_clips/`, which is a customer's shop floor, and
    # CONTRIBUTING's assets allowlist says a frame of one cannot be un-published. The
    # README's figure was blurred by hand once and the pipeline was not committed, so
    # nobody could reproduce it; it is code now.
    ap.add_argument(
        "--no-blur",
        action="store_true",
        help="do NOT blur faces -- only for a private check, never for anything shared",
    )
    ap.add_argument(
        "--staff-colours",
        metavar="MODEL_JSON",
        default=None,
        help="colour staff blue and customers green using this `analytics.staff` model "
        "(e.g. runs/staff_model01/model_Kaohsiung-cam04.json). Refused unless the model "
        "was held out on THIS camera and scores at least staff.MIN_DEPLOY_ACCURACY on "
        "it. Without it, figures keep their identity colours.",
    )
    ap.add_argument(
        "--staff-min-accuracy",
        type=float,
        default=None,
        help="lower the held-out accuracy `analytics.staff` requires for this camera. "
        "The default floor is derived (see MIN_DEPLOY_ACCURACY); passing this is the "
        "explicit exception the refusal message asks for, and the number used is printed "
        "and recorded, so a figure never carries a threshold nobody can see.",
    )
    # Standing is the DEFAULT by decision (2026-09-02): the user saw the posed README
    # figures shipped and reverted them the same day. The capability stays -- heads_video
    # poses its figures because showing the pose head is its point -- but this tool's
    # output is the published README figure, and a decision about a published figure is
    # encoded as the default, not as a flag the next re-cut has to remember (the same
    # morning, a re-cut went red for forgetting `--staff-colours`).
    ap.add_argument(
        "--posed-figures",
        action="store_true",
        help="drive each figure from the pose head's keypoints (`meshes.human_posed`) "
        "where they can carry it, standing otherwise. A posed figure shows what the "
        "person is doing and must not be measured -- the lift is the cheap one "
        "(PLAN 7c.30).",
    )
    ap.add_argument(
        "--no-heatmap",
        action="store_true",
        help="do NOT accumulate the amber dwell field on the 3D panel's floor. The "
        "field is heatmap3d.py's construction, live: occupancy-seconds per 0.25 m "
        "cell, p99-scaled, clipped to the walkable polygon.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="render the frames in this many processes -- heads_video's replay "
        "mechanism: a fanned-out records pass runs the model once, then render workers "
        "replay the track state from the record and draw their own chunk with no model "
        "loaded at all.",
    )
    ap.add_argument("--chunk", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--chunk-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--records", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--records-only", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    camera = args.camera

    # Loaded and licensed before a single frame is decoded: an unusable model has to stop
    # the run at second zero, not after a twelve-minute render nobody can trust.
    staff_model = None
    if args.staff_colours:
        from syncai_hydranet.analytics.staff import MIN_DEPLOY_ACCURACY

        # **Written back onto the namespace on purpose, not merely for convenience.**
        # The track log records `vars(args)`, so this line is what makes it record the
        # *effective* floor -- 0.90 when the flag was not passed -- instead of `null`. A
        # cleanup that computes the floor into a local variable would silently turn the
        # record back into "no threshold stated".
        args.staff_min_accuracy = (
            MIN_DEPLOY_ACCURACY if args.staff_min_accuracy is None else args.staff_min_accuracy
        )
        staff_model = require_camera(
            StaffModel.load(args.staff_colours), camera, min_accuracy=args.staff_min_accuracy
        )
        print(
            f"{camera}: staff colours from {args.staff_colours} -- "
            f"{staff_model.accuracy:.3f} held out on this camera "
            f"({staff_model.held_out_n} crops), fitted on {staff_model.n_crops} from "
            f"{len(staff_model.trained_cameras)} other cameras"
        )

    clip = (
        Path(args.clip)
        if args.clip
        else sorted((ROOT / "datasets/studioa_clips" / camera).glob("archive_*11*.mp4"))[0]
    )  # archive names are UTC: *11* is 19:29 local, an open and populated store

    cf = CameraFile.load(ROOT / f"runs/commission01/{camera}.camera.json")
    x_lo, x_hi, z_lo, z_hi = walkable_bounds(cf)  # camera.json metres, like the zone
    if args.workers > 1 and not args.chunk:
        return _render_in_chunks(args, camera, clip, cf, (x_lo, x_hi, z_lo, z_hi), staff_model)

    cfg = load_config(str(RUN / "config.yaml"), validate=False)
    size = cfg["data"]["input_size"]
    person_label = list(cfg["model"]["heads"]["detection"]["classes"]).index("person")
    # A render worker replays recorded model outputs and never loads the network --
    # that is what makes the fan-out nearly linear on a box the GPU is busy on.
    records = None
    if args.records:
        records = json.loads(Path(args.records).read_text())
        model = device = None
    else:
        device = str(pick_device())
        model = build_model(cfg).to(device).eval()
        model.load_state_dict(select_weights(load_checkpoint(RUN / args.checkpoint), "ema"))
    frame_lo, frame_hi = 0, args.frames
    if args.chunk:
        frame_lo, frame_hi = (int(v) for v in args.chunk.split(":"))
    if args.records_only:
        w, h, _ = probe_video(str(clip))
        Path(args.records_only).write_text(
            json.dumps(
                _records_pass(
                    args,
                    clip,
                    w,
                    h,
                    size,
                    model,
                    device,
                    cf,
                    person_label,
                    (frame_lo, frame_hi),
                    staff_model,
                )
            )
        )
        return 0

    # the static scene, built once; the view frozen so the room does not swim
    scene_mesh.SS = 1
    _cf2, items, heights, shapes = scene_mesh.build_scene_regular(camera)
    # merchandise stays in: the product regions are now tiled with unit-sized items
    # sitting on the counter tops, which is what a store looks like. They read as
    # clutter only when they are region-sized slabs, which is what they used to be.
    if args.metre_scale != 1.0:
        # scaling the scene AND the eye leaves the projection identical -- this changes
        # what the numbers mean, never what the picture shows
        items = [((m[0] * args.metre_scale, m[1]), *rest) for m, *rest in items]
        heights = {k: v * args.metre_scale for k, v in heights.items()}
    xs = np.concatenate([m[0][:, 0] for m, *_ in items])
    zs = np.concatenate([m[0][:, 2] for m, *_ in items])
    cx_m, cz_m = float(np.median(xs)), float(np.median(zs))
    eye = [cx_m + 7.5, 5.6, cz_m - 6.5]
    target = [cx_m, 0.6, cz_m]
    # what the crop must contain: the furniture and every floor position a figure may
    # legally take, so a shopper at the edge of the zone is never cropped out of frame
    walk = [np.asarray(z.points_m, float) for z in cf.zones if z.kind == "walkable"]
    zone_xz = (np.vstack(walk) if walk else np.zeros((0, 2))) * args.metre_scale
    crop_meshes = [m[0] for m, *_ in items]
    if len(zone_xz):
        crop_meshes.append(np.stack([zone_xz[:, 0], np.zeros(len(zone_xz)), zone_xz[:, 1]], 1))

    panel_w, panel_h = 890, 540
    out_w, out_h = 960 + 8 + panel_w, 540
    # Every render keeps its own file and `demo_<camera>.mp4` points at the newest.
    # One fixed name meant each render destroyed the last: a version the user had
    # approved was overwritten by a worse one and there was nothing to go back to, and
    # "which file is the newest" stopped being answerable from the filesystem.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # `assets/dev/` because a render is working material, not a result. The result is the
    # figure `demo_gif.py` cuts from it, and that is the only thing `assets/` itself holds
    # -- one directory a reader opens, one the pipeline writes through, and the ignore rule
    # is then two lines rather than a growing list of patterns.
    dev = ROOT / "assets/dev"
    dev.mkdir(parents=True, exist_ok=True)
    out_path = dev / f"demo_{camera}_{stamp}.mp4"
    latest = dev / f"demo_{camera}.mp4"
    if args.chunk_out:  # a worker writes its segment and nothing else
        out_path = Path(args.chunk_out)
    part_path = out_path.with_suffix(".mp4.part")
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-framerate", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
         "-f", "mp4", str(part_path)],
        stdin=subprocess.PIPE,
    )  # fmt: skip

    # staff-memory on in BOTH the render and its replay, or a chunked render's
    # colours diverge from a single-process one. 15/0.2 measured 2026-09-03:
    # warm-up colour pops 66->46, zero contamination in 15 decidable pairs.
    tracker = Tracker(staff_memory_gap=15, staff_memory_iou=0.2)
    # frame 0 can never show a track -- the tracker confirms at 3 hits -- so the
    # frame-check samples land where there is something to check
    checks = {args.frames // 8, args.frames // 2, args.frames - 1}
    tmp = dev / f"_demo_panel_{camera}_{os.getpid()}.png"
    crop = None
    # `scene_mesh.height_caption`, not a second copy of it. A copy here once went on
    # labelling p99 numbers as "measured p85" for a whole video after the estimator became
    # per-class, because `tests/test_scene_mesh_caption.py` covers the function and could
    # not cover the copy.
    header = scene_mesh.height_caption(heights)
    # A third header line, and only when there is one. This panel crops the render's own
    # caption away, so passing `shapes` to `render` would not reach the figure -- and the
    # figure is the artefact somebody looks at.
    not_believed = scene_mesh.implausible_caption(shapes)
    seen_ids: set[int] = set()
    positions: list[dict] = []
    # The dicts that outlive a frame, owned and mutated by `figures.track_states`.
    state: dict = {
        "smoothed": {},
        "statures": {},
        "history": {},
        "last_heading": {},
        "n_skipped": 0,
    }
    bounds = (x_lo, x_hi, z_lo, z_hi)
    vel_window = max(1, round(VEL_WINDOW_S * args.fps))
    # The live dwell field: the same construction `heatmap3d.py` draws offline, kept
    # honest by sharing `syncai_bev3d.heatmap`'s maths. The grid lives in camera.json
    # metres (st.sm); tiles are scaled only at draw. Workers accumulate during REPLAY
    # too, so a chunk's field equals a single-process one at every frame.
    heat_grid = None
    if not args.no_heatmap:
        pad = 1.0
        heat_x0, heat_z0 = x_lo - pad, z_lo - pad
        heat_grid = np.zeros(
            (
                int(np.ceil((x_hi + pad - heat_x0) / HEAT_CELL_M)),
                int(np.ceil((z_hi + pad - heat_z0) / HEAT_CELL_M)),
            )
        )
        for k in range(HEAT_LEVELS):
            scene_mesh.PALETTE[f"heat_{k:02d}"] = heat_colour((k + 1) / HEAT_LEVELS)
        heat_sq = [
            (-HEAT_CELL_M / 2 * args.metre_scale, -HEAT_CELL_M / 2 * args.metre_scale),
            (HEAT_CELL_M / 2 * args.metre_scale, -HEAT_CELL_M / 2 * args.metre_scale),
            (HEAT_CELL_M / 2 * args.metre_scale, HEAT_CELL_M / 2 * args.metre_scale),
            (-HEAT_CELL_M / 2 * args.metre_scale, HEAT_CELL_M / 2 * args.metre_scale),
        ]
    n = n_det = n_fp = n_placed = n_outside = n_blur = n_posed = 0
    src_w, src_h, _ = probe_video(str(clip))
    # The plate is this camera's own empty shop, named by its camera.json. Missing is a
    # refusal rather than a silent single-instrument run: the whole argument for two
    # instruments is that neither is trusted alone.
    plate_arr = None
    if not args.no_blur:
        plate_path = ROOT / cf.plate_file if cf.plate_file else None
        if plate_path is None or not plate_path.exists():
            print(
                f"{camera}: camera.json names no readable plate_file, so the second blur "
                f"instrument cannot run. Pass --no-blur only if nothing here is shared.",
                file=sys.stderr,
            )
            return 1
        plate_arr = np.asarray(
            Image.open(plate_path).convert("RGB").resize((src_w, src_h)), np.uint8
        )
    # `model` is None exactly when `records` is not: a render worker replays instead of
    # predicting, and the branch below never dereferences the side it does not have.
    assert (model is None) == (records is not None)
    for frame in decode_frames(str(clip), src_w, src_h, args.fps):
        if n >= min(args.frames, frame_hi):
            break
        img = Image.fromarray(frame)
        if records is not None:
            # A render worker: the model already ran in the records pass; everything
            # the loop would have computed from it is read back, unrounded.
            rec = records[n]
            boxes_src = np.asarray(rec["boxes"], np.float32).reshape(-1, 4)
            kps_src = (
                np.zeros((0, 17, 3), np.float32)
                if rec["kps"] is None
                else np.asarray(rec["kps"], np.float32).reshape(-1, 17, 3)
            )
            blur_boxes = [tuple(v) for v in rec["blur"]]
            n_det += rec["n_det"]
            n_fp += rec["n_fp"]
            staff_p = None if rec["staff"] is None else np.asarray(rec["staff"], float)
            det = None
        else:
            assert model is not None  # no records means the model was loaded above
            x, _canvas, region = preprocess(img, size)
            with torch.no_grad():
                # One forward pass at the LOWER of the two thresholds: the display set is
                # filtered out of it below, so the blur set costs nothing extra.
                out = model.predict(x.to(device), score_thr=min(BLUR_THR, args.score_thr))
            det = out.get("detection", [{}])[0]
            boxes_src = np.zeros((0, 4), np.float32)
            kps_src = np.zeros((0, 17, 3), np.float32)
            blur_boxes = []
        if det and len(det.get("boxes", [])):
            b_all = det["boxes"].cpu().numpy()
            lab = det["labels"].cpu().numpy()
            sc = det["scores"].cpu().numpy()
            x0, y0, cw, _ch = region
            b_all = (b_all - np.array([x0, y0, x0, y0])) * (src_w / cw)
            person = lab == person_label
            blur_boxes = [tuple(v) for v in b_all[person & (sc >= BLUR_THR)]]
            b = b_all[person & (sc >= args.score_thr)]
            n_det += len(b)
            keep = [
                i
                for i, bb in enumerate(b)
                if not in_fp_zone(cf, (bb[0] + bb[2]) / 4, (bb[1] + bb[3]) / 4)
            ]
            n_fp += len(b) - len(keep)
            boxes_src = b[keep]
            # The pose rows are index-aligned with the detections, so the same filters
            # have to be applied to them or a figure wears another person's skeleton.
            # They are carried in source pixels to match `boxes_src`.
            pose_rows = out.get("pose", [None])[0]
            if pose_rows is not None and args.posed_figures:
                kp = pose_rows.cpu().numpy()[person & (sc >= args.score_thr)][keep].copy()
                kp[:, :, 0] = (kp[:, :, 0] - x0) * (src_w / cw)
                kp[:, :, 1] = (kp[:, :, 1] - y0) * (src_w / cw)
                kps_src = kp
        # BEFORE the blur block below, and `frame` rather than `img` for the same reason:
        # the face blur covers the torso band this reads (the argument lives beside
        # STAFF_COLOR in syncai_bev3d.figures, where it moved on 2026-09-02).
        if records is None:
            staff_p = None
            if staff_model is not None:
                staff_p = np.array(
                    [
                        staff_model.probability_of_crop(_torso_crop(frame, bb))
                        for bb in boxes_src
                    ]
                )
        tracks = [t for t in tracker.update(boxes_src, n, staff_scores=staff_p) if t.hits >= 3]
        # The sequential half -- lens-corrected foot points, position EMA, stature
        # medians, headings -- is the shared `figures.track_states`, replayable by any
        # chunked render. This file carried an inline copy until 2026-09-02, and the
        # copies had already diverged twice: stale literals for the measured constants,
        # and a fallback stature this copy forgot to scale under --metre-scale. Its
        # skip count is this file's `n_outside`.
        skips_before = state["n_skipped"]
        states = track_states(
            tracks,
            n,
            cf,
            state,
            vel_window,
            args.metre_scale,
            args.fps,
            bounds,
            verdict_of=None if staff_model is None else _display_verdict,
            source_size_px=(src_w, src_h),
        )
        n_outside += state["n_skipped"] - skips_before
        n_placed += len(states)
        if heat_grid is not None:
            for st in states:
                hgi = int((st.sm[0] - heat_x0) / HEAT_CELL_M)
                hgj = int((st.sm[1] - heat_z0) / HEAT_CELL_M)
                if 0 <= hgi < heat_grid.shape[0] and 0 <= hgj < heat_grid.shape[1]:
                    heat_grid[hgi, hgj] += 1.0 / args.fps
        if n < frame_lo:
            # A worker replaying its way to its own chunk: the tracker and the state
            # dicts advance, nothing is drawn or encoded.
            n += 1
            continue

        # Two instruments, applied to the SOURCE image before anything is drawn on it, so
        # a box outline can never sit on top of an unblurred face. The detector set is
        # taken at BLUR_THR and without the false-positive filter: an FP polygon exists to
        # stop a hanging packet becoming a track, and blurring one costs nothing.
        if not args.no_blur:
            for bb in blur_boxes:
                blur_region(img, *bb)
            if plate_arr is not None:
                for bb in plate_person_boxes(frame, plate_arr):
                    blur_region(img, *bb)
            n_blur += len(blur_boxes)

        # left: source view with boxes and ids
        view_img = img.resize((960, 540))
        d = ImageDraw.Draw(view_img)
        figures, ghosts = [], []
        if heat_grid is not None and heat_grid.any():
            hnorm, _htop = heat_normalise(heat_grid)
            specs = tile_specs(hnorm, heat_x0, heat_z0, HEAT_CELL_M, cf=cf)
            for hx, hz, hlevel, halpha in specs:
                tile = place(
                    extrude(heat_sq, 0.012),
                    Placement(hx * args.metre_scale, hz * args.metre_scale, None),
                )
                figures.append((tile, f"heat_{hlevel:02d}", halpha, False))
        moving = 0
        for t in tracks:
            seen_ids.add(t.track_id)
            bx = np.asarray(t.box, float) / 2.0
            box_col = track_colour(t, None if staff_model is None else _display_verdict)
            d.rectangle(list(bx), outline=box_col, width=2)
            d.text((bx[0] + 3, bx[1] + 2), f"#{t.track_id}", fill=box_col)

        for st in states:
            heading, speed, stature = st.heading, st.speed, st.stature
            at = Placement(st.x_m, st.z_m, heading)
            # the figure carries its track's colour, so the same shopper is the same
            # colour in both panels and the two views can be read against each other
            # The key selects the figure's colour in the 3D panel, so it has to partition
            # figures the same way the colour does. Keeping it on `track_id % 8` while the
            # colour follows a verdict lets two tracks with different verdicts share one
            # key, and the last one written wins -- a staff member silently repainted as
            # a shopper.
            key = (
                f"person_{st.track_id % len(TRACK_COLORS)}"
                if staff_model is None
                else ("person_staff" if st.verdict is True else "person_customer")
            )
            scene_mesh.PALETTE[key] = st.col
            col = st.col
            # The figure shows what the person is doing, when the pose can carry it.
            joints = None
            if args.posed_figures and len(kps_src):
                iou = box_iou(st.box, boxes_src)
                if iou.size and float(iou.max()) > 0.3:
                    joints = lift_fronto_parallel(
                        kps_src[int(iou.argmax())], st.sm[0], st.sm[1], cf
                    )
            body = None
            if joints is not None:
                # already absolute in scene metres, so it is not `place`d -- only scaled.
                # A geometry failure on one figure must not cost the other 899 frames.
                try:
                    body = human_posed(joints * args.metre_scale, stature)
                    n_posed += 1
                except ValueError as exc:
                    print(f"  frame {n}: posed figure refused ({exc}); standing instead")
            if body is None:
                body = place(human(stature), at)
            disc = place(ground_disc(0.45), at)
            figures.append((body, key, 255, True))
            figures.append((disc, key, 200, False))
            if heading is not None:
                wedge = place(extrude(facing_wedge(), 0.025), at)
                figures.append((wedge, key, 255, False))
                ghosts.append((wedge, col, 150))
            # faint on purpose: at alpha 105 a shopper standing *behind* the display
            # counter read as standing *on* it. A ghost has to look like a ghost.
            ghosts.append((body, col, 62))
            # **A plumb line from the head to the disc, because the ghost was not enough.**
            # The figure stands at y = 0; when a 2.4 m shelf is between it and the eye its
            # legs are hidden and the occlusion boundary is the shelf's top edge, so the
            # picture reads "a shopper standing on the shelf". Dropping the ghost to 62
            # fixed the x-ray layer and left that reading intact -- on 2026-08-28 a
            # reviewer found it in two of four sampled frames, and could not tell whether
            # the tracker or the reconstruction was wrong. The line is drawn over the
            # furniture with the disc, so the floor position is never inferred from where
            # the body appears to stop.
            plumb = place(box(PLUMB_W_M, stature, PLUMB_W_M), at)
            ghosts.append((plumb, col, 120))
            ghosts.append((disc, col, 150))  # the disc is the position; never hide it
            if speed >= VEL_FLOOR_MS and heading is not None:
                arrow = place(extrude(velocity_arrow(speed * VEL_SECONDS_SHOWN), 0.02), at)
                figures.append((arrow, key, 235, False))
                ghosts.append((arrow, col, 150))
                moving += 1
            positions.append(
                {
                    "frame": n,
                    "track_id": int(st.track_id),
                    "x_m": round(st.x_m, 3),
                    "z_m": round(st.z_m, 3),
                    "in_walkable": bool(point_in_walkable(cf, st.sm[0], st.sm[1])),
                    "speed_ms": round(speed, 3),
                    "stature_m": round(stature, 3),
                    "stature_frame_m": (
                        None if not np.isfinite(st.h_one) else round(st.h_one, 3)
                    ),
                    "heading_rad": None if heading is None else round(heading, 4),
                    "staff": st.verdict,
                }
            )

        view3d = scene_mesh.render(
            camera, items + figures, heights, tmp, eye=eye, target=target
        )
        pan = Image.open(tmp).convert("RGB")
        if ghosts:  # x-ray pass: an occluded shopper still reads, over the furniture
            draw_scene(ImageDraw.Draw(pan, "RGBA"), view3d, ghosts, bg=scene_mesh.BG, fog=False)
        if crop is None:
            crop = content_crop(view3d, crop_meshes, pan.size, panel_w / panel_h)
        panel = pan.crop(crop).resize((panel_w, panel_h))
        ph = ImageDraw.Draw(panel)
        ph.text(
            (10, 8),
            f"{camera}  ·  figures at tracked floor positions, each drawn at its "
            f"own measured stature; arrow = 1 s of travel, above 0.3 m/s"
            + ("" if heat_grid is None else "  ·  amber floor = cumulative dwell")
            + ("" if args.metre_scale == 1.0 else f"  ·  metres x{args.metre_scale:g}"),
            fill=(216, 224, 236),
        )
        # Two lines: the panel is 890 px and the per-class caption does not fit on one.
        # It used to fit because it said less -- a single "measured p85:" over every
        # class, which was the shorter sentence and the wrong one.
        left, _, right = header.partition("|")
        ph.text((10, 26), left.strip(), fill=(170, 182, 200))
        ph.text((10, 40), right.strip(), fill=(140, 152, 170))
        if not_believed:
            ph.text((10, 54), not_believed, fill=(236, 168, 96))

        composite = Image.new("RGB", (out_w, out_h), (7, 9, 13))
        composite.paste(view_img, (0, 0))
        composite.paste(panel, (968, 0))
        dd = ImageDraw.Draw(composite)
        dd.rectangle([0, 524, 960, 540], fill=(0, 0, 0))
        dd.text(
            (6, 526),
            f"{camera}  detections+tracks (FP zones applied)  {args.checkpoint} "
            f"thr={args.score_thr}  |  frame {n}  tracks {len(tracks)}  moving {moving}",
            fill=(255, 255, 255),
        )
        if staff_model is not None:
            # The legend carries its own accuracy, because a colour that means "staff" is
            # a claim and a viewer cannot tell a 1.00 camera from a 0.42 one by looking.
            lx = 660
            for label, colour in (("staff", STAFF_COLOR), ("customer", CUSTOMER_COLOR)):
                dd.rectangle([lx, 529, lx + 8, 537], fill=colour)
                dd.text((lx + 12, 526), label, fill=colour)
                lx += 20 + 6 * len(label)
            dd.text(
                (lx + 4, 526),
                f"torso colour, {staff_model.accuracy:.2f} held out here",
                fill=(150, 155, 165),
            )
        enc.stdin.write(np.asarray(composite, np.uint8).tobytes())
        if n in checks:
            composite.save(dev / f"demo_{camera}_{stamp}_check{n:03d}.png")
        if n % 100 == 0:
            print(f"  {n}/{args.frames}", flush=True)
        n += 1

    enc.stdin.close()
    rc = enc.wait()
    tmp.unlink(missing_ok=True)
    if rc != 0:
        print(f"ffmpeg exited {rc}; leaving {part_path}", file=sys.stderr)
        return 1
    part_path.replace(out_path)
    if args.chunk_out:
        # a worker's whole deliverable is its segment; the orchestrator owns the
        # stable name, the track log and the summary
        return 0
    # a copy, not a symlink: these get opened by players and copied to other machines,
    # and a dangling link is a worse failure than a duplicated 7 MB
    #
    # **Not under `--no-blur`, and this cost real damage on 2026-08-28.** That flag says
    # of itself "only for a private check, never for anything shared", and then wrote its
    # unblurred render straight into `demo_<camera>.mp4` -- the one filename every other
    # command, README and person treats as "the render". The stamped file stays, because
    # the private check is a legitimate use and deleting its output would just push people
    # back to a shared name; what it may not do is silently become the shared one.
    if args.no_blur:
        print(
            f"  --no-blur: {latest.name} left pointing at the last blurred render. "
            f"This file is unpublishable; it is at {out_path.name} only."
        )
    else:
        latest.write_bytes(out_path.read_bytes())
    _write_demo_tracks(args, camera, clip, staff_model, n, positions)
    print(f"wrote {out_path} ({n} frames @ {args.fps} fps)")
    if not args.no_blur:
        print(f"  newest also at {latest}")
    print(
        f"  person detections {n_det}, dropped by FP zone {n_fp}; "
        f"track placements {n_placed}, outside walkable+{PLACE_MARGIN_M:g}m {n_outside}; "
        f"{sum(p['in_walkable'] for p in positions)} of {n_placed} inside the walkable "
        f"polygon; {len(seen_ids)} distinct track ids; stature median "
        f"{np.median([p['stature_m'] for p in positions]):.2f} m; "
        f"{n_posed} of {n_placed} figures pose-driven"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
