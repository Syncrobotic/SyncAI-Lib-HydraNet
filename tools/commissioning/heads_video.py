"""Every head of one network, in one frame, from one forward pass.

The point being demonstrated is not that three models agree -- it is that there is one
model. `HydraNet.predict` runs the shared trunk once and each head decodes off the neck,
so the terrain map, the boxes and the skeletons in any given frame of this video came out
of the same convolution. Nothing here runs a second network.

Six panels, in a 2x3 grid:

  detection   all four classes, colour-coded, with the tracker's ids
  terrain     the dense class map, `data.terrain_classes`' own palette
  pose        17 COCO keypoints per detected person, decoded inside their box
  metres      L1: the commissioned scene with a figure at each track's floor position
  masks       the commissioning masks camera.json ships, on the plate they came from
  legend      both taxonomies' colour keys, side by side

The metres panel is the only one that needs commissioning, and it is the one that shows
what the network panels are *for*: pixels become a person standing at a measured place.
(This docstring said "four panels" until 2026-09-02; the masks and legend panels
arrived later and it was never updated.)

Usage:
  uv run python tools/commissioning/heads_video.py <camera> --checkpoint PATH
      [--frames 900] [--fps 5] [--metre-scale 1.0]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from syncai_bev3d import scene_mesh
from syncai_bev3d.figures import (
    KP_MIN_CONF,
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
    track_states,
    velocity_arrow,
    walkable_bounds,
)
from syncai_bev3d.meshes import (
    Placement,
    extrude,
    ground_disc,
    human,
    human_posed,
    place,
)
from syncai_bev3d.shading import draw_scene
from syncai_hydranet.analytics import Tracker
from syncai_hydranet.analytics.staff import StaffModel, require_camera, track_staff
from syncai_hydranet.config import load_config
from syncai_hydranet.data.video import frames as decode_frames
from syncai_hydranet.data.video import probe as probe_video
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.face_blur import BLUR_THR, blur_region, plate_person_boxes
from syncai_hydranet.utils.visualize import preprocess, terrain_palette

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
PANEL = (960, 540)
# Commissioning's taxonomy, drawn in `syncai_bev3d.scene_mesh`'s own colours so the mask panel
# and the 3D panel name the same thing the same way. Order is paint order: surfaces, then the
# things that sit on them.
MASK_ORDER = (
    "floor",
    "wall",
    "column",
    "display_table",
    "display_shelf",
    "door",
    "product",
    "product_boxed_stock",
    "product_macbook",
    "product_ipad",
    "product_iphone",
)
DET_COLORS = {
    "person": (120, 220, 120),
    "bag": (255, 190, 60),
    "boxed_stock": (220, 120, 255),
    "device": (80, 190, 255),
}
SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (0, 1), (0, 2), (1, 3), (2, 4),
)  # fmt: skip
# A 2 px stroke and an 11 px default-bitmap label survive a full-resolution still and do
# not survive the thing this video actually is: a 960x540 panel, h.264 at crf 22, watched
# scaled down. The first render drew every box correctly and was reported as having none.
BOX_W = 3
FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
FONT_SMALL = ImageFont.truetype("DejaVuSans.ttf", 15)


def chip(d: ImageDraw.ImageDraw, xy, text: str, rgb) -> None:
    """A filled label chip. Coloured text on the frame competes with the frame."""
    x, y = xy
    w = d.textlength(text, font=FONT_SMALL)
    # keep the chip inside the panel: a label that runs off the edge is the one thing
    # in the frame that is not a model output, so it must not look like clipping
    x = min(max(0.0, x), PANEL[0] - w - 9)
    y = max(0, y - 18)
    d.rectangle([x, y, x + w + 8, y + 18], fill=(*rgb, 235))
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    d.text(
        (x + 4, y + 1), text, fill=(0, 0, 0) if lum > 140 else (255, 255, 255), font=FONT_SMALL
    )


def commissioning_overlay(camera: str, plate: Image.Image) -> tuple[Image.Image, list[str]]:
    """The masks `camera.json` carries, on the plate they were computed from.

    This panel exists because the two taxonomies get confused for each other. The dense
    head runs every frame and knows six surface classes; these masks are computed **once
    per camera, offline, by the SAM 3 teacher** and are what `camera.json` ships. Door,
    display_table, display_shelf and the four product subclasses live only here -- and
    `product` is absent from the dense head on purpose, having been handed to detection.
    """
    base = np.asarray(plate.convert("RGB").resize(PANEL)).astype(float)
    out = base.copy()
    present: list[str] = []
    for name in MASK_ORDER:
        f = ROOT / "runs/commission01" / camera / "masks" / f"{name}.png"
        if not f.exists():
            continue
        m = np.asarray(Image.open(f).convert("L").resize(PANEL, Image.Resampling.NEAREST)) > 127
        if not m.any():
            continue
        rgb = np.array(scene_mesh.PALETTE.get(name, (200, 200, 200)), float)
        out[m] = 0.42 * out[m] + 0.58 * rgb
        present.append(name)
    return Image.fromarray(out.astype(np.uint8)), present


def legend_panel(net_names: list[str], net_palette, mask_names: list[str]) -> Image.Image:
    """Which colour means what, in both taxonomies, side by side.

    The whole point of the mask panel is the comparison, and a comparison nobody can read
    the keys of is a picture of two colourful rooms.
    """
    img = Image.new("RGB", PANEL, (14, 17, 22))
    d = ImageDraw.Draw(img)
    d.text((14, 16), "dense head, every frame", fill=(255, 255, 255), font=FONT)
    y = 46
    for i, n in enumerate(net_names):
        d.rectangle([14, y, 40, y + 18], fill=tuple(int(c) for c in net_palette[i]))
        d.text((48, y + 1), n, fill=(225, 230, 238), font=FONT_SMALL)
        y += 24
    d.text((360, 16), "commissioning masks, once per camera", fill=(255, 255, 255), font=FONT)
    y = 46
    for n in mask_names:
        d.rectangle([360, y, 386, y + 18], fill=scene_mesh.PALETTE.get(n, (200, 200, 200)))
        d.text((394, y + 1), n, fill=(225, 230, 238), font=FONT_SMALL)
        y += 24
    d.text(
        (14, PANEL[1] - 70),
        "`product` is absent from the dense head on purpose: it was moved to detection.",
        fill=(170, 180, 195),
        font=FONT_SMALL,
    )
    d.text(
        (14, PANEL[1] - 48),
        "display_table / display_shelf are one class, `fixture`, to the network.",
        fill=(170, 180, 195),
        font=FONT_SMALL,
    )
    return img


def label(img: Image.Image, text: str, sub: str = "") -> None:
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, PANEL[0], 46 if sub else 26], fill=(0, 0, 0, 190))
    d.text((10, 3), text, fill=(255, 255, 255), font=FONT)
    if sub:
        d.text((10, 24), sub, fill=(185, 195, 210), font=FONT_SMALL)


def _positions_from_boxes(boxes_all, cf, args, bounds):
    """Replay the track state over recorded boxes to get every figure's floor position.

    The sidecar has to describe the render that wrote it, and the cheapest way to be sure
    of that is to derive it from the same input and the same function the render used --
    `_track_states` over the same boxes, which is deterministic and costs microseconds a
    frame. Building it a second way would let the record and the picture disagree, which
    is the failure the sidecar exists to prevent.
    """
    tracker = Tracker()
    state: dict = {"history": {}, "last_heading": {}, "smoothed": {}, "statures": {}}
    vel_window = max(1, round(VEL_WINDOW_S * args.fps))
    out = []
    for n, rec in enumerate(boxes_all):
        b = np.asarray(rec["boxes"], np.float32).reshape(-1, 4)
        sp = None if rec.get("staff") is None else np.asarray(rec["staff"], float)
        tracks = [t for t in tracker.update(b, n, staff_scores=sp) if t.hits >= 3]
        for st in track_states(
            tracks, n, cf, state, vel_window, args.metre_scale, args.fps, bounds
        ):
            out.append(
                {
                    "frame": n,
                    "track_id": int(st.track_id),
                    "x_m": round(float(st.x_m), 4),
                    "z_m": round(float(st.z_m), 4),
                    "stature_m": round(float(st.stature), 4),
                    "speed_ms": round(float(st.speed), 4),
                    "heading_rad": None if st.heading is None else round(float(st.heading), 4),
                }
            )
    return out


def _write_sidecar(path, args, camera, clip, n_frames, positions):
    """`<render>.render.json`, the record `demo_gif` reads in preference to the per-camera log.

    **Written because not writing it made a verdict describe a different render.** `demo_gif`
    falls back to `runs/commission01/<camera>.demo_tracks.json` when a render has no sidecar,
    and that file belongs to `demo_video`. Cutting a gif from a `heads_video` render therefore
    produced an audit naming `--staff-colours` and a `--checkpoint` this tool never had, and
    read `source_clip` from it too -- so had the two tools defaulted to different clips, the
    blur audit would have run against footage the figure does not contain, and passed.

    Every argument wholesale, for the reason `demo_video`'s own log gives: an allowlist is
    broken again the day the next flag lands.
    """
    path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "tool": "heads_video.py",
                "camera": camera,
                "clip": Path(clip).name,
                "checkpoint": args.checkpoint,
                "score_thr": args.score_thr,
                "blur_score_thr": BLUR_THR,
                "blur_faces": not args.no_blur,
                "fps": args.fps,
                "frames": n_frames,
                # `heads_video` has no staff/customer model. Stated rather than omitted:
                # a missing key reads as "not recorded", and this is "not applicable".
                "staff_model": None
                if args.staff_colours is None
                else {
                    "path": args.staff_colours,
                    "min_accuracy_required": args.staff_min_accuracy,
                },
                "positions": positions,
            },
            indent=2,
        )
        + "\n"
    )


def _worker_cmd(args, camera, n_frames=None):
    """The invariant half of a worker's command line: this script, this model, this clip."""
    cmd = [sys.executable, str(Path(__file__).resolve()), camera,
           "--checkpoint", args.checkpoint, "--config", args.config,
           "--frames", str(n_frames if n_frames is not None else args.frames),
           "--fps", str(args.fps), "--score-thr", str(args.score_thr),
           "--metre-scale", str(args.metre_scale)]  # fmt: skip
    if args.clip:
        cmd += ["--clip", str(args.clip)]
    if args.no_blur:
        cmd.append("--no-blur")
    if args.standing_figures:
        cmd.append("--standing-figures")
    if args.staff_colours:
        cmd += ["--staff-colours", args.staff_colours,
                "--staff-min-accuracy", str(args.staff_min_accuracy)]  # fmt: skip
    return cmd


def _person_boxes_per_frame(
    args, clip, src_w, src_h, size, model, device, cf, person, rng=None, staff_model=None
):
    """Every frame's person boxes, in source pixels, filtered exactly as the render does.

    The pre-pass a chunked render needs. It is the model over the whole clip, which is the
    price of a tracker whose state a worker can rebuild: the boxes are the only input that
    state has, they are small enough to hand to six processes as JSON, and replaying them
    is microseconds where re-running the model would be a third of the work again.
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
            res = model.predict(x.to(device), score_thr=args.score_thr)
        det = res.get("detection", [{}])[0]
        x0, y0, cw, _ch = region
        boxes = np.zeros((0, 4), np.float32)
        if det and len(det.get("boxes", [])):
            b = det["boxes"].cpu().numpy()
            lab = det["labels"].cpu().numpy()
            pb = (b[lab == person] - np.array([x0, y0, x0, y0])) * (src_w / cw)
            # The membership point is the box CENTRE at half res: fp_polygons.py derives
            # the zones from gray-box centres, and demo_video tests the same point. This
            # file tested the top-left corner until 2026-09-02, so the two renders
            # dropped different detections through the same polygons.
            boxes = pb[
                [
                    i
                    for i, bb in enumerate(pb)
                    if not in_fp_zone(cf, (bb[0] + bb[2]) / 4, (bb[1] + bb[3]) / 4)
                ]
            ]
        # **Staff probabilities are recorded here, not left to the workers.** A track's
        # verdict is accumulated over its whole life, so a worker replaying the state has
        # to be handed the same per-box evidence a single-process run saw. Read off the
        # RAW frame: the face blur covers the torso band this classifier reads.
        sp = (
            None
            if staff_model is None
            else [
                float(staff_model.probability_of_crop(_torso_crop(frame, bb))) for bb in boxes
            ]
        )
        # **Not rounded.** These boxes are replayed into the tracker, the position EMA and
        # the stature median, and a worker that replays a rounded box arrives at frame 600
        # holding marginally different state from the single-process run -- which showed up
        # as one frame of twenty-four differing under a lossless encoder. The record has to
        # be exactly what the model produced.
        out.append({"boxes": np.asarray(boxes, np.float64).tolist(), "staff": sp})
        if len(out) % 200 == 0:
            print(f"  boxes {lo + len(out)}/{hi}", flush=True)
    return out


def _render_in_chunks(args, model, cf, bounds, clip):
    """Fan the frames out over processes, then join the segments back into one file."""
    camera = args.camera
    t0 = time.time()
    # The box pass runs the model and never touches the tracker, so it fans out the same
    # way the drawing does. Leaving it sequential was 270 s of a 535 s render.
    del model
    torch.cuda.empty_cache()
    per_b = math.ceil(args.frames / args.workers)
    b_edges = [(i, min(i + per_b, args.frames)) for i in range(0, args.frames, per_b)]
    b_paths, b_procs = [], []
    for k, (lo, hi) in enumerate(b_edges):
        bp = ROOT / f"assets/dev/_heads_{camera}_{os.getpid()}_box{k:02d}.json"
        b_paths.append(bp)
        b_procs.append(
            subprocess.Popen(
                [*_worker_cmd(args, camera), "--chunk", f"{lo}:{hi}", "--boxes-only", str(bp)]
            )
        )
    if any(pr.wait() for pr in b_procs):
        for bp in b_paths:
            bp.unlink(missing_ok=True)
        print(f"{camera}: a box pass failed; nothing written", file=sys.stderr)
        return 1
    boxes = []
    for bp in b_paths:
        boxes += json.loads(bp.read_text())
        bp.unlink(missing_ok=True)
    n_frames = len(boxes)
    box_path = ROOT / f"assets/dev/_heads_{camera}_{os.getpid()}.boxes.json"
    box_path.write_text(json.dumps(boxes))
    print(f"  pre-pass: {n_frames} frames of boxes in {time.time() - t0:.0f} s", flush=True)

    per = math.ceil(n_frames / args.workers)
    edges = [(i, min(i + per, n_frames)) for i in range(0, n_frames, per)]
    segs, procs = [], []
    for k, (lo, hi) in enumerate(edges):
        seg = ROOT / f"assets/dev/_heads_{camera}_{os.getpid()}_seg{k:02d}.mp4"
        segs.append(seg)
        procs.append(
            subprocess.Popen(
                [
                    *_worker_cmd(args, camera, n_frames),
                    "--chunk",
                    f"{lo}:{hi}",
                    "--chunk-out",
                    str(seg),
                    "--boxes",
                    str(box_path),
                ]
            )
        )
    codes = [pr.wait() for pr in procs]
    box_path.unlink(missing_ok=True)
    if any(codes):
        for seg in segs:
            seg.unlink(missing_ok=True)
        print(f"{camera}: a chunk failed with {codes}; nothing written", file=sys.stderr)
        return 1

    listing = ROOT / f"assets/dev/_heads_{camera}_{os.getpid()}.concat.txt"
    listing.write_text("".join(f"file '{s}'\n" for s in segs))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    final = ROOT / f"assets/dev/heads_{camera}_{stamp}.mp4"
    rc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(final)],
    ).returncode  # fmt: skip
    listing.unlink(missing_ok=True)
    for seg in segs:
        seg.unlink(missing_ok=True)
    if rc != 0:
        print(f"{camera}: concat failed", file=sys.stderr)
        return 1
    _write_sidecar(
        final.with_suffix(".render.json"),
        args,
        camera,
        clip,
        n_frames,
        _positions_from_boxes(boxes, cf, args, bounds),
    )
    latest = ROOT / f"assets/dev/heads_{camera}.mp4"
    latest.write_bytes(final.read_bytes())
    (latest.with_suffix(".render.json")).write_bytes(
        final.with_suffix(".render.json").read_bytes()
    )
    print(
        f"wrote {final} ({n_frames} frames @ {args.fps} fps, {args.workers} workers, "
        f"{time.time() - t0:.0f} s)"
    )
    print(f"  newest also at {latest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("camera")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/hydranet_retail_pose01.yaml")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--metre-scale", type=float, default=1.0)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="render the frames in this many processes. The tracker is sequential by "
        "nature, so a pre-pass runs the model over every frame to record the person "
        "boxes, and each worker REPLAYS the track state from those before drawing its "
        "own chunk -- replaying costs microseconds a frame and keeps every track id, "
        "colour and smoothed position identical to a single-process render.",
    )
    ap.add_argument("--chunk", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--chunk-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--boxes", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--boxes-only", default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--staff-colours",
        default=None,
        help="colour the L1 figures staff blue and customers green using this "
        "`analytics.staff` model, the same rule and the same gate `demo_video` applies. "
        "Without it the figures keep one colour per track id, which is what tells two "
        "tracks apart when nothing is classifying them.",
    )
    ap.add_argument("--staff-min-accuracy", type=float, default=None)
    ap.add_argument(
        "--standing-figures",
        action="store_true",
        help="draw the L1 figures as the standing mannequin instead of the measured pose",
    )
    # Blur is default ON, and the flag exists to be refused rather than to be
    # convenient. This tool renders a customer's shop floor across most of its panels
    # and had **no blur at all** until 2026-08-29, while writing to
    # `assets/heads_<camera>.mp4` -- a shared filename, in the directory whose whole
    # convention is that nothing enters it unaudited. `demo_video.py` has carried the
    # two instruments for days; there was never a reason for this one not to, only
    # nobody had asked. (This comment sat above --workers for a while after the flags
    # between them were inserted.)
    ap.add_argument(
        "--no-blur",
        action="store_true",
        help="do NOT blur faces -- only for a private check, never for anything shared",
    )
    args = ap.parse_args()
    camera = args.camera

    clip = (
        Path(args.clip)
        if args.clip
        else sorted((ROOT / "datasets/studioa_clips" / camera).glob("archive_*11*.mp4"))[0]
    )
    cf = CameraFile.load(ROOT / f"runs/commission01/{camera}.camera.json")
    cfg = load_config(args.config, validate=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), "ema"))
    size = cfg["data"]["input_size"]
    det_names = list(cfg["model"]["heads"]["detection"]["classes"])
    person = det_names.index("person")
    terrain_names = list(cfg["data"].get("terrain_classes") or [])
    palette = terrain_palette(terrain_names, cfg["model"]["heads"]["terrain"]["num_classes"])

    plate = Image.open(ROOT / cf.plate_file).convert("RGB")
    p_masks, mask_names = commissioning_overlay(camera, plate)
    label(
        p_masks,
        "commissioning masks - SAM 3, once per camera, offline",
        "  ".join(n.replace("product_", "") for n in mask_names),
    )
    p_legend = legend_panel(terrain_names, palette, mask_names)

    scene_mesh.SS = 1
    _cf2, items, heights, shapes = scene_mesh.build_scene_regular(camera)
    not_believed = scene_mesh.implausible_caption(shapes)
    if args.metre_scale != 1.0:
        items = [((m[0] * args.metre_scale, m[1]), *rest) for m, *rest in items]
        heights = {k: v * args.metre_scale for k, v in heights.items()}
    xs = np.concatenate([m[0][:, 0] for m, *_ in items])
    zs = np.concatenate([m[0][:, 2] for m, *_ in items])
    cx_m, cz_m = float(np.median(xs)), float(np.median(zs))
    eye = [cx_m + 7.5, 5.6, cz_m - 6.5]
    target = [cx_m, 0.6, cz_m]
    walk = [np.asarray(z.points_m, float) for z in cf.zones if z.kind == "walkable"]
    zone_xz = (np.vstack(walk) if walk else np.zeros((0, 2))) * args.metre_scale
    crop_meshes = [m[0] for m, *_ in items]
    if len(zone_xz):
        crop_meshes.append(np.stack([zone_xz[:, 0], np.zeros(len(zone_xz)), zone_xz[:, 1]], 1))
    x_lo, x_hi, z_lo, z_hi = walkable_bounds(cf)

    staff_model = None
    if args.staff_colours:
        from syncai_hydranet.analytics.staff import MIN_DEPLOY_ACCURACY

        # Written back onto the namespace for the reason `demo_video` gives: the sidecar
        # records `vars(args)`, and a floor computed into a local would record as null.
        args.staff_min_accuracy = (
            MIN_DEPLOY_ACCURACY if args.staff_min_accuracy is None else args.staff_min_accuracy
        )
        staff_model = require_camera(
            StaffModel.load(args.staff_colours), camera, min_accuracy=args.staff_min_accuracy
        )
        print(
            f"{camera}: staff colours from {args.staff_colours} -- "
            f"{staff_model.accuracy:.3f} held out on this camera "
            f"({staff_model.held_out_n} crops)"
        )

    src_w, src_h, _ = probe_video(str(clip))
    if args.boxes_only:
        lo, hi = (int(v) for v in args.chunk.split(":"))
        Path(args.boxes_only).write_text(
            json.dumps(
                _person_boxes_per_frame(
                    args,
                    clip,
                    src_w,
                    src_h,
                    size,
                    model,
                    device,
                    cf,
                    person,
                    (lo, hi),
                    staff_model,
                )
            )
        )
        return 0
    if args.workers > 1 and not args.chunk:
        return _render_in_chunks(args, model, cf, (x_lo, x_hi, z_lo, z_hi), clip)

    out_w, out_h = PANEL[0] * 2, PANEL[1] * 3
    stamp = time.strftime("%Y%m%d-%H%M%S")
    final = ROOT / f"assets/dev/heads_{camera}_{stamp}.mp4"
    latest = ROOT / f"assets/dev/heads_{camera}.mp4"
    if args.chunk_out:
        final = latest = Path(args.chunk_out)
    part = final.with_suffix(".mp4.part")
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-framerate", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-f", "mp4", str(part)],
        stdin=subprocess.PIPE,
    )  # fmt: skip

    tracker = Tracker()
    tmp = ROOT / f"assets/dev/_heads_{camera}_{os.getpid()}.png"
    crop = None
    seq_state: dict = {"history": {}, "last_heading": {}, "smoothed": {}, "statures": {}}
    boxes_log: list = []
    bounds = (x_lo, x_hi, z_lo, z_hi)
    frame_lo, frame_hi = 0, args.frames
    boxes_all: list | None = None
    if args.chunk:
        frame_lo, frame_hi = (int(v) for v in args.chunk.split(":"))
        boxes_all = json.loads(Path(args.boxes).read_text())
    vel_window = max(1, round(VEL_WINDOW_S * args.fps))
    n = 0
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
    for frame in decode_frames(str(clip), src_w, src_h, args.fps):
        if n >= frame_hi:
            break
        # Frames before this worker's chunk are REPLAYED, not rendered: the tracker, the
        # position smoothing and the stature median are sequential state, and the only
        # input they need is the person boxes the pre-pass already recorded. Replaying
        # them costs microseconds, and it is what makes a chunked render produce the same
        # track ids, colours and smoothed positions a single-process one does.
        if boxes_all is not None and n < frame_lo:
            rec = boxes_all[n]
            boxes_src = np.asarray(rec["boxes"], np.float32).reshape(-1, 4)
            sp = None if rec.get("staff") is None else np.asarray(rec["staff"], float)
            kps_src = np.zeros((0, 17, 3), np.float32)
            track_states(
                [t for t in tracker.update(boxes_src, n, staff_scores=sp) if t.hits >= 3],
                n,
                cf,
                seq_state,
                vel_window,
                args.metre_scale,
                args.fps,
                bounds,
                verdict_of=None if staff_model is None else track_staff,
            )
            n += 1
            continue
        img = Image.fromarray(frame)
        x, _canvas, region = preprocess(img, size)
        with torch.no_grad():
            res = model.predict(x.to(device), score_thr=args.score_thr)
        det = res.get("detection", [{}])[0]
        pose_rows = res.get("pose", [None])[0]
        x0, y0, cw, _ch = region
        to_panel = PANEL[0] / cw

        # **Before any panel is drawn from `img`.** Three of the four panels are the
        # source frame with something painted on it, so blurring after would leave two of
        # them showing the faces the third had covered. A second forward pass rather than
        # one at the lower threshold: the pose rows are index-aligned with the detections,
        # so widening the display set to reach the blur set would put skeletons on
        # 0.07-confidence boxes and change what the figure claims.
        if not args.no_blur:
            with torch.no_grad():
                blur_res = model.predict(x.to(device), score_thr=BLUR_THR)
            b_det = blur_res.get("detection", [{}])[0]
            if b_det and len(b_det.get("boxes", [])):
                bb_all = b_det["boxes"].cpu().numpy()
                b_lab = b_det["labels"].cpu().numpy()
                bb_all = (bb_all - np.array([x0, y0, x0, y0])) * (src_w / cw)
                for box in bb_all[b_lab == person]:
                    blur_region(img, *box)
            if plate_arr is not None:
                for box in plate_person_boxes(frame, plate_arr):
                    blur_region(img, *box)

        # --- 1. detection, every class the head has ---------------------------------
        p_det = img.resize(PANEL)
        d = ImageDraw.Draw(p_det, "RGBA")
        counts: dict[str, int] = {}
        boxes_src = np.zeros((0, 4), np.float32)
        kps_src = np.zeros((0, 17, 3), np.float32)
        if det and len(det.get("boxes", [])):
            b = det["boxes"].cpu().numpy()
            lab = det["labels"].cpu().numpy()
            for bi, li in zip(b, lab, strict=True):
                name = det_names[li]
                counts[name] = counts.get(name, 0) + 1
                bb = (bi - np.array([x0, y0, x0, y0])) * to_panel
                col = DET_COLORS.get(name, (200, 200, 200))
                d.rectangle(list(bb), outline=col, width=BOX_W)
                chip(d, (bb[0], bb[1]), name, col)
            keep = lab == person
            pb = (b[keep] - np.array([x0, y0, x0, y0])) * (src_w / cw)
            # Box CENTRE at half res, matching fp_polygons.py's derivation and
            # demo_video -- see the comment at the other call site above.
            surviving = [
                i
                for i, bb in enumerate(pb)
                if not in_fp_zone(cf, (bb[0] + bb[2]) / 4, (bb[1] + bb[3]) / 4)
            ]
            boxes_src = pb[surviving]
            # The pose rows are index-aligned with the detections, so the same two filters
            # have to be applied to them or the figure in panel 4 wears another person's
            # skeleton. They are carried in source pixels to match `boxes_src`.
            if pose_rows is not None:
                kp = pose_rows.cpu().numpy()[keep][surviving].copy()
                kp[:, :, 0] = (kp[:, :, 0] - x0) * (src_w / cw)
                kp[:, :, 1] = (kp[:, :, 1] - y0) * (src_w / cw)
                kps_src = kp
        label(
            p_det,
            "detection head - 4 classes",
            "  ".join(f"{k} {v}" for k, v in counts.items()),
        )

        # --- 2. terrain, the dense map ----------------------------------------------
        seg = res.get("terrain")
        p_seg = img.resize(PANEL)
        if seg is not None:
            # already argmaxed by the head: [B, H, W] of class ids. Crop away the
            # letterbox padding `region` describes before colouring, or the pad band
            # gets a class colour and reads as a prediction
            cls = seg[0].cpu().numpy()[y0 : y0 + _ch, x0 : x0 + cw]
            rgb = palette[np.clip(cls, 0, len(palette) - 1)].astype(np.uint8)
            p_seg = Image.blend(p_seg, Image.fromarray(rgb).resize(PANEL), 0.55)
        label(p_seg, "terrain head - dense class map", "  ".join(terrain_names))

        # --- 3. pose ----------------------------------------------------------------
        p_pose = img.resize(PANEL)
        d = ImageDraw.Draw(p_pose)
        n_people = 0
        if det and len(det.get("boxes", [])) and pose_rows is not None:
            lab = det["labels"].cpu().numpy()
            keep = lab == person
            kps = pose_rows.cpu().numpy()[keep]
            bxs = det["boxes"].cpu().numpy()[keep]
            for kp, bx in zip(kps, bxs, strict=True):
                k = kp.copy()
                k[:, 0] = (k[:, 0] - x0) * to_panel
                k[:, 1] = (k[:, 1] - y0) * to_panel
                bh = (bx[3] - bx[1]) * to_panel
                w = max(1, round(bh / 90))
                r = max(1.2, bh / 70)
                ok = k[:, 2] >= KP_MIN_CONF
                for a, bb2 in SKELETON:
                    if ok[a] and ok[bb2]:
                        d.line(
                            [tuple(k[a, :2]), tuple(k[bb2, :2])], fill=(70, 230, 255), width=w
                        )
                for j in range(17):
                    if ok[j]:
                        px, py = k[j, :2]
                        d.ellipse([px - r, py - r, px + r, py + r], fill=(255, 215, 60))
                n_people += 1
        label(p_pose, "pose head - 17 keypoints per person", f"{n_people} skeletons this frame")

        # --- 4. metres, which is what the other three are for ------------------------
        staff_p = (
            None
            if staff_model is None
            else np.array(
                [staff_model.probability_of_crop(_torso_crop(frame, bb)) for bb in boxes_src]
            )
        )
        boxes_log.append(
            {
                "boxes": np.asarray(boxes_src, np.float64).tolist(),
                "staff": None if staff_p is None else [float(v) for v in staff_p],
            }
        )
        tracks = [t for t in tracker.update(boxes_src, n, staff_scores=staff_p) if t.hits >= 3]
        states = track_states(
            tracks,
            n,
            cf,
            seq_state,
            vel_window,
            args.metre_scale,
            args.fps,
            bounds,
            verdict_of=None if staff_model is None else track_staff,
        )
        figures, ghosts = [], []
        n_posed = 0
        for st in states:
            col, sm, stature, heading, speed = (st.col, st.sm, st.stature, st.heading, st.speed)
            x_m, z_m = st.x_m, st.z_m
            t = st
            at = Placement(x_m, z_m, heading)
            # The key has to partition figures the same way `col` does -- `demo_video`
            # carries the argument: keying on `track_id % 8` while `col` follows a
            # verdict lets two tracks with different verdicts share one key, and the
            # last one written wins, a staff member silently repainted as a shopper.
            key = (
                (f"person_{'staff' if col == STAFF_COLOR else 'customer'}")
                if args.staff_colours
                else f"person_{t.track_id % len(TRACK_COLORS)}"
            )
            scene_mesh.PALETTE[key] = col
            # The figure shows what the person is doing, when the pose can carry it.
            joints = None
            if not args.standing_figures and len(kps_src):
                iou = box_iou(t.box, boxes_src)
                if iou.size and float(iou.max()) > 0.3:
                    joints = lift_fronto_parallel(kps_src[int(iou.argmax())], sm[0], sm[1], cf)
            if joints is None:
                body = place(human(stature), at)
            else:
                # already absolute in scene metres, so it is not `place`d -- only scaled.
                # A geometry failure on one figure must not cost the other 899 frames.
                try:
                    body = human_posed(joints * args.metre_scale, stature)
                    n_posed += 1
                except ValueError as exc:
                    print(f"  frame {n}: posed figure refused ({exc}); standing instead")
                    body = place(human(stature), at)
            disc = place(ground_disc(0.45), at)
            figures += [(body, key, 255, True), (disc, key, 200, False)]
            ghosts += [(body, col, 62), (disc, col, 80)]
            if heading is not None:
                wedge = place(extrude(facing_wedge(), 0.025), at)
                figures.append((wedge, key, 255, False))
                ghosts.append((wedge, col, 150))
            if speed >= VEL_FLOOR_MS and heading is not None:
                arrow = place(extrude(velocity_arrow(speed * VEL_SECONDS_SHOWN), 0.02), at)
                figures.append((arrow, key, 235, False))
                ghosts.append((arrow, col, 150))
        view3d = scene_mesh.render(
            camera, items + figures, heights, tmp, eye=eye, target=target
        )
        pan = Image.open(tmp).convert("RGB")
        if ghosts:
            draw_scene(ImageDraw.Draw(pan, "RGBA"), view3d, ghosts, bg=scene_mesh.BG, fog=False)
        if crop is None:
            crop = content_crop(view3d, crop_meshes, pan.size, PANEL[0] / PANEL[1])
        p_m = pan.crop(crop).resize(PANEL)
        label(
            p_m,
            "L1 - tracks on the commissioned floor, in metres",
            f"{len(tracks)} tracks, {n_posed} drawn in their measured pose  "
            f"wedge = facing  arrow = 1 s of travel"
            + (f"   |   {not_believed}" if not_believed else ""),
        )

        canvas = Image.new("RGB", (out_w, out_h), (7, 9, 13))
        canvas.paste(p_det, (0, 0))
        canvas.paste(p_seg, (PANEL[0], 0))
        canvas.paste(p_pose, (0, PANEL[1]))
        canvas.paste(p_m, (PANEL[0], PANEL[1]))
        canvas.paste(p_masks, (0, PANEL[1] * 2))
        canvas.paste(p_legend, (PANEL[0], PANEL[1] * 2))
        dd = ImageDraw.Draw(canvas)
        dd.rectangle([0, out_h - 28, out_w, out_h], fill=(0, 0, 0))
        dd.text(
            (10, out_h - 23),
            f"{camera}  ·  one HydraNet forward pass per frame, three heads off one trunk  ·  "
            f"{Path(args.checkpoint).name}  thr={args.score_thr}  frame {n}",
            fill=(255, 255, 255),
            font=FONT_SMALL,
        )
        enc.stdin.write(np.asarray(canvas, np.uint8).tobytes())
        if n % 100 == 0:
            print(f"  {n}/{args.frames}", flush=True)
        n += 1

    enc.stdin.close()
    rc = enc.wait()
    tmp.unlink(missing_ok=True)
    if rc != 0:
        print(f"ffmpeg exited {rc}", file=sys.stderr)
        return 1
    part.replace(final)
    # A chunk worker holds only its own frames' boxes, so its positions would describe a
    # slice while reading as a whole render. The parent writes the sidecar for the joined
    # file, from the boxes every worker was given.
    if not args.chunk:
        _write_sidecar(
            final.with_suffix(".render.json"),
            args,
            camera,
            clip,
            n,
            _positions_from_boxes(boxes_log, cf, args, bounds),
        )
    # Not under `--no-blur`: that flag says "never for anything shared" and this is the
    # filename everything else reads as this camera's render. Same fix as `demo_video`.
    if args.no_blur:
        print(
            f"  --no-blur: {latest.name} left as it was. This render is unpublishable; "
            f"it is at {final.name} only."
        )
    else:
        latest.write_bytes(final.read_bytes())
    print(f"wrote {final} ({n - frame_lo} frames @ {args.fps} fps)")
    if not args.no_blur:
        print(f"  newest also at {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
