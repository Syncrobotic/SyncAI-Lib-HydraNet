"""Every head of one network, in one frame, from one forward pass.

The point being demonstrated is not that three models agree -- it is that there is one
model. `HydraNet.predict` runs the shared trunk once and each head decodes off the neck,
so the terrain map, the boxes and the skeletons in any given frame of this video came out
of the same convolution. Nothing here runs a second network.

Four panels:

  detection   all four classes, colour-coded, with the tracker's ids
  terrain     the dense class map, `data.terrain_classes`' own palette
  pose        17 COCO keypoints per detected person, decoded inside their box
  metres      L1: the commissioned scene with a figure at each track's floor position

The fourth panel is the only one that needs commissioning, and it is the one that shows
what the other three are *for*: pixels become a person standing at a measured place.

Usage:
  uv run python tools/commissioning/heads_video.py <camera> --checkpoint PATH
      [--frames 900] [--fps 5] [--metre-scale 1.0]
"""

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import scene_mesh
from demo_video import (
    TRACK_COLORS,
    VEL_FLOOR_MS,
    VEL_SECONDS_SHOWN,
    VEL_WINDOW_S,
    content_crop,
    facing_wedge,
    in_fp_zone,
    stature_m,
    velocity_arrow,
    walkable_bounds,
)

from syncai_bev3d.meshes import Placement, extrude, ground_disc, human, place
from syncai_bev3d.shading import draw_scene
from syncai_hydranet.analytics import Tracker
from syncai_hydranet.config import load_config
from syncai_hydranet.data.video import frames as decode_frames
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import pixel_to_ground, undistort_points
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.visualize import preprocess, terrain_palette

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
PANEL = (960, 540)
# Commissioning's taxonomy, drawn in `scene_mesh`'s own colours so the mask panel and the
# 3D panel name the same thing the same way. Order is paint order: surfaces, then the
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
KP_MIN_CONF = 0.2
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
    _cf2, items, heights, _shapes = scene_mesh.build_scene_regular(camera)
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

    out_w, out_h = PANEL[0] * 2, PANEL[1] * 3
    stamp = time.strftime("%Y%m%d-%H%M%S")
    final = ROOT / f"assets/heads_{camera}_{stamp}.mp4"
    latest = ROOT / f"assets/heads_{camera}.mp4"
    part = final.with_suffix(".mp4.part")
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-framerate", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-f", "mp4", str(part)],
        stdin=subprocess.PIPE,
    )  # fmt: skip

    tracker = Tracker()
    tmp = ROOT / f"assets/_heads_{camera}_{os.getpid()}.png"
    crop = None
    history: dict[int, dict[int, tuple[float, float]]] = {}
    last_heading: dict[int, float] = {}
    smoothed: dict[int, tuple[float, float]] = {}
    statures: dict[int, list[float]] = {}
    vel_window = max(1, round(VEL_WINDOW_S * args.fps))
    n = 0
    for frame in decode_frames(str(clip), 1920, 1080, args.fps):
        if n >= args.frames:
            break
        img = Image.fromarray(frame)
        x, _canvas, region = preprocess(img, size)
        with torch.no_grad():
            res = model.predict(x.to(device), score_thr=args.score_thr)
        det = res.get("detection", [{}])[0]
        pose_rows = res.get("pose", [None])[0]
        x0, y0, cw, _ch = region
        to_panel = PANEL[0] / cw

        # --- 1. detection, every class the head has ---------------------------------
        p_det = img.resize(PANEL)
        d = ImageDraw.Draw(p_det, "RGBA")
        counts: dict[str, int] = {}
        boxes_src = np.zeros((0, 4), np.float32)
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
            pb = (b[keep] - np.array([x0, y0, x0, y0])) * (1920.0 / cw)
            boxes_src = pb[
                [i for i, bb in enumerate(pb) if not in_fp_zone(cf, bb[0] / 2, bb[1] / 2)]
            ]
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
        tracks = [t for t in tracker.update(boxes_src, n) if t.hits >= 3]
        figures, ghosts = [], []
        for t in tracks:
            col = TRACK_COLORS[t.track_id % len(TRACK_COLORS)]
            mid_x = (t.box[0] + t.box[2]) / 2 / 2.0
            pts = np.array([[mid_x, t.box[3] / 2.0], [mid_x, t.box[1] / 2.0]])
            if cf.lens is not None:
                pts = undistort_points(pts, cf.lens.k1, cf.lens.centre_px, cf.lens.radius_px)
            fx, fz = pixel_to_ground(pts[:1, 0], pts[:1, 1], cf.camera, cf.plane)
            if not (np.isfinite(fx[0]) and np.isfinite(fz[0])):
                continue
            raw = (float(fx[0]), float(fz[0]))
            prev_s = smoothed.get(t.track_id)
            sm = (
                raw
                if prev_s is None
                else (0.35 * raw[0] + 0.65 * prev_s[0], 0.35 * raw[1] + 0.65 * prev_s[1])
            )
            smoothed[t.track_id] = sm
            if not (x_lo <= sm[0] <= x_hi and z_lo <= sm[1] <= z_hi):
                continue
            h_one = stature_m(sm[0], sm[1], float(pts[1, 1]), cf)
            if 1.2 <= h_one <= 2.6:
                statures.setdefault(t.track_id, []).append(h_one)
            seen_h = statures.get(t.track_id, [])
            stature = (
                float(np.median(seen_h)) if len(seen_h) >= 3 else 1.70
            ) * args.metre_scale
            x_m, z_m = sm[0] * args.metre_scale, sm[1] * args.metre_scale
            hist = history.setdefault(t.track_id, {})
            hist[n] = (x_m, z_m)
            speed = 0.0
            prev = hist.get(n - vel_window)
            if prev is not None:
                dx, dz = x_m - prev[0], z_m - prev[1]
                speed = math.hypot(dx, dz) / (vel_window / args.fps)
                if speed >= VEL_FLOOR_MS:
                    last_heading[t.track_id] = math.atan2(dx, dz)
            heading = last_heading.get(t.track_id)
            at = Placement(x_m, z_m, heading)
            key = f"person_{t.track_id % len(TRACK_COLORS)}"
            scene_mesh.PALETTE[key] = col
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
            f"{len(tracks)} tracks  wedge = facing  arrow = 1 s of travel",
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
    latest.write_bytes(final.read_bytes())
    print(f"wrote {final} ({n} frames @ {args.fps} fps)")
    print(f"  newest also at {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
