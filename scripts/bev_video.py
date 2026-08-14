#!/usr/bin/env python3
"""Render a clip as camera view plus metric top-down map.

    python3 scripts/bev_video.py --config configs/hydranet_indoor.yaml \\
        --checkpoint runs/hydranet_indoor_det60/best.pt \\
        --input assets/clip.mp4 --output assets/clip_bev.mp4

For footage with **no depth and no calibration** -- an archived site clip, a phone video.
The depth path (`bev_scene.py`) cannot run on these, so the floor comes from geometry
instead: assume a vertical field of view, a camera height and a down-pitch, and every
walkable pixel has exactly one place it can be on that plane.

That makes the metric scale an *assumption*, and the panel says so rather than implying a
measurement. Get the height or the pitch wrong and the map is wrong by a smooth factor
that looks entirely plausible -- which is why the numbers used are printed on the frame,
and why the robot build fits the plane from depth instead of assuming it.

What survives the assumption: the *shape* of the free space, where its boundary is
relative to the robot, and whether an obstacle sits left or right. What does not: any
absolute distance, to better than the error in the assumed height.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from live_view_orin import COCO_NAMES  # noqa: E402

from syncai_hydranet.cli.infer_video import frames, probe  # noqa: E402  # isort: skip
from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.geometry.bev import IGNORE, BevGrid, scene  # noqa: E402
from syncai_hydranet.geometry.ground import Camera, GroundPlane  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights  # noqa: E402
from syncai_hydranet.utils.visualize import (  # noqa: E402
    TRAV_COLORS,
    crop_box,
    overlay,
    preprocess,
)

CELL_RGB = {0: (224, 72, 60), 1: (250, 200, 40), 2: (40, 220, 90)}
PANEL_BG = (14, 18, 24)


def free_space_map(bev: np.ndarray, grid: BevGrid, n_rays: int = 240) -> np.ndarray:
    """Keep what a camera can actually assert: free space, its boundary, nothing beyond.

    Projecting a mask onto the floor treats every pixel as if it lay on the floor. For
    walkable pixels that is true. For a wall or a shelf it is not -- the ray through a
    wall pixel meets the ground plane somewhere well past the wall -- so the naive map
    paints the whole far field `blocked`, and a map that says "obstacle" where it means
    "I cannot see" will be trusted in the wrong direction.

    What a single camera knows along each ray is: floor out to the last walkable cell,
    something standing at that boundary, and nothing at all behind it. That is what this
    returns. It is also why the far field of an honest camera map is mostly empty, which
    is the correct shape for one camera and the reason a robot carries a second sensor.
    """
    xx, zz = grid.centres()
    ang = np.arctan2(xx, zz)
    rng = np.hypot(xx, zz)
    lo, hi = float(ang.min()), float(ang.max())
    ray = np.clip(
        ((ang - lo) / max(hi - lo, 1e-9) * (n_rays - 1)).astype(np.int32), 0, n_rays - 1
    )

    walkable = (bev == 2) | (bev == 1)
    reach = np.zeros(n_rays, dtype=np.float64)  # last walkable range per ray
    np.maximum.at(reach, ray[walkable], rng[walkable])

    out = np.full_like(bev, IGNORE)
    within = rng <= reach[ray]
    out[within & walkable] = bev[within & walkable]
    # The boundary: one cell thick, where the floor stopped. Rays that never saw floor
    # get no boundary, because nothing was established about them.
    edge = (np.abs(rng - reach[ray]) <= grid.cell * 1.5) & (reach[ray] > 0) & ~walkable
    out[edge] = 0
    return out


def render_bev(bev: np.ndarray, grid: BevGrid, objects, height: int) -> Image.Image:
    """The top-down panel: floor cells, distance rules, and objects where they stand."""
    rgb = np.full((*bev.shape, 3), PANEL_BG, dtype=np.uint8)
    for value, colour in CELL_RGB.items():
        rgb[bev == value] = colour
    # Even width: libx264 with yuv420p rejects odd dimensions, and the panel's aspect
    # follows the --range window, so this is not a constant.
    width = max(int(bev.shape[1] * height / bev.shape[0]) // 2 * 2, 2)
    panel = Image.fromarray(rgb).resize((width, height), Image.NEAREST)
    draw = ImageDraw.Draw(panel)
    px_per_m = panel.height / (grid.z_max - grid.z_min)

    for z in range(int(grid.z_min) + 1, int(grid.z_max) + 1):
        y = panel.height - (z - grid.z_min) * px_per_m
        draw.line([(0, y), (panel.width, y)], fill=(70, 82, 96), width=1)
        draw.text((6, y - 13), f"{z} m", fill=(120, 136, 156))

    for obj in objects:
        x_px = (obj["x_m"] - grid.x_min) / (grid.x_max - grid.x_min) * panel.width
        y_px = panel.height - (obj["z_m"] - grid.z_min) * px_per_m
        if not (0 <= x_px < panel.width and 0 <= y_px < panel.height):
            continue
        r = 6
        draw.ellipse([x_px - r, y_px - r, x_px + r, y_px + r], outline=(235, 240, 250), width=2)
        draw.text(
            (x_px + 9, y_px - 7), f"{obj['name']} {obj['range_m']:.1f}m", fill=(235, 240, 250)
        )
    return panel


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--weights",
        choices=["ema", "model"],
        default="ema",
        help="EMA weights need enough training steps to be meaningful; see docs/TRAIN_MACOS.md",
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="bev.mp4")
    ap.add_argument("--fps", type=float, default=6.0, help="sampling and output fps")
    ap.add_argument("--max-frames", type=int, default=0, help="0 means all")
    ap.add_argument("--score-thr", type=float, default=0.15)
    # The assumptions. Printed on every frame, because a plausible-looking map built on a
    # wrong height is the failure mode this whole panel has.
    ap.add_argument("--camera-height", type=float, default=1.5, metavar="M")
    ap.add_argument(
        "--pitch", type=float, default=15.0, metavar="DEG", help="positive looks down"
    )
    ap.add_argument("--vfov", type=float, default=55.0, metavar="DEG")
    ap.add_argument("--range", type=float, default=9.0, metavar="M")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config, args.set)
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(select_weights(ckpt, args.weights))
    size = cfg["data"]["input_size"]

    src_w, src_h, src_fps = probe(args.input)
    grid = BevGrid(z_max=args.range)
    plane = GroundPlane(height=args.camera_height, pitch=np.radians(args.pitch))
    print(f"{Path(args.input).name}: {src_w}x{src_h} @ {src_fps:.1f} fps -> {args.fps} fps")
    print(
        f"assumed camera: {args.camera_height:.2f} m high, {args.pitch:.0f} deg down, "
        f"{args.vfov:.0f} deg vfov -- the metric scale is only as good as these"
    )

    writer = None
    n = 0
    for frame in frames(args.input, src_w, src_h, args.fps):
        x, canvas, region = preprocess(Image.fromarray(frame), size)
        with torch.no_grad():
            out = model.predict(x.to(device), score_thr=args.score_thr)
        x0, y0, cw, ch = region
        trav = crop_box(out["traversability"][0].cpu().numpy(), region)
        view = overlay(canvas.crop((x0, y0, x0 + cw, y0 + ch)), trav, TRAV_COLORS)

        # The mask is in letterboxed coordinates; the camera model must match it.
        cam = Camera.from_vfov(trav.shape[0], trav.shape[1], args.vfov)
        det = out.get("detection", [{}])[0]
        boxes = det["boxes"].cpu().numpy() if det and len(det.get("boxes", [])) else None
        if boxes is not None:
            boxes = boxes - np.array([x0, y0, x0, y0], dtype=np.float32)
        payload, bev = scene(
            trav,
            cam,
            plane,
            grid=grid,
            boxes=boxes,
            labels=det["labels"].cpu().numpy() if boxes is not None else None,
            scores=det["scores"].cpu().numpy() if boxes is not None else None,
            names=dict(enumerate(COCO_NAMES)),
        )
        bev = free_space_map(np.asarray(bev), grid)
        panel = render_bev(bev, grid, payload["objects"], view.height)

        out_w = (view.width + panel.width + 8) // 2 * 2
        out_h = view.height // 2 * 2
        out_img = Image.new("RGB", (out_w, out_h), PANEL_BG)
        out_img.paste(view, (0, 0))
        out_img.paste(panel, (view.width + 8, 0))
        d = ImageDraw.Draw(out_img)
        known = float((bev != IGNORE).mean())
        d.text(
            (8, view.height - 18),
            f"assumed {args.camera_height:.1f} m / {args.pitch:.0f}deg down / "
            f"{args.vfov:.0f}deg vfov - scale is an assumption, not a measurement",
            fill=(150, 165, 185),
        )
        d.text(
            (view.width + 14, 8),
            f"known: {100 * known:.0f}% of the window (the rest is behind something)",
            fill=(120, 136, 156),
        )

        if writer is None:
            writer = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{out_img.width}x{out_img.height}",
                    "-r",
                    str(args.fps),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "23",
                    args.output,
                ],
                stdin=subprocess.PIPE,
            )
        writer.stdin.write(np.asarray(out_img).tobytes())
        n += 1
        if n % 25 == 0:
            print(f"  {n} frames", flush=True)
        if args.max_frames and n >= args.max_frames:
            break

    if writer is not None:
        writer.stdin.close()
        writer.wait()
    print(f"wrote {args.output} ({n} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
