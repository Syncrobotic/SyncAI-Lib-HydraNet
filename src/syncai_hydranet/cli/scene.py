"""Render a frame or a clip as camera view plus metric floor map, and emit the scene.

    hydranet-scene --config configs/hydranet_indoor.yaml \
        --checkpoint runs/hydranet_indoor_det60/best.pt \
        --input assets/clip.mp4 --output assets/clip_bev.mp4

    hydranet-scene --config ... --checkpoint ... --input frame.jpg --json scene.json

For footage with **no depth and no calibration** -- an archived site clip, a phone video.
The depth path (`scripts/bev_scene.py`) cannot run on these, so the floor comes from
geometry instead: assume a vertical field of view, a camera height and a down-pitch, and
every walkable pixel has exactly one place it can be on that plane.

That makes the metric scale an *assumption*, and the panel says so rather than implying a
measurement. Get the height or the pitch wrong and the map is wrong by a smooth factor
that looks entirely plausible -- which is why the numbers used are printed on the frame,
and why the robot build fits the plane from depth instead of assuming it.

What survives the assumption: the *shape* of the free space, where its boundary is
relative to the robot, and whether an obstacle sits left or right. What does not: any
absolute distance, to better than the error in the assumed height.

``--json`` writes the scene payload -- metres and class ids, no colours -- which is the
handoff format for anything that renders: an RViz overlay, a costmap publisher, the 3D
page. For a clip it is one JSON object per line, because a scene is per-frame and an
array would make a reader load the whole clip to see the first frame of it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..config import load_config
from ..data.coco_subsets import COCO_NAMES
from ..geometry import bev3d
from ..geometry.bev import IGNORE, BevGrid, free_space_map, project_mask, scene
from ..geometry.ground import Camera, GroundPlane
from ..geometry.scene_types import PlaneScene
from ..models.hydranet import build_model
from ..utils.checkpoint import load_checkpoint, select_weights
from ..utils.device import pick_device
from ..utils.visualize import (
    TRAV_COLORS,
    crop_box,
    overlay,
    preprocess,
    terrain_palette,
)
from .infer_video import frames, probe

CELL_RGB = {0: (224, 72, 60), 1: (250, 200, 40), 2: (40, 220, 90)}
PANEL_BG = (14, 18, 24)
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
# Lower than SCORE_THR_VIEW on purpose: a box that is only worth drawing is a different
# question from a box that is worth placing on a floor map, and the archived site clips
# were rendered at this threshold.
SCORE_THR_SCENE = 0.15


def render_bev(bev: np.ndarray, grid: BevGrid, objects, height: int) -> Image.Image:
    """The top-down panel: floor cells, distance rules, and objects where they stand."""
    rgb = np.full((*bev.shape, 3), PANEL_BG, dtype=np.uint8)
    for value, colour in CELL_RGB.items():
        rgb[bev == value] = colour
    # Even width: libx264 with yuv420p rejects odd dimensions, and the panel's aspect
    # follows the --range window, so this is not a constant.
    width = max(int(bev.shape[1] * height / bev.shape[0]) // 2 * 2, 2)
    panel = Image.fromarray(rgb).resize((width, height), Image.Resampling.NEAREST)
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


class SceneReport(PlaneScene):
    """A `PlaneScene` plus what this renderer measured about its own output.

    These two fields are written into the JSON `hydranet-scene` emits but are not part of
    what `geometry.bev.scene` produces, so they belong here rather than in the geometry
    layer's type. The distinction is not academic: a consumer reading the geometry
    payload directly will not have them, and before this class the only way to learn that
    was to notice two subscript assignments at the end of `compose`.
    """

    known_fraction: float
    pose_is_assumed: bool


class SceneRecord(SceneReport):
    """One line of the `--json` JSONL: a `SceneReport` stamped with its frame index.

    A third shape, because it genuinely is one. `compose` cannot supply `frame` -- it
    sees a single image and not its position in a stream -- so the field is added at the
    write site and only when `--json` was asked for. Three payload shapes had been
    flowing through this file, all of them annotated `dict`; this is the last of them.
    """

    frame: int


def compose(
    frame: Image.Image,
    model,
    device,
    args,
    *,
    size,
    use_lb: bool,
    palette,
    terrain_classes,
    plane: GroundPlane,
    grid: BevGrid,
) -> tuple[Image.Image, SceneReport]:
    """One frame -> the composed panel and its scene payload."""
    x, canvas, region = preprocess(frame, size, use_lb)
    with torch.no_grad():
        out = model.predict(x.to(device), score_thr=args.score_thr)
    x0, y0, cw, ch = region
    base = canvas.crop((x0, y0, x0 + cw, y0 + ch))
    trav = crop_box(out["traversability"][0].cpu().numpy(), region)
    view = overlay(base, trav, TRAV_COLORS)
    terrain = None
    if "terrain" in out:
        terrain = crop_box(out["terrain"][0].cpu().numpy(), region)
        terrain_view = overlay(base, terrain, palette)
    else:
        terrain_view = base.copy()

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

    # Detections belong on the camera view too: three heads, three things to see.
    dv = ImageDraw.Draw(view)
    if boxes is not None:
        for bx, lab, sc in zip(
            boxes, det["labels"].cpu().numpy(), det["scores"].cpu().numpy(), strict=True
        ):
            dv.rectangle(
                [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])],
                outline=(90, 200, 255),
                width=2,
            )
            name = COCO_NAMES[int(lab)] if int(lab) < len(COCO_NAMES) else str(int(lab))
            dv.text(
                (float(bx[0]) + 3, float(bx[1]) + 2),
                f"{name} {float(sc):.2f}",
                fill=(210, 240, 255),
            )

    if args.flat_bev:
        panel = render_bev(bev, grid, payload["objects"], view.height)
        out_w = (view.width + panel.width + 8) // 2 * 2
        out_h = view.height // 2 * 2
        out_img = Image.new("RGB", (out_w, out_h), PANEL_BG)
        out_img.paste(view, (0, 0))
        out_img.paste(panel, (view.width + 8, 0))
    else:
        terrain_bev = project_mask(terrain, cam, plane, grid) if terrain is not None else None
        col_h = view.height * 2 + 8
        pw = max(int(col_h * 0.78) // 2 * 2, 2)
        panel = bev3d.render(
            bev,
            terrain_bev,
            grid,
            payload["objects"],
            (pw, col_h),
            trav_colors=TRAV_COLORS,
            terrain_colors=palette,
            bg=PANEL_BG,
            class_names=terrain_classes,
        )
        out_w = (view.width + pw + 8) // 2 * 2
        out_h = col_h // 2 * 2
        out_img = Image.new("RGB", (out_w, out_h), PANEL_BG)
        out_img.paste(view, (0, 0))
        out_img.paste(terrain_view, (0, view.height + 8))
        out_img.paste(panel, (view.width + 8, 0))

    d = ImageDraw.Draw(out_img)
    if not args.flat_bev:
        # A strip behind the caption: the source clips have their own burnt-in camera
        # name in the same corner, and two texts on top of each other are unreadable in
        # exactly the frames someone screenshots.
        for label, top in (("traversability + detections", 0), ("terrain", view.height + 8)):
            d.rectangle([0, top, 250, top + 18], fill=(0, 0, 0))
            d.text((6, top + 3), label, fill=(205, 220, 240))
    known = float((bev != IGNORE).mean())
    note = args.pose_note or (
        f"assumed {args.camera_height:.1f} m / {args.pitch:.0f}deg down / "
        f"{args.vfov:.0f}deg vfov - scale is an assumption, not a measurement"
    )
    d.rectangle(
        [0, view.height - 20, min(len(note) * 6 + 10, view.width), view.height], fill=(0, 0, 0)
    )
    d.text((6, view.height - 17), note, fill=(165, 180, 200))
    d.text(
        (view.width + 14, 8),
        f"known: {100 * known:.0f}% of the window (the rest is behind something)",
        fill=(120, 136, 156),
    )
    # Built as one dict rather than assigned onto `payload`, because these two fields are
    # not part of what `geometry.bev.scene` produces -- they are what *this* renderer
    # measured about its own output. `SceneReport` is where that difference is written
    # down; before it, the JSON this CLI writes had a shape no type described and the
    # only way to learn about these keys was to read this function.
    return out_img, {
        **payload,
        "known_fraction": round(known, 4),
        "pose_is_assumed": args.pose_note is None,
    }


def encoder_argv(width: int, height: int, args) -> list[str]:
    """The ffmpeg command for a raw RGB pipe. Split out so the loop stays readable."""
    # fmt: off
    return [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(args.fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        args.output,
    ]
    # fmt: on


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-scene", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--weights",
        choices=["ema", "model"],
        default="ema",
        help="EMA weights need enough training steps to be meaningful; see docs/TRAIN_MACOS.md",
    )
    ap.add_argument("--input", required=True, help="image or video file")
    ap.add_argument("--output", default=None, help="rendered panel; omit to write none")
    ap.add_argument("--json", default=None, help="scene payload; JSON lines for a clip")
    ap.add_argument("--fps", type=float, default=6.0, help="sampling and output fps")
    ap.add_argument("--max-frames", type=int, default=0, help="0 means all")
    ap.add_argument("--score-thr", type=float, default=SCORE_THR_SCENE)
    # The assumptions. Printed on every frame, because a plausible-looking map built on a
    # wrong height is the failure mode this whole panel has.
    ap.add_argument("--camera-height", type=float, default=1.5, metavar="M")
    ap.add_argument(
        "--pitch", type=float, default=15.0, metavar="DEG", help="positive looks down"
    )
    ap.add_argument("--vfov", type=float, default=55.0, metavar="DEG")
    ap.add_argument("--range", type=float, default=9.0, metavar="M")
    ap.add_argument(
        "--flat-bev",
        action="store_true",
        help="the original top-down panel instead of the perspective one",
    )
    ap.add_argument(
        "--pose-note",
        default=None,
        help="replace the on-frame assumption line, e.g. when the pose came from a fit",
    )
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


@dataclass(frozen=True)
class Renderer:
    """The model and the per-run settings every frame needs, as one value.

    These are fixed for a whole run and `compose` needs all of them for every frame.
    Passing them individually is what kept both frame loops inside `main` -- there were
    eight names to thread, so it was easier to leave the loop where they were already in
    scope. Bundled, the loops can move out and be called from something that is not
    argparse.
    """

    model: torch.nn.Module
    device: torch.device | str
    compose_kw: dict


def build_renderer(
    cfg: dict,
    checkpoint: str,
    weights: str,
    *,
    z_max: float,
    pitch_deg: float,
    camera_height: float,
) -> Renderer:
    """Load the model and settle everything that does not change between frames."""
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(checkpoint)
    model.load_state_dict(select_weights(ckpt, weights))
    terrain_classes = cfg["data"].get("terrain_classes")
    n_terrain = cfg["model"]["heads"].get("terrain", {}).get("num_classes")
    return Renderer(
        model=model,
        device=device,
        compose_kw={
            "size": cfg["data"]["input_size"],
            "use_lb": bool(cfg["data"].get("letterbox", True)),
            "palette": terrain_palette(terrain_classes, n_terrain),
            "terrain_classes": terrain_classes,
            "plane": GroundPlane(height=camera_height, pitch=np.radians(pitch_deg)),
            "grid": BevGrid(z_max=z_max),
        },
    )


def render_still(in_path: Path, renderer: Renderer, args) -> int:
    """One image in, one panel and/or one JSON document out."""
    out_img, payload = compose(
        Image.open(in_path), renderer.model, renderer.device, args, **renderer.compose_kw
    )
    if args.output:
        out_img.save(args.output)
        print(f"wrote {args.output}")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json} ({len(payload['objects'])} objects)")
    return 0


def render_video(in_path: Path, renderer: Renderer, args) -> int:
    """Every sampled frame in, an encoded video and/or a JSONL stream out."""
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found. On macOS: brew install ffmpeg")
    src_w, src_h, src_fps = probe(args.input)
    print(f"{in_path.name}: {src_w}x{src_h} @ {src_fps:.1f} fps -> {args.fps} fps")

    writer, n = None, 0
    with contextlib.ExitStack() as stack:
        jsonl = stack.enter_context(Path(args.json).open("w")) if args.json else None
        try:
            for frame in frames(args.input, src_w, src_h, args.fps):
                out_img, payload = compose(
                    Image.fromarray(frame),
                    renderer.model,
                    renderer.device,
                    args,
                    **renderer.compose_kw,
                )
                if jsonl is not None:
                    record: SceneRecord = {**payload, "frame": n}
                    jsonl.write(json.dumps(record) + "\n")
                if args.output:
                    if writer is None:
                        writer = subprocess.Popen(
                            encoder_argv(out_img.width, out_img.height, args),
                            stdin=subprocess.PIPE,
                        )
                    # stdin=PIPE was requested just above, so the pipe exists; Popen
                    # types it Optional because that argument is optional in general.
                    sink = writer.stdin
                    assert sink is not None
                    sink.write(np.asarray(out_img).tobytes())
                n += 1
                if n % 25 == 0:
                    print(f"  {n} frames", flush=True)
                if args.max_frames and n >= args.max_frames:
                    break
        finally:
            if writer is not None:
                assert writer.stdin is not None
                writer.stdin.close()
                writer.wait()
    print(f"wrote {args.output or args.json} ({n} frames)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is None and args.json is None:
        sys.exit("nothing to write: pass --output, --json, or both")

    cfg = load_config(args.config, args.set)
    # Every panel this command draws starts from free space, and free space is the
    # traversability head. Refuse here, naming the config key, rather than raising
    # KeyError on the first frame after the model and the video are both loaded --
    # configs/hydranet_retail_objects.yaml drops that head deliberately.
    if "traversability" not in (cfg["model"]["heads"] or {}):
        sys.exit(
            f"{args.config} has no traversability head, and the scene panel is built "
            "from free space: the floor polygon, the wall it raises at the boundary and "
            "the ground projection of every box all start there. Use a config that "
            "keeps model.heads.traversability, or hydranet-infer-video for a plain "
            "terrain overlay."
        )
    renderer = build_renderer(
        cfg,
        args.checkpoint,
        args.weights,
        z_max=args.range,
        pitch_deg=args.pitch,
        camera_height=args.camera_height,
    )
    print(
        f"assumed camera: {args.camera_height:.2f} m high, {args.pitch:.0f} deg down, "
        f"{args.vfov:.0f} deg vfov -- the metric scale is only as good as these"
    )
    in_path = Path(args.input)
    if in_path.suffix.lower() in VIDEO_EXTS:
        return render_video(in_path, renderer, args)
    return render_still(in_path, renderer, args)


if __name__ == "__main__":
    raise SystemExit(main())
