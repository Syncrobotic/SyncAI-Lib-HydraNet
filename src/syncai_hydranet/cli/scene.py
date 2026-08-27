"""Render a frame or a clip as camera view plus metric floor map, and emit the scene.

    hydranet-scene --config configs/hydranet_indoor.yaml \
        --checkpoint runs/hydranet_indoor_det60/best.pt \
        --input assets/clip.mp4 --output assets/clip_bev.mp4

    hydranet-scene --config ... --checkpoint ... --input frame.jpg --json scene.json

For footage with **no depth and no calibration** -- an archived site clip, a phone video.
The depth path (`syncai_bev3d/depth_scene.build_scene`) cannot run on these, so the floor
comes from geometry instead: assume a vertical field of view, a camera height and a
down-pitch, and every walkable pixel has exactly one place it can be on that plane.

Its two former front ends are gone: `live/render.py`, which drove it on the quadruped,
went with that line on 2026-08-19 (`eca3814`), and `scripts/bev_demo.py`, which reviewed
it without one, went in `500cdd2`. The commissioning renders in
`tools/commissioning/scene_mesh.py` are what replaced the second.

**The rule that sentence is a worked example of.** It named three things and all three had
stopped resolving, but for two different reasons, and the reasons want different
treatment. Something that **moved** gets re-aimed at where it now lives -- a reader wants
the code, and it exists. Something that was **deleted** gets `git show <commit>^:<path>`,
because there is nothing to re-aim at and a pointer that merely looks plausible is worse
than one that is obviously historical. What is never right is leaving either in the
present tense.

**And the same rule again, arriving from the other side: inline the measurement, point at
the argument.** A number cited through a pointer dies with the file it points at, and that
is not hypothetical -- the 0.847 NYUv2 scale factor was lost for two days in exactly that
way, because `data/nyu_depth.py` delegated it to a script instead of stating it. An
*argument* survives a `git show` because a reader who follows it gets the reasoning
intact; a *measurement* does not, because nobody follows a pointer to check a number they
have no reason to doubt. So a deleted file's reasoning gets a pointer and its numbers get
copied out.

That makes the metric scale an *assumption*, and the panel says so rather than implying a
measurement. Get the height or the pitch wrong and the map is wrong by a smooth factor
that looks entirely plausible -- which is why the numbers used are printed on the frame,
and why the commissioning path fits the plane from depth instead of assuming it. That
contrast is the whole reason this CLI is separate: `syncai_bev3d` gets to measure a fixed
camera once; this has to work on a clip that arrives with nothing.

What survives the assumption: the *shape* of the free space, where its boundary is
relative to the camera, and whether an obstacle sits left or right. What does not: any
absolute distance, to better than the error in the assumed height.

``--json`` writes the scene payload -- metres and class ids, no colours -- one JSON object
per line for a clip, because a scene is per-frame and an array would make a reader load the
whole clip to see the first frame of it. **Nothing in this repository reads it.** It was
the handoff format for an RViz overlay, a costmap publisher and the robot dashboard's 3D
page, and all three went with the quadruped line on 2026-08-19 (`eca3814`, `500cdd2`). The
format is kept because it is the one output of this CLI that is not a picture, and an
unused output whose consumers were *deleted* is a different thing from one whose consumers
were never written -- but a reader looking for the code that eats this will not find any.
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

from syncai_bev3d import bev3d
from syncai_bev3d.bev import IGNORE, BevGrid, free_space_map, project_mask, scene
from syncai_bev3d.scene_types import PlaneScene

from ..config import load_config
from ..data.coco_subsets import COCO_NAMES, head_order, retail_box_label
from ..data.label_maps import get_scheme, terrain_to_traversability
from ..data.label_maps_retail_security import get_det_vocab
from ..data.video import finish_encoder, frames, probe
from ..geometry.ground import Camera, GroundPlane
from ..models.hydranet import build_model
from ..utils.checkpoint import load_checkpoint, select_weights
from ..utils.device import pick_device
from ..utils.temporal import FixedCameraStabiliser
from ..utils.visualize import (
    TRAV_COLORS,
    crop_box,
    overlay,
    preprocess,
    terrain_palette,
)

PANEL_BG = (14, 18, 24)
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
# Lower than SCORE_THR_VIEW on purpose: a box that is only worth drawing is a different
# question from a box that is worth placing on a floor map, and the archived site clips
# were rendered at this threshold.
SCORE_THR_SCENE = 0.15


class SceneReport(PlaneScene):
    """A `PlaneScene` plus what this renderer measured about its own output.

    These two fields are written into the JSON `hydranet-scene` emits but are not part of
    what `syncai_bev3d.bev.scene` produces, so they belong here rather than in the geometry
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
    trav_map: dict | None = None,
    det_names: tuple[str, ...] | None = None,
    stabiliser: FixedCameraStabiliser | None = None,
) -> tuple[Image.Image, SceneReport]:
    """One frame -> the composed panel and its scene payload.

    `stabiliser` is the one argument here that carries state between calls, so it is not
    part of `Renderer.compose_kw`: a still has no history to vote over, and two clips
    rendered by one process must not inherit each other's background plate.
    """
    x, canvas, region = preprocess(frame, size, use_lb)
    with torch.no_grad():
        out = model.predict(x.to(device), score_thr=args.score_thr)
    x0, y0, cw, ch = region
    base = canvas.crop((x0, y0, x0 + cw, y0 + ch))
    if "traversability" in out:
        trav = crop_box(out["traversability"][0].cpu().numpy(), region)
    else:
        # Derived, not predicted. `trav_map` is the taxonomy's own terrain -> go/blocked
        # table, and `main` refuses the run if the config's scheme does not ship one, so
        # this is never a guess about which classes are walkable.
        trav = terrain_to_traversability(
            crop_box(out["terrain"][0].cpu().numpy(), region), trav_map
        )
    if stabiliser is not None:
        trav = stabiliser(np.asarray(base), trav)
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
        names=dict(enumerate(det_names or ())),
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
            name = (
                det_names[int(lab)]
                if det_names and int(lab) < len(det_names)
                else str(int(lab))
            )
            dv.text(
                (float(bx[0]) + 3, float(bx[1]) + 2),
                f"{name} {float(sc):.2f}",
                fill=(210, 240, 255),
            )

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
    # not part of what `syncai_bev3d.bev.scene` produces -- they are what *this* renderer
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
        help="EMA weights need enough training steps to be meaningful; see docs/DEPLOY.md",
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
        "--stabilise",
        type=int,
        default=0,
        metavar="FRAMES",
        help="clips only: fixed-camera temporal vote over N frames, applied only where "
        "the image is unchanged",
    )
    ap.add_argument(
        "--pose-note",
        default=None,
        help="replace the on-frame assumption line, e.g. when the pose came from a fit",
    )
    ap.add_argument(
        "--vocab",
        choices=["coco", "retail"],
        default="coco",
        help="how to name a box. retail reads COCO's answer as a shop noun -- "
        "fixture/oven, product/book -- which is a rename of the same head's same "
        "output, not extra knowledge. Refused on a head that is not COCO's 80",
    )
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


def detection_class_names(cfg: dict) -> tuple[str, ...] | None:
    """What this config's detection channels mean, or None if it cannot be established.

    **This function exists because the panel was labelling boxes with COCO's names
    unconditionally.** Rendered against `hydranet_retail_openvocab.yaml`, whose head has
    two classes, channel 0 was drawn as `person` and channel 1 as `bicycle`. They are
    `boxed_stock` and `device`. Nothing errored, and "person" over a shopper-height box in
    a shop is the most convincing possible wrong answer -- the label was read off a
    hardcoded table and happened to name something the scene contains.

    The guard is the **count**, because it is the one piece of evidence available:
    `COCO_NAMES` is only correct for a head that has exactly as many channels as COCO has
    names. A head with two is not a narrowed COCO head whose names still apply; it is a
    different vocabulary, and `--detection-classes` narrowing at export makes that
    ordinary rather than exotic.

    When names cannot be established the caller falls back to the **bare channel index**.
    That is deliberately worse to read and honest: `0` tells a viewer to go and look up
    what channel 0 is, and `person` tells them not to.

    Order of preference, most specific first:

    1. the head's own ``classes:`` -- declared in head order, and already checked against
       the text-embedding matrix's names when there is one;
    2. a dataset's ``det_vocab:`` -- authoritative when present, because the vocabulary
       *is* the channel order and every source maps its category names into it;
    3. a COCO dataset's ``classes:`` narrowing list, put through ``head_order`` because
       ``CocoDetDataset`` sorts by category id and a config's writing order is not that;
    4. ``COCO_NAMES`` only if the channel count agrees with it exactly.

    Rule 2 goes **above** rule 3 and the ordering is load-bearing rather than tidy.
    `hydranet_retail_security.yaml` narrows COCO to `[person, backpack, handbag,
    suitcase]` -- four names, against a four-channel head -- so rule 3 would find a count
    that agrees and return `person, backpack, handbag, suitcase`. The head's channels are
    `person, bag, boxed_stock, device`: three of the four names would be wrong, the count
    check would pass, and two of the wrong names would be plausible on a shop frame. This
    function exists to stop exactly that, and it took a new vocabulary two hours old to
    re-open the hole.
    """
    heads = (cfg.get("model") or {}).get("heads") or {}
    det = next(
        (h for h in heads.values() if isinstance(h, dict) and h.get("type") == "fcos"), None
    )
    if det is None:
        return None
    declared = det.get("classes")
    if declared:
        return tuple(str(c) for c in declared)
    for ds in (cfg.get("data") or {}).get("datasets") or []:
        if isinstance(ds, dict) and ds.get("det_vocab"):
            return get_det_vocab(str(ds["det_vocab"])).classes
    for ds in (cfg.get("data") or {}).get("datasets") or []:
        if isinstance(ds, dict) and ds.get("type") == "coco" and ds.get("classes"):
            return tuple(head_order(ds["classes"]))
    if det.get("num_classes") == len(COCO_NAMES):
        return tuple(COCO_NAMES)
    return None


def apply_vocab(names: tuple[str, ...] | None, vocab: str) -> tuple[str, ...] | None:
    """Read a COCO head's answers as shop nouns, or refuse to.

    `hydranet-infer-image` has had `--vocab retail` since it was hard-coded out of the box
    drawing; the panel this file renders never got it, so a shop's display cases came back
    as `oven 0.18` and `refrigerator 0.20` -- the boxes in the right places and only the
    vocabulary wrong. That is not extra knowledge and this is not a model change: it is the
    same head's same output, renamed by `data/coco_subsets.RETAIL_OBJECT_GROUP`.

    **It refuses on any head that is not COCO's 80.** The rename is index-addressed
    through `COCO_NAMES`, so applying it to a two-class open-vocabulary head would put
    `product/book` over a `boxed_stock` box -- the same failure `detection_class_names`
    exists to stop, wearing a shop's clothes. Refusing loudly beats renaming quietly,
    because a plausible wrong noun is the one nobody checks.

    The group keeps the COCO word beside it -- `fixture/oven` -- because the COCO word is
    the evidence for the grouping, and hiding it makes a wrong grouping unfalsifiable from
    the frame. `syncai_bev3d.meshes.detected_class` is what reads the class back out.
    """
    if vocab != "retail":
        return names
    if names is None or tuple(names) != tuple(COCO_NAMES):
        raise ValueError(
            "--vocab retail needs the COCO 80-class detection head it renames: this "
            f"config's head resolves to {len(names) if names else 0} classes. The mapping "
            "is addressed by COCO index, so applying it here would name channels after "
            "classes they do not hold."
        )
    return tuple(retail_box_label(i) for i in range(len(names)))


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
    trav_map: dict | None = None,
    vocab: str = "coco",
) -> Renderer:
    """Load the model and settle everything that does not change between frames."""
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(checkpoint)
    model.load_state_dict(select_weights(ckpt, weights))
    terrain_classes = cfg["data"].get("terrain_classes")
    det_names = detection_class_names(cfg)
    try:
        det_names = apply_vocab(det_names, vocab)
    except ValueError as exc:
        raise SystemExit(f"hydranet-scene: {exc}") from exc
    n_terrain = cfg["model"]["heads"].get("terrain", {}).get("num_classes")
    return Renderer(
        model=model,
        device=device,
        compose_kw={
            "size": cfg["data"]["input_size"],
            "use_lb": bool(cfg["data"].get("letterbox", True)),
            "palette": terrain_palette(terrain_classes, n_terrain),
            "terrain_classes": terrain_classes,
            "det_names": det_names,
            "plane": GroundPlane(height=camera_height, pitch=np.radians(pitch_deg)),
            "grid": BevGrid(z_max=z_max),
            "trav_map": trav_map,
        },
    )


def _trav_map_for(cfg: dict) -> dict | None:
    """The terrain -> traversability table of whichever scheme supervises terrain.

    Read off the config's datasets rather than hard-coded, so this follows the taxonomy
    a run actually trained on. Every terrain-supervising dataset in one config shares a
    taxonomy -- a run whose masks disagreed about what id 3 means would be broken long
    before it reached here -- so the first table found is the table.
    """
    for ds in cfg.get("data", {}).get("datasets", []) or []:
        if "terrain" not in (ds.get("supervises") or ()):
            continue
        name = ds.get("label_map")
        if not name:
            continue
        trav = getattr(get_scheme(name), "trav", None)
        if trav:
            return dict(trav)
    return None


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

    stabiliser = (
        FixedCameraStabiliser(window=args.stabilise, num_classes=len(TRAV_COLORS))
        if args.stabilise > 1
        else None
    )
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
                    stabiliser=stabiliser,
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
            code = finish_encoder(writer)

    # Only reached when the loop finished; an exception on the way here has already
    # propagated and is the more informative failure. A non-zero status is a full disk
    # or an unwritable path, and this used to print `wrote ...` for both.
    if code not in (None, 0):
        sys.exit(
            f"ffmpeg exited {code} while encoding {args.output}; "
            f"the file is incomplete or was never written."
        )
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
    trav_map = None
    if "traversability" not in (cfg["model"]["heads"] or {}):
        # The object taxonomies drop the head because it is a lookup on terrain rather
        # than a second signal. That lookup is exactly what the panel needs, so derive
        # it rather than refusing -- `RETAIL_OBJECTS_TO_TRAV` exists for this and says
        # so. Refuse only when the config's scheme ships no table, because inventing
        # which classes are walkable is the one thing that must not be guessed here.
        trav_map = _trav_map_for(cfg)
        if not trav_map:
            sys.exit(
                f"{args.config} has no traversability head and its label_map ships no "
                "terrain -> traversability table, so free space cannot be derived. The "
                "scene panel is built from free space: the floor polygon, the wall it "
                "raises at the boundary and the ground projection of every box all "
                "start there. Use a config that keeps model.heads.traversability, or "
                "hydranet-infer-video for a plain terrain overlay."
            )
        print(
            "no traversability head: free space derived from terrain via the "
            f"{len(trav_map)}-class table of this config's label_map"
        )
    renderer = build_renderer(
        cfg,
        args.checkpoint,
        args.weights,
        z_max=args.range,
        pitch_deg=args.pitch,
        camera_height=args.camera_height,
        trav_map=trav_map,
        vocab=args.vocab,
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
