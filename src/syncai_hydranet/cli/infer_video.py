"""Run inference over a video and write an overlay clip.

    hydranet-infer-video --config configs/hydranet_indoor.yaml \
        --checkpoint runs/hydranet_indoor/best.pt --input clip.mp4 \
        --output clip_pred.mp4 --fps 10

Decoding and encoding both go through the system ffmpeg, so neither opencv nor PyAV is
required. Preprocessing letterboxes to preserve aspect ratio: phone footage rarely
matches the training aspect ratio, and stretching distorts the frame badly.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..config import load_config
from ..data.coco_subsets import COCO_NAMES, retail_box_label
from ..data.label_maps_retail_security import get_det_vocab

# Re-exported: seven scripts import these from here, and the move is not
# theirs to absorb. `data/video.py` is where they live now.
from ..data.video import finish_encoder, frames, probe
from ..models.heads.detection import SCORE_THR_VIEW
from ..models.hydranet import build_model
from ..utils.checkpoint import load_checkpoint, select_weights
from ..utils.device import pick_device
from ..utils.visualize import (
    TRAV_COLORS,
    crop_box,
    depth_colors,
    overlay,
    preprocess,
    terrain_palette,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-infer-video", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="video file")
    ap.add_argument("--output", default="pred.mp4")
    ap.add_argument(
        "--weights",
        choices=["ema", "model"],
        default="ema",
        help="EMA weights need enough training steps to be meaningful; see docs/TRAIN_MACOS.md",
    )
    ap.add_argument("--fps", type=float, default=None, help="sampling and output fps")
    ap.add_argument("--score-thr", type=float, default=SCORE_THR_VIEW)
    ap.add_argument("--max-frames", type=int, default=0, help="0 means all")
    ap.add_argument("--layout", choices=["side", "trav", "terrain", "quad"], default="side")
    ap.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="metres at the far end of the depth colour ramp (--layout quad). Defaults to "
        "the head's own `max_depth`, which is the range it was trained to emit -- but an "
        "indoor corridor spends that whole range in its first fifth, so a smaller value "
        "reads better. Fixed either way: never per-frame, see utils.visualize.depth_colors",
    )
    ap.add_argument(
        "--nms-thr",
        type=float,
        default=0.6,
        help="IoU above which two boxes of one class are treated as one detection. "
        "Raise it for crowded scenes, where shoppers overlap and the tighter default "
        "merges two people into one. Note the audit's evidence for missed people points "
        "at the SCORE threshold, not at this: on Taichung-cam01 person detections went "
        "62 -> 72 -> 112 as the score cut fell 0.35 -> 0.25 -> 0.15. This knob is "
        "exposed because it was hard-coded, not because it was measured",
    )
    ap.add_argument(
        "--vocab",
        choices=["coco", "retail"],
        default="coco",
        help="how to name a box. retail reads COCO's answer as a shop noun -- "
        "product/book, fixture/refrigerator -- which is a rename of the same head's "
        "same output, not extra knowledge; see data/coco_subsets.py",
    )
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


def crop_box_f(arr, region):
    """`crop_box` for a float map. Kept separate rather than generalised: the class-index
    version is used by every existing layout and its dtype is load-bearing there."""
    x0, y0, cw, ch = region
    return arr[y0 : y0 + ch, x0 : x0 + cw]


def _labelled(img: Image.Image, text: str) -> Image.Image:
    """A copy of `img` with `text` in a dark strip. A copy, because the panels in a quad
    are the same objects the other layouts return, and captioning them in place would put
    a stale label on whatever reused them."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width, 18], fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))
    return out


def render_frame(
    frame, model, device, args, *, size, terrain_colors, name_of, model_max_depth=10.0
) -> Image.Image:
    """One decoded frame -> the panel to encode.

    Split out of `main`'s loop because everything interesting happens here: the model
    call, both overlays, the boxes and the `--layout` choice. Left inline it was reachable
    only by running ffmpeg against a real video, so the layout rules -- including the
    no-traversability-head branch below -- had no way to be tested at all.
    """
    x, lb, region = preprocess(Image.fromarray(frame), size)
    x = x.to(device)
    with torch.no_grad():
        res = model.predict(x, score_thr=args.score_thr, nms_thr=args.nms_thr)

    x0, y0, cw, ch = region
    terr = crop_box(res["terrain"][0].cpu().numpy(), region)
    content = lb.crop((x0, y0, x0 + cw, y0 + ch))

    vis_terr = overlay(content, terr, terrain_colors)
    # See infer_image: traversability is optional, because it is a lookup on terrain
    # and the object-segmentation configs drop the head entirely.
    vis_trav = (
        overlay(content, crop_box(res["traversability"][0].cpu().numpy(), region), TRAV_COLORS)
        if "traversability" in res
        else None
    )
    box_panel = vis_trav if vis_trav is not None else vis_terr

    det = res.get("detection", [{}])[0]
    if det and len(det.get("boxes", [])):
        draw = ImageDraw.Draw(box_panel)
        for box, score, label in zip(
            det["boxes"].cpu().numpy(),
            det["scores"].cpu().numpy(),
            det["labels"].cpu().numpy(),
            strict=True,
        ):
            b = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
            if b[2] <= 0 or b[3] <= 0 or b[0] >= cw or b[1] >= ch:
                continue
            draw.rectangle(b, outline=(255, 255, 255), width=2)
            draw.text(
                (b[0] + 2, b[1] + 2),
                f"{name_of(int(label))} {score:.2f}",
                fill=(255, 255, 255),
            )

    if args.layout == "quad":
        if "depth" not in res:
            sys.exit(
                "--layout quad, but this config has no depth head (model.heads.depth). "
                "Use --layout side."
            )
        if vis_trav is None:
            sys.exit("--layout quad, but this config has no traversability head.")
        vmax = args.depth_max or float(model_max_depth)
        dep = crop_box_f(res["depth"][0].cpu().numpy(), region)
        vis_depth = Image.fromarray(depth_colors(dep, vmax))
        panels = [
            (_labelled(box_panel, "traversability + detection"), (0, 0)),
            (_labelled(vis_terr, "terrain"), (cw, 0)),
            (_labelled(content, "camera"), (0, ch)),
            (_labelled(vis_depth, f"depth  0-{vmax:g} m  (near red, far blue)"), (cw, ch)),
        ]
        out_img = Image.new("RGB", (cw * 2, ch * 2))
        for panel, at in panels:
            out_img.paste(panel, at)
        return out_img

    if vis_trav is None:
        # Nothing to put in the left panel and nothing to select with --layout.
        # Checked here rather than at argument-parse time because whether the head
        # exists is a property of the checkpoint's config, not of the command line.
        if args.layout == "trav":
            sys.exit(
                "--layout trav, but this config has no traversability head "
                "(model.heads.traversability). Use --layout terrain."
            )
        return vis_terr
    if args.layout == "trav":
        return vis_trav
    if args.layout == "terrain":
        return vis_terr
    out_img = Image.new("RGB", (cw * 2, ch))
    out_img.paste(vis_trav, (0, 0))
    out_img.paste(vis_terr, (cw, 0))
    return out_img


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found. On macOS: brew install ffmpeg")

    cfg = load_config(args.config, args.set)
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(select_weights(ckpt, args.weights))

    size = cfg["data"]["input_size"]  # (H, W)
    depth_cfg = (cfg.get("model", {}).get("heads", {}) or {}).get("depth") or {}
    model_max_depth = float(depth_cfg.get("max_depth", 10.0))
    terrain_colors = terrain_palette(cfg["data"].get("terrain_classes"))
    # See infer_image: the class id is the least useful of the three things known about
    # a detection, and scene.py already drew names rather than integers.
    name_of = (
        retail_box_label
        if args.vocab == "retail"
        else (lambda i: COCO_NAMES[i] if i < len(COCO_NAMES) else str(i))
    )
    # Both --vocab options assume an 80-class COCO head. A checkpoint trained against a
    # shared `det_vocab` numbers its classes from that vocabulary instead, and naming its
    # id 2 "car" (COCO) or "product" (retail rename) is wrong twice over -- measured on
    # the first b03 demo render, where every `boxed_stock` drew as `car`. The config
    # already says which vocabulary froze the ids, so it outranks the flag.
    vocabs = {d["det_vocab"] for d in cfg["data"].get("datasets", []) if d.get("det_vocab")}
    if len(vocabs) == 1:
        vocab_classes = get_det_vocab(next(iter(vocabs))).classes
        name_of = lambda i: vocab_classes[i] if i < len(vocab_classes) else str(i)  # noqa: E731
        print(f"det_vocab {next(iter(vocabs))!r} names the boxes; --vocab ignored")
    src_w, src_h, src_fps = probe(args.input)
    out_fps = args.fps or src_fps
    print(
        f"input {src_w}x{src_h} @ {src_fps:.1f}fps -> model input {size[1]}x{size[0]} "
        f"(letterbox) device={device}"
    )

    writer = None
    enc_size = None
    n_done, t0 = 0, time.time()
    sample_fps = args.fps if args.fps and args.fps < src_fps else None
    # The encoder starts inside the loop, so its shutdown has to survive whatever
    # ends the loop. Without the `finally`, an exception mid-render left ffmpeg with
    # no EOF on its stdin: an unplayable mp4 with no moov atom, and an orphan.
    try:
        for frame in frames(args.input, src_w, src_h, sample_fps):
            out_img = render_frame(
                frame,
                model,
                device,
                args,
                size=size,
                terrain_colors=terrain_colors,
                name_of=name_of,
                model_max_depth=model_max_depth,
            )
            if writer is None:
                ow, oh = out_img.size
                ow, oh = ow - ow % 2, oh - oh % 2  # H.264 requires even dimensions
                enc_size = (ow, oh)
                writer = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{ow}x{oh}",
                        "-r",
                        f"{out_fps}",
                        "-i",
                        "-",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-crf",
                        "20",
                        args.output,
                    ],
                    stdin=subprocess.PIPE,
                )
            # `writer` and `enc_size` are set together on the first frame and never again,
            # and `Popen.stdin` is Optional only because `stdin=PIPE` is optional in
            # general. One place to state all of that, rather than three re-derivations
            # spread over the next four lines.
            sink = writer.stdin
            assert sink is not None and enc_size is not None
            if out_img.size != enc_size:
                out_img = out_img.resize(enc_size, Image.Resampling.BILINEAR)
            sink.write(np.asarray(out_img, np.uint8).tobytes())

            n_done += 1
            if n_done % 30 == 0:
                print(f"  {n_done} frames ({n_done / (time.time() - t0):.1f} fps)")
            if args.max_frames and n_done >= args.max_frames:
                break

    finally:
        code = finish_encoder(writer)

    # Only reached when the loop finished: an exception on the way here has already
    # propagated, and it is the more informative failure. A non-zero status here is
    # a full disk or an unwritable path, and the old code printed `done:` for both.
    if code not in (None, 0):
        sys.exit(
            f"ffmpeg exited {code} while encoding {args.output}; "
            f"the file is incomplete or was never written."
        )
    dt = time.time() - t0
    print(
        f"done: {n_done} frames in {dt:.1f}s "
        f"({n_done / max(dt, 1e-6):.1f} fps) -> {args.output}"
    )


if __name__ == "__main__":
    main()
