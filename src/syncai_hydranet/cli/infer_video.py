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
import json
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..config import load_config
from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD
from ..models.heads.detection import SCORE_THR_VIEW
from ..models.hydranet import build_model
from ..utils.checkpoint import load_checkpoint
from ..utils.device import pick_device
from ..utils.visualize import TERRAIN_COLORS, TRAV_COLORS, crop_box, letterbox, overlay


def probe(path: str) -> tuple[int, int, float]:
    """Return display width, height and fps, accounting for rotation metadata."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:stream_side_data=rotation",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    st = json.loads(out)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    num, _, den = st.get("r_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1)
    rot = 0
    for sd in st.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) % 180 == 90:  # ffmpeg autorotates on decode, swapping the axes
        w, h = h, w
    return w, h, fps


def frames(path: str, w: int, h: int, stride_fps: float | None):
    """Yield RGB frames from a rawvideo pipe."""
    vf = f"fps={stride_fps}" if stride_fps else "null"
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    n = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-infer-video", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="video file")
    ap.add_argument("--output", default="pred.mp4")
    ap.add_argument("--fps", type=float, default=None, help="sampling and output fps")
    ap.add_argument("--score-thr", type=float, default=SCORE_THR_VIEW)
    ap.add_argument("--max-frames", type=int, default=0, help="0 means all")
    ap.add_argument("--layout", choices=["side", "trav", "terrain"], default="side")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found. On macOS: brew install ffmpeg")

    cfg = load_config(args.config, args.set)
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"])

    size = cfg["data"]["input_size"]  # (H, W)
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
    for frame in frames(args.input, src_w, src_h, sample_fps):
        img = Image.fromarray(frame)
        lb, region = letterbox(img, size)

        arr = (np.asarray(lb, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(arr.transpose(2, 0, 1))[None].to(device)
        with torch.no_grad():
            res = model.predict(x, score_thr=args.score_thr)

        x0, y0, cw, ch = region
        trav = crop_box(res["traversability"][0].cpu().numpy(), region)
        terr = crop_box(res["terrain"][0].cpu().numpy(), region)
        content = lb.crop((x0, y0, x0 + cw, y0 + ch))

        vis_trav = overlay(content, trav, TRAV_COLORS)
        vis_terr = overlay(content, terr, TERRAIN_COLORS)

        det = res.get("detection", [{}])[0]
        if det and len(det.get("boxes", [])):
            draw = ImageDraw.Draw(vis_trav)
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
                    (b[0] + 2, b[1] + 2), f"{int(label)}:{score:.2f}", fill=(255, 255, 255)
                )

        if args.layout == "trav":
            out_img = vis_trav
        elif args.layout == "terrain":
            out_img = vis_terr
        else:
            out_img = Image.new("RGB", (cw * 2, ch))
            out_img.paste(vis_trav, (0, 0))
            out_img.paste(vis_terr, (cw, 0))

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
        if out_img.size != enc_size:
            out_img = out_img.resize(enc_size, Image.BILINEAR)
        writer.stdin.write(np.asarray(out_img, np.uint8).tobytes())

        n_done += 1
        if n_done % 30 == 0:
            print(f"  {n_done} frames ({n_done / (time.time() - t0):.1f} fps)")
        if args.max_frames and n_done >= args.max_frames:
            break

    if writer:
        writer.stdin.close()
        writer.wait()
    dt = time.time() - t0
    print(
        f"done: {n_done} frames in {dt:.1f}s "
        f"({n_done / max(dt, 1e-6):.1f} fps) -> {args.output}"
    )


if __name__ == "__main__":
    main()
