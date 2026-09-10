#!/usr/bin/env python3
"""Cut a strip of frames around each mid-view track death an endings run recorded.

    python3 scripts/death_strips.py runs/endings08 --verdict available --out scratch/strips

`scripts/track_endings.py` records, since 2026-09-10, where every mid-view death happened
(the track's last observed frame and box) and, for an `available` verdict, the box the
tracker could have taken and the frame it was on. This turns those numbers into
something a person can look at: the last frame with the dead track's box in red, the
witness frame with the box the tracker did not take in green, side by side, one strip per
death, one contact sheet per camera. The question each strip answers is *why* the
tracker did not take that box -- a neighbour took it, the prediction drifted, the box
jumped -- which is the question the four-arm measurement could count but not answer.

Boxes are in the undistorted source pixels the tracker used; the frame is raw, so the
corners go back through the lens before drawing. Nothing here is blurred: these strips
are a working instrument for a reviewer inside the project and are written outside the
tree by default. They are not publishable and this script does not make them so.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import distort_points

ROOT = Path(__file__).resolve().parent.parent
COMMISSIONED = ROOT / "runs/commission01"
_SWEEP = json.loads((ROOT / "configs/sweep_clips.json").read_text())
CLIPS = ROOT / _SWEEP["corpus_root"]
SWEEP_CLIPS: dict[str, str] = _SWEEP["clips"]


def extract(clip: Path, t_s: float) -> Image.Image:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t_s:.3f}", "-i", str(clip), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True,
        check=True,
    )  # fmt: skip
    return Image.open(io.BytesIO(out.stdout)).convert("RGB")


def raw_box(box, cam_file: CameraFile, frame_wh) -> list[float]:
    """Undistorted source px -> the raw frame's px: the corners through the lens, hull."""
    x0, y0, x1, y1 = box
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)
    if cam_file.lens is not None and abs(cam_file.lens.k1) > 1e-12:
        lens = cam_file.lens
        # The lens is fitted on the calibrated frame; the tracker's pixels are the
        # stream's. Scale in, distort, scale out.
        w, h = cam_file.image_size_px
        s_in = np.array([w / frame_wh[0], h / frame_wh[1]])
        corners = distort_points(corners * s_in, lens.k1, lens.centre_px, lens.radius_px) / s_in
    return [float(corners[:, 0].min()), float(corners[:, 1].min()),
            float(corners[:, 0].max()), float(corners[:, 1].max())]  # fmt: skip


def crop_around(img: Image.Image, boxes: list[list[float]], pad: float = 1.2) -> tuple:
    """A crop that holds every box with room, and the offset to subtract from a box."""
    b = np.array(boxes)
    cx, cy = (b[:, 0].min() + b[:, 2].max()) / 2, (b[:, 1].min() + b[:, 3].max()) / 2
    half = max(b[:, 2].max() - b[:, 0].min(), b[:, 3].max() - b[:, 1].min()) * pad / 2
    half = max(half, 120.0)
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(img.width, int(cx + half)), min(img.height, int(cy + half))
    return img.crop((x0, y0, x1, y1)), (x0, y0)


def strip(camera: str, tid: str, death: dict, wit: dict, cam_file: CameraFile, fps: float):
    clip = CLIPS / camera / SWEEP_CLIPS[camera]
    last_f = int(death["last_frame"])
    a = extract(clip, last_f / fps)
    wh = a.size
    last = raw_box(death["last_box"], cam_file, wh)
    tiles = []
    boxes_a = [last]
    crop, off = crop_around(a, boxes_a)
    d = ImageDraw.Draw(crop)
    d.rectangle([last[0] - off[0], last[1] - off[1], last[2] - off[0], last[3] - off[1]],
                outline=(230, 25, 75), width=3)  # fmt: skip
    d.text((4, 4), f"f{last_f} last seen  s{death.get('score_last')}", fill=(255, 255, 0))
    tiles.append(crop.resize((320, 320)))
    if "best_assoc_frame" in wit:
        wf = int(wit["best_assoc_frame"])
        b = extract(clip, wf / fps)
        took = raw_box(wit["best_assoc_box"], cam_file, wh)
        crop, off = crop_around(b, [last, took])
        d = ImageDraw.Draw(crop)
        d.rectangle([last[0] - off[0], last[1] - off[1], last[2] - off[0], last[3] - off[1]],
                    outline=(230, 25, 75), width=2)  # fmt: skip
        d.rectangle([took[0] - off[0], took[1] - off[1], took[2] - off[0], took[3] - off[1]],
                    outline=(60, 180, 75), width=3)  # fmt: skip
        d.text((4, 4), f"f{wf} available s{wit['best_assoc']:.2f} iou{wit['best_iou']:.2f}",
               fill=(255, 255, 0))  # fmt: skip
        tiles.append(crop.resize((320, 320)))
    out = Image.new("RGB", (320 * len(tiles) + 8, 336), (30, 30, 30))
    for i, t in enumerate(tiles):
        out.paste(t, (i * 320 + 4, 12))
    ImageDraw.Draw(out).text(
        (6, 0), f"{camera} track {tid} len {death['length']}", fill=(255, 255, 255)
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="an endings run directory with fleet.json")
    ap.add_argument(
        "--verdict", default="available", help="which mid-view deaths; 'all' for every one"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=24, help="strips per camera at most")
    args = ap.parse_args()
    fleet = json.loads((args.run / "fleet.json").read_text())
    fps = float(fleet["settings"]["fps"])
    args.out.mkdir(parents=True, exist_ok=True)
    for cam in fleet["cameras"]:
        camera = cam["camera"]
        deaths = cam.get("deaths")
        if not deaths:
            print(f"{camera}: this run recorded no death frames (re-run track_endings)")
            continue
        cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
        strips = []
        for tid, death in sorted(deaths.items(), key=lambda kv: -kv[1]["length"]):
            wit = cam["witness_detail"].get(tid, {})
            if args.verdict != "all" and wit.get("verdict") != args.verdict:
                continue
            strips.append(strip(camera, tid, death, wit, cam_file, fps))
            if len(strips) >= args.limit:
                break
        if not strips:
            print(f"{camera}: no {args.verdict} deaths")
            continue
        cols = 2
        rows = (len(strips) + cols - 1) // cols
        w, h = strips[0].size
        sheet = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
        for i, s in enumerate(strips):
            sheet.paste(s, ((i % cols) * w, (i // cols) * h))
        path = args.out / f"{camera}_{args.verdict}.png"
        sheet.save(path)
        print(f"{camera}: {len(strips)} strips -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
