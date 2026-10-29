#!/usr/bin/env python3
"""Second pass over a `pose_overlay.py` video: draw the events on the people.

`pose_overlay.py` writes its video before it computes events, so the video carries boxes
and skeletons and the events live only in `--events-json`. This reads both and writes a
new video where, for every frame inside an event, the track's box is redrawn in the
event's colour with a label, and a banner at the top names every active event.

    python tools/pose/events_overlay.py assets/smoke_Tao-Hsin-cam03.mp4 \\
        runs/pose_smoke_taohsin03.json assets/smoke_Tao-Hsin-cam03_labelled.mp4

Boxes in the JSON are in SOURCE pixels (the clip's own resolution); the overlay video is
usually smaller, so `--source-size` (default 1920x1080) is used to rescale them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLOURS = {
    "reach_to_shelf": (255, 140, 0),
    "crouch": (255, 220, 0),
    "fall": (255, 40, 40),
}
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


def _font(size: int) -> ImageFont.ImageFont:
    for c in FONT_CANDIDATES:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def _probe(path: Path) -> tuple[int, int, str]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=p=0", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    return int(out[0]), int(out[1]), out[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("events_json")
    ap.add_argument("out")
    ap.add_argument("--source-size", default="1920x1080", help="WxH the JSON boxes are in")
    a = ap.parse_args()

    video, out = Path(a.video), Path(a.out)
    w, h, rate = _probe(video)
    sw, sh = (int(v) for v in a.source_size.lower().split("x"))
    sx, sy = w / sw, h / sh

    events = json.load(open(a.events_json))["events"]
    # frame -> list of (type, track_id, box in video pixels)
    per_frame: dict[int, list[tuple[str, int, tuple[float, float, float, float]]]] = defaultdict(list)
    for e in events:
        for r in e["track_rows"]:
            if not r["in_event"] or "box" not in r:
                continue
            b = r["box"]
            per_frame[r["frame"]].append(
                (e["type"], r["track_id"], (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy))
            )
    print(f"{len(events)} events cover {len(per_frame)} frames of {video.name}")

    font = _font(max(14, h // 40))
    banner_font = _font(max(16, h // 32))

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    enc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", rate, "-i", "-",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p", str(out),
        ],
        stdin=subprocess.PIPE,
    )
    assert dec.stdout is not None and enc.stdin is not None
    n_bytes = w * h * 3
    n = 0
    while True:
        buf = dec.stdout.read(n_bytes)
        if len(buf) < n_bytes:
            break
        hits = per_frame.get(n)
        if hits:
            img = Image.frombuffer("RGB", (w, h), buf, "raw", "RGB", 0, 1).copy()
            d = ImageDraw.Draw(img)
            for etype, tid, (x0, y0, x1, y1) in hits:
                col = COLOURS.get(etype, (255, 0, 255))
                d.rectangle((x0, y0, x1, y1), outline=col, width=4)
                label = f"{etype}  #{tid}"
                tw, th = d.textbbox((0, 0), label, font=font)[2:]
                ty = max(0, y0 - th - 6)
                d.rectangle((x0, ty, x0 + tw + 8, ty + th + 6), fill=col)
                d.text((x0 + 4, ty + 2), label, fill=(0, 0, 0), font=font)
            names = sorted({f"{t} #{i}" for t, i, _ in hits})
            banner = "EVENT  " + "   ".join(names)
            bw, bh = d.textbbox((0, 0), banner, font=banner_font)[2:]
            d.rectangle((0, 0, bw + 24, bh + 12), fill=(200, 0, 0))
            d.text((12, 6), banner, fill=(255, 255, 255), font=banner_font)
            buf = np.asarray(img).tobytes()
        enc.stdin.write(buf)
        n += 1
    enc.stdin.close()
    dec.stdout.close()
    enc.wait()
    dec.wait()
    print(f"wrote {out}  ({n} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
