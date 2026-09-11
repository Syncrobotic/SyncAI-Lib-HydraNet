#!/usr/bin/env python3
"""Four readings of one crowded counter, side by side, for three minutes.

    python3 scripts/counter_compare_video.py --camera Kaohsiung-cam04 \\
        --clip datasets/studioa_clips/Kaohsiung-cam04/archive_<start>_<end>.mp4

The four panels are the four things PLAN 7a.41 measured on stills on 2026-09-10, on the
same frames at the same moment, so a reader can see rather than be told:

    top-left      the shipped detector (person01) at the shipped 0.35 -- the front row
    top-right     every box down to 0.15, the low band in green where the dense head
                  vouches for it -- mechanism 1's candidates, and why a huddle is one blob
    bottom-left   the same detector on the counter window upscaled x2 -- mechanism 2
    bottom-right  SAM 3 on that window -- the labels person02 trains on (mechanism 3)

Each panel carries its count of high-band boxes. The window is the densest 960x600 of the
detector's own boxes over the first --window-frames frames, the same rule
`tools/annotation/counter_person_labels.py` uses, so this and the dataset look at the
same region. Written to assets/dev/ (ignored wholesale: customer footage) to a .part and
renamed on success. Nothing is blurred -- it is a working comparison, not a figure, and
the boxes are the content.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from syncai_bev3d.teachers.boxes import boxes_from_masks  # noqa: E402
from syncai_bev3d.teachers.sam3 import load_sam3, segment  # noqa: E402
from syncai_hydranet import shipped  # noqa: E402
from syncai_hydranet.analytics.clip_tracks import PERSON, to_source_pixels  # noqa: E402
from syncai_hydranet.data.video import frames, probe  # noqa: E402
from syncai_hydranet.serving.camera import BIRTH_REF  # noqa: E402
from syncai_hydranet.serving.decode import confirm_mask  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import preprocess  # noqa: E402

RED, GREEN, GREY, YELLOW, WHITE = (
    (230, 25, 75),
    (60, 180, 75),
    (150, 150, 150),
    (255, 220, 0),
    (255, 255, 255),
)


def densest_window(centres, frame_wh, window_wh, cell=20):
    fw, fh = frame_wh
    ww, wh = window_wh
    nx, ny = fw // cell, fh // cell
    if len(centres) == 0:
        return (fw - ww) // 2, (fh - wh) // 2, (fw + ww) // 2, (fh + wh) // 2
    hist, _, _ = np.histogram2d(
        centres[:, 1], centres[:, 0], bins=[ny, nx], range=[[0, fh], [0, fw]]
    )
    kx, ky = ww // cell, wh // cell
    summed = ndimage.uniform_filter(hist, size=(ky, kx), mode="constant")
    sub = summed[ky // 2 : ny - (ky - ky // 2) + 1, kx // 2 : nx - (kx - kx // 2) + 1]
    iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
    cx, cy = (ix + kx // 2 + 0.5) * cell, (iy + ky // 2 + 0.5) * cell
    x0 = int(min(max(cx - ww / 2, 0), fw - ww))
    y0 = int(min(max(cy - wh / 2, 0), fh - wh))
    return x0, y0, x0 + ww, y0 + wh


def draw_boxes(img, boxes, scores, colour, width=3, label=True):
    d = ImageDraw.Draw(img)
    for b, s in zip(boxes, scores, strict=True):
        d.rectangle(list(map(float, b)), outline=colour, width=width)
        if label:
            d.text((b[0] + 3, b[1] + 3), f"{s:.2f}", fill=colour)


def caption(img, text, font):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
    d.text((12, 8), text, fill=WHITE, font=font)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--window", default="960x600")
    ap.add_argument(
        "--window-frames", type=int, default=100, help="frames the window is found over"
    )
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--sam3-score", type=float, default=0.5)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "assets/dev")
    args = ap.parse_args()
    ww, wh = (int(v) for v in args.window.lower().split("x"))

    device = pick_device(None)
    model, cfg, _ = shipped.load_model(
        shipped.SHIPPED_CONFIG, shipped.for_detection(), device=device
    )
    person_id = list(cfg["data"]["terrain_classes"]).index("person")
    size = cfg["data"]["input_size"]
    proc, sam = load_sam3("facebook/sam3", str(device))
    w, h, _ = probe(args.clip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    def detect(img_np, thr):
        x, _, region = preprocess(Image.fromarray(img_np), size)
        import torch

        with torch.no_grad():
            res = model.predict(x.to(device), score_thr=thr)
        det = res["detection"][0]
        lab = det["labels"].cpu().numpy()
        keep = lab == PERSON
        cb = det["boxes"].cpu().numpy()[keep]
        sc = det["scores"].cpu().numpy()[keep]
        cmap = res["terrain"][0].cpu().numpy()
        vouched = confirm_mask(cb, cmap, person_id)
        return to_source_pixels(cb, region, img_np.shape[1], img_np.shape[0]), sc, vouched

    # Pass 1: the window, from the detector's own boxes over the opening frames.
    centres = []
    for n, fr in enumerate(frames(args.clip, w, h, args.fps)):
        if n >= args.window_frames:
            break
        b, s, _ = detect(fr, BIRTH_REF)
        centres += [[(x0 + x1) / 2, (y0 + y1) / 2] for x0, y0, x1, y1 in b]
    window = densest_window(np.array(centres).reshape(-1, 2), (w, h), (ww, wh))
    x0, y0, x1, y1 = window
    print(f"window {window}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"compare_{args.camera}_{stamp}.mp4"
    part = out.with_suffix(".mp4.part")
    pw, ph = 960, 540  # each panel
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{pw * 2}x{ph * 2}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-f", "mp4", str(part)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    assert ff.stdin is not None
    t0 = time.perf_counter()
    counts = {"a": [], "b": [], "c": [], "d": []}
    for n, fr in enumerate(frames(args.clip, w, h, args.fps)):
        if n >= args.frames:
            break
        base = Image.fromarray(fr).convert("RGB")
        b, s, vouched = detect(fr, 0.15)
        hi = s >= BIRTH_REF
        # A: shipped
        a = base.copy()
        draw_boxes(a, b[hi], s[hi], RED)
        caption(a, f"A  shipped detector @0.35   {int(hi.sum())} people", font)
        # B: everything to 0.15, vouched low in green
        bb = base.copy()
        draw_boxes(bb, b[hi], s[hi], RED)
        lo_v = (~hi) & vouched
        lo_n = (~hi) & (~vouched)
        draw_boxes(bb, b[lo_v], s[lo_v], GREEN)
        draw_boxes(bb, b[lo_n], s[lo_n], GREY, width=1, label=False)
        caption(
            bb,
            f"B  down to 0.15: +{int(lo_v.sum())} low boxes the dense head vouches for (green)",
            font,
        )
        # C: x2 crop pass
        crop = fr[y0:y1, x0:x1]
        big = np.asarray(
            Image.fromarray(crop).resize(
                (int(crop.shape[1] * args.upscale), int(crop.shape[0] * args.upscale)),
                Image.Resampling.LANCZOS,
            )
        )
        cb, cs, _ = detect(big, 0.15)
        cb = cb / args.upscale + np.array([x0, y0, x0, y0])
        c = base.copy()
        ImageDraw.Draw(c).rectangle(list(window), outline=YELLOW, width=3)
        chi = cs >= BIRTH_REF
        draw_boxes(c, cb[chi], cs[chi], RED)
        draw_boxes(c, cb[~chi], cs[~chi], GREY, width=1, label=False)
        in_win_a = int(
            ((b[:, 0] + b[:, 2]) / 2 >= x0)
            .__and__((b[:, 0] + b[:, 2]) / 2 < x1)
            .__and__(hi)
            .sum()
        )
        caption(
            c,
            f"C  x{args.upscale:.0f} counter pass @0.35   {int(chi.sum())} people in the "
            f"window (A had {in_win_a})",
            font,
        )
        # D: SAM 3 on the crop
        sm = boxes_from_masks(
            segment(proc, sam, Image.fromarray(big), "person", args.sam3_score, str(device))
        )
        if len(sm):
            sm[:, :4] = sm[:, :4] / args.upscale + np.array([x0, y0, x0, y0])
        d = base.copy()
        ImageDraw.Draw(d).rectangle(list(window), outline=YELLOW, width=3)
        draw_boxes(d, sm[:, :4], sm[:, 4], RED)
        caption(
            d,
            f"D  SAM 3 on the x2 window @{args.sam3_score}   {len(sm)} people -- "
            "person02's labels",
            font,
        )
        counts["a"].append(int(hi.sum()))
        counts["b"].append(int(lo_v.sum()))
        counts["c"].append(int(chi.sum()))
        counts["d"].append(len(sm))
        sheet = Image.new("RGB", (pw * 2, ph * 2))
        for k, im in enumerate((a, bb, c, d)):
            sheet.paste(im.resize((pw, ph)), ((k % 2) * pw, (k // 2) * ph))
        stamp_text = f"{args.camera}  t={n / args.fps:5.1f}s"
        sd = ImageDraw.Draw(sheet)
        tw = sd.textlength(stamp_text, font=font)
        sd.text((pw * 2 - tw - 16, ph * 2 - 34), stamp_text, fill=WHITE, font=font)
        ff.stdin.write(np.asarray(sheet).tobytes())
        if n % 100 == 0:
            means = {k: np.mean(v) for k, v in counts.items()}
            print(
                f"frame {n}  {time.perf_counter() - t0:.0f}s  A {means['a']:.1f} "
                f"B+{means['b']:.1f} C {means['c']:.1f} D {means['d']:.1f}",
                flush=True,
            )
    ff.stdin.close()
    ff.wait()
    part.rename(out)
    means = {k: np.mean(v) for k, v in counts.items()}
    print(
        f"wrote {out}  frames {len(counts['a'])}  mean people A {means['a']:.1f}  "
        f"B extra {means['b']:.1f}  C {means['c']:.1f}  D {means['d']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
