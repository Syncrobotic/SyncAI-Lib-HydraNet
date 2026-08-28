#!/usr/bin/env python3
"""The README figure, as code, plus the check that says it may be published.

    uv run python tools/commissioning/demo_gif.py Kaohsiung-cam04

`assets/demo_Taichung-cam10.gif` was made by hand. That is the same shape as the defect
`bddec6c` fixed a few hours earlier -- the previous README figure had been *blurred* by
hand and no one could reproduce it -- and it survived only because the blur was the half
somebody noticed. A figure of a customer's shop floor that cannot be re-made is a figure
nobody can re-check when the pipeline behind it moves, which it did twice on 2026-08-28.

---------------------------------------------------------------------------
THE CHECK, AND ONE THAT WAS TRIED AND DOES NOT WORK

The bar is the one the 2026-08-28 README figure was held to: every frame of the gif read
by eye on contact sheets, and **the detector re-run on the SOURCE frames at a threshold
far below the render's, with every person it finds required to fall inside a region the
render blurred.** That is what this reproduces, using `demo_video.blur_rect` rather than a
second copy of the rectangle arithmetic -- a copy would answer the question about itself.

**What was tried first and measured wrong, because it is the kind of check that looks
fine.** Run the detector on the *rendered panel* and refuse any person whose head still
has fine texture. It does not work, and the numbers say why rather than an argument:
over 120 frames of Kaohsiung-cam04 it found 921 boxes whose head-gradient ratios were
p50 0.99, p90 3.17, with no separation in any score band -- and **zero boxes above 0.35**,
where the render itself had people every frame. The rendered panel is half resolution,
carries drawn boxes and grey slabs, and has been through h264; the detector is not
reading the image it was trained on, and gradient energy cannot tell a face from
merchandise anyway. Both halves of that instrument answered a different question than the
one asked.

The contact sheets are still written and are still the thing a person has to look at.
This refuses a figure that is definitely wrong; it cannot promise one is right.

---------------------------------------------------------------------------
WHAT IT WILL NOT DO

**It cannot tell a `--no-blur` render by its name, and does not pretend to.** What stops
one being published is the audit: the source-frame check is against the blur regions the
render *would* have applied, so an unblurred render fails on the first frame containing a
person only if the auditor knows the render skipped them -- it does not. `--skip-audit`
and `--no-blur` are both deliberate acts, and neither is a filename this can inspect.
Do not publish a figure cut from a `--no-blur` render.

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import demo_video

from syncai_hydranet.config import load_config
from syncai_hydranet.data.video import frames as decode_frames
from syncai_hydranet.data.video import probe as probe_video
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
RUN = ROOT / "runs/hydranet_retail_security_b03_cw_xl"
GIF_W = 760  # what `assets/demo_Taichung-cam10.gif` ships at, kept so the README is even
AUDIT_THR = 0.03  # far under the render's 0.35: the audit has to see what the render did not
# A head is "covered" when this much of it lies inside the union of blurred rectangles.
# Not 1.0: the rectangles are axis-aligned and a person leaning gives a head box whose
# corner pixels sit outside one by a few pixels. 0.98 leaves no room for a face.
HEAD_COVERAGE = 0.98
SHEET_COLS = 5


def person_boxes(frame: np.ndarray, model, device, size, person_label, thr: float):
    """Person boxes in one SOURCE frame, in source pixels, at `thr`."""
    img = Image.fromarray(frame)
    x, _canvas, region = preprocess(img, size)
    with torch.no_grad():
        out = model.predict(x.to(device), score_thr=thr)
    det = out.get("detection", [{}])[0]
    if not det or not len(det.get("boxes", [])):
        return np.zeros((0, 4)), np.zeros(0)
    b = det["boxes"].cpu().numpy()
    lab = det["labels"].cpu().numpy()
    sc = det["scores"].cpu().numpy()
    x0, y0, cw, _ch = region
    b = (b - np.array([x0, y0, x0, y0])) * (frame.shape[1] / cw)
    keep = lab == person_label
    return b[keep], sc[keep]


def head_uncovered(box, rects, w: int, h: int) -> float:
    """Fraction of this person's head that no blurred rectangle covers.

    A mask rather than pairwise IoU, because a head is routinely covered by two
    overlapping rectangles and neither covers it alone -- which is the case a per-rectangle
    test reports as a naked face.
    """
    head = demo_video.blur_rect(w, h, *box)
    if head is None:
        return 0.0  # too small to carry a readable face; `blur_rect` declines it too
    hx0, hy0, hx1, hy1 = head
    mask = np.zeros((hy1 - hy0, hx1 - hx0), bool)
    for rx0, ry0, rx1, ry1 in rects:
        ox0, oy0 = max(rx0, hx0), max(ry0, hy0)
        ox1, oy1 = min(rx1, hx1), min(ry1, hy1)
        if ox1 > ox0 and oy1 > oy0:
            mask[oy0 - hy0 : oy1 - hy0, ox0 - hx0 : ox1 - hx0] = True
    return float(1.0 - mask.mean())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("camera")
    ap.add_argument(
        "--video",
        default=None,
        help="the render to cut from. Default: the newest stamped "
        "assets/demo_<camera>_<stamp>.mp4, which is never a --no-blur render.",
    )
    ap.add_argument(
        "--start",
        default="auto",
        help="first frame of the window, or `auto` (default): the busiest run of "
        "--count frames, read off the render's own `<camera>.demo_tracks.json`. A "
        "hand-picked number is a choice nobody can re-derive, and on a quiet camera the "
        "difference is the whole figure -- Tao-Hsin-cam04's default clip has someone in "
        "10%% of its frames.",
    )
    ap.add_argument("--count", type=int, default=120, help="frames in the gif")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth frame")
    ap.add_argument("--duration-ms", type=int, default=100)
    ap.add_argument("--width", type=int, default=GIF_W)
    ap.add_argument(
        "--colors",
        type=int,
        default=128,
        help="palette size. 128 keeps the blue/green figure colours separable from the "
        "shop's greys, which 64 does not.",
    )
    ap.add_argument(
        "--out", default=None, help="default assets/demo_<camera>.gif, the README's name"
    )
    ap.add_argument(
        "--skip-audit",
        action="store_true",
        help="write the gif without the unblurred-face audit. Only for a private check; "
        "a figure that has not been audited must not be committed.",
    )
    a = ap.parse_args()

    if a.video:
        video = Path(a.video)
    else:
        stamped = sorted(ROOT.glob(f"assets/demo_{a.camera}_2*.mp4"))
        if not stamped:
            print(
                f"no stamped render for {a.camera}. Run demo_video.py first; "
                f"assets/demo_{a.camera}.mp4 is a copy and does not say which run it is.",
                file=sys.stderr,
            )
            return 1
        video = stamped[-1]
    log = ROOT / f"runs/commission01/{a.camera}.demo_tracks.json"
    if not log.exists():
        print(
            f"{log} is missing. demo_video.py writes it beside every render, and both the "
            "window and the audit are read from it; there is nothing to cut from safely.",
            file=sys.stderr,
        )
        return 1
    meta = json.loads(log.read_text())
    start = a.start
    if isinstance(start, str) and start == "auto":
        d = meta
        occ = np.zeros(max(d["frames"], 1))
        for row in d["positions"]:
            occ[row["frame"]] += 1
        span = a.count * a.stride
        if span > len(occ):
            print(f"{span} frames asked for and the render is {len(occ)}", file=sys.stderr)
            return 1
        window = np.convolve(occ, np.ones(span), "valid")
        start = int(window.argmax())
        held = float((occ[start : start + span] > 0).mean())
        print(
            f"{a.camera}: busiest window starts at {start} -- "
            f"{int(window[start])} placements, someone in {held * 100:.0f}% of its frames"
        )
        if held < 0.5:
            print(
                "  and that is the BUSIEST window in this render: over half its frames "
                "are empty. A figure cut from it shows an empty shop; consider a "
                "different clip before publishing it.",
                file=sys.stderr,
            )
    else:
        start = int(start)
    print(f"{a.camera}: cutting {a.count} frames from {video.name} at {start}")

    w, h, _fps = probe_video(str(video))
    picked: list[Image.Image] = []
    for i, frame in enumerate(decode_frames(str(video), w, h, None)):
        if i < start:
            continue
        if (i - start) % a.stride:
            continue
        picked.append(Image.fromarray(frame.copy()))
        if len(picked) >= a.count:
            break
    if len(picked) < a.count:
        print(
            f"{video.name} gave {len(picked)} frames from {start} and {a.count} were "
            "asked for; the window runs past the end of the render.",
            file=sys.stderr,
        )
        return 1

    out = Path(a.out) if a.out else ROOT / f"assets/demo_{a.camera}.gif"
    sheets_dir = ROOT / f"runs/commission01/{a.camera}.gif_check"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    # Contact sheets first, and of *every* gif frame: the audit below is a refusal, and a
    # person still has to look. Six sheets of twenty was what the 2026-08-28 figure was
    # checked on; the layout follows the frame count rather than a fixed number of sheets.
    for old in sheets_dir.glob("sheet_*.png"):
        old.unlink()
    per_sheet = SHEET_COLS * 4
    tw = 480
    th = round(picked[0].height * tw / picked[0].width)
    for s0 in range(0, len(picked), per_sheet):
        chunk = picked[s0 : s0 + per_sheet]
        rows = (len(chunk) + SHEET_COLS - 1) // SHEET_COLS
        sheet = Image.new("RGB", (SHEET_COLS * tw, rows * th), (10, 12, 16))
        for k, im in enumerate(chunk):
            sheet.paste(im.resize((tw, th)), ((k % SHEET_COLS) * tw, (k // SHEET_COLS) * th))
        sheet.save(sheets_dir / f"sheet_{s0 // per_sheet:02d}.png")
    n_sheets = len(list(sheets_dir.glob("sheet_*.png")))
    print(f"  {n_sheets} contact sheets over {len(picked)} frames -> {sheets_dir}")

    if not a.skip_audit:
        # The audit runs on the SOURCE clip, not on the render. `demo_tracks.json` names
        # the clip and the fps the render decoded at, so the window maps back exactly.
        clip = ROOT / "datasets/studioa_clips" / a.camera / meta["clip"]
        if not clip.exists():
            print(f"the render's source clip is gone: {clip}", file=sys.stderr)
            return 1
        cfg = load_config(str(RUN / "config.yaml"), validate=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(cfg).to(device).eval()
        model.load_state_dict(select_weights(load_checkpoint(RUN / "last.pt"), "ema"))
        size = cfg["data"]["input_size"]
        person_label = list(cfg["model"]["heads"]["detection"]["classes"]).index("person")
        src_w, src_h, _ = probe_video(str(clip))
        plate_arr = None
        cf = CameraFile.load(ROOT / f"runs/commission01/{a.camera}.camera.json")
        plate_path = ROOT / cf.plate_file if cf.plate_file else None
        if plate_path is not None and plate_path.exists():
            plate_arr = np.asarray(
                Image.open(plate_path).convert("RGB").resize((src_w, src_h)), np.uint8
            )
        naked: list[tuple[int, list, float, float]] = []
        n_people = 0
        for i, frame in enumerate(decode_frames(str(clip), src_w, src_h, meta["fps"])):
            if i < start:
                continue
            if i >= start + a.count * a.stride:
                break
            if (i - start) % a.stride:
                continue
            # exactly what `demo_video` blurs: its detector set at BLUR_THR, plus the
            # static-plate instrument. Neither is re-derived here; both are imported.
            b_blur, _ = person_boxes(
                frame, model, device, size, person_label, demo_video.BLUR_THR
            )
            rects = [
                r for bb in b_blur if (r := demo_video.blur_rect(src_w, src_h, *bb)) is not None
            ]
            if plate_arr is not None:
                rects += [
                    r
                    for bb in demo_video.plate_person_boxes(frame, plate_arr)
                    if (r := demo_video.blur_rect(src_w, src_h, *bb)) is not None
                ]
            b_audit, sc = person_boxes(frame, model, device, size, person_label, AUDIT_THR)
            for bb, s0 in zip(b_audit, sc, strict=True):
                n_people += 1
                un = head_uncovered(bb, rects, src_w, src_h)
                if un > 1.0 - HEAD_COVERAGE:
                    naked.append((i, [round(float(v)) for v in bb], float(s0), un))
        print(
            f"  audit on {clip.name} at score {AUDIT_THR}: {n_people} person boxes over "
            f"{a.count} frames, {len(naked)} whose head is not inside a blurred region"
        )
        if naked:
            for i, bb, s0, un in naked[:10]:
                print(
                    f"    source frame {i}: box {bb} score {s0:.2f} "
                    f"head {un * 100:.0f}% outside every blurred rectangle",
                    file=sys.stderr,
                )
            print(
                f"{a.camera}: NOT written. {len(naked)} person boxes in this window have a "
                "head the render did not blur, and this is a customer's shop floor. Look "
                f"at {sheets_dir}, then move the window or fix the blur.",
                file=sys.stderr,
            )
            return 1

    gw = a.width
    gh = round(picked[0].height * gw / picked[0].width)
    small = [im.resize((gw, gh), Image.Resampling.LANCZOS) for im in picked]
    # One palette for the whole animation, from the busiest frame, and every frame mapped
    # onto it. Saving RGB frames and letting Pillow quantise each one independently gives
    # a per-frame palette, no cross-frame reuse and a file eight times the size -- 12.4 MB
    # against 1.6 MB for the figure already in the README, measured on this same window.
    #
    # **No dithering, and it is the single biggest term in the file size.** Measured on
    # this window: 128 colours dithered is 6.96 MB and undithered is 2.15 MB, 64 colours
    # 5.97 against 1.67. Dither trades a little banding for a lot of high-frequency noise,
    # and noise is exactly what GIF's run-length compression cannot pack. 128 colours
    # undithered keeps the blue and green figure colours separable from the shop's greys,
    # which is the one thing this figure exists to show.
    base = max(small, key=lambda im: len(im.getcolors(maxcolors=1 << 24) or [1]))
    pal = base.quantize(colors=a.colors, method=Image.Quantize.MEDIANCUT)
    small = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in small]
    small[0].save(
        out,
        save_all=True,
        append_images=small[1:],
        duration=a.duration_ms,
        loop=0,
        optimize=True,
    )
    print(
        f"wrote {out} ({len(small)} frames, {gw}x{gh}, {a.duration_ms} ms, "
        f"{out.stat().st_size / 1e6:.2f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
