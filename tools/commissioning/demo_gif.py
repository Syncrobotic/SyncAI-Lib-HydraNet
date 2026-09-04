#!/usr/bin/env python3
"""The README figure, as code, plus the check that says it may be published.

    uv run python tools/commissioning/demo_gif.py Kaohsiung-cam04

**A figure of a customer's shop floor that cannot be re-made is a figure nobody can
re-check when the pipeline behind it moves.** Both README figures were once cut by hand,
and the blur on one of them was applied by hand too.

---------------------------------------------------------------------------
THE CHECK, AND ONE THAT WAS TRIED AND DOES NOT WORK

The bar is the one the 2026-08-28 README figure was held to: every frame of the gif read
by eye on contact sheets, and **the detector re-run on the SOURCE frames at a threshold
far below the render's, with every person it finds required to fall inside a region the
render blurred.** That is what this reproduces, using `utils.face_blur.blur_rect` rather than a
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
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


from syncai_hydranet.data.video import frames as decode_frames
from syncai_hydranet.data.video import probe as probe_video
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.shipped import (
    SHIPPED_RUN,
    load_model,
)
from syncai_hydranet.utils.face_blur import blur_rect, plate_person_boxes
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")

# The run the tools ship from, named once in `syncai_hydranet.shipped`. Six files
# used to carry their own copy of this string and the best run was in none of them.
RUN = SHIPPED_RUN
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
    head = blur_rect(w, h, *box)
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
        "assets/dev/demo_<camera>_<stamp>.mp4. `assets/dev/` is where renders live; "
        "`assets/` itself holds only the figures cut from them.",
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
        stamped = sorted(ROOT.glob(f"assets/dev/demo_{a.camera}_2*.mp4"))
        if not stamped:
            print(
                f"no stamped render for {a.camera}. Run demo_video.py first; "
                f"assets/demo_{a.camera}.mp4 is a copy and does not say which run it is.",
                file=sys.stderr,
            )
            return 1
        video = stamped[-1]
    # **A sidecar beside the render wins over the per-camera log.** `cli/scene.py` writes
    # `<render>.render.json`, because the per-camera path below is overwritten by the next
    # render of that camera -- so a verdict naming an older render cannot be checked
    # against the record that produced it. `demo_video.py` still writes the per-camera
    # file and is unchanged; this only adds a place to look first.
    sidecar = video.with_suffix(".render.json")
    log = (
        sidecar if sidecar.exists() else ROOT / f"runs/commission01/{a.camera}.demo_tracks.json"
    )
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

    # `ROOT /` rather than `Path(...)`: an absolute `--out` is returned unchanged and a
    # relative one is anchored to the repository, which is what every other path here
    # already assumes. Given `--out assets/x.gif` from the repo root, the old form
    # produced a relative path, `verdict.relative_to(ROOT)` raised **after** the verdict
    # was written and before the gif was -- leaving a verdict on disk describing a render
    # that had not replaced the figure beside it.
    out = ROOT / Path(a.out) if a.out else ROOT / f"assets/demo_{a.camera}.gif"
    verdict = out.with_suffix(".audit.json")
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
        # The threshold THIS render blurred at, not today's constant. Absent means the
        # render predates the field, and there is then no way to know what it covered --
        # so the audit refuses rather than assuming the current value, which is larger and
        # would make every old figure look cleaner than it was.
        if "blur_score_thr" not in meta:
            print(
                f"{log.name} records no `blur_score_thr`, so this render predates "
                "2026-08-28 and what it blurred at is unknown. Re-render before cutting "
                "a figure from it; auditing it against today's threshold would report a "
                "blur set larger than the one actually applied.",
                file=sys.stderr,
            )
            return 1
        if not meta.get("blur_faces", True):
            print(
                f"{log.name} says this render was made with --no-blur. A figure cut from "
                "it must not be published.",
                file=sys.stderr,
            )
            return 1
        render_blur_thr = float(meta["blur_score_thr"])
        model, cfg, device = load_model(
            str(RUN / "config.yaml"), RUN / "last.pt", validate=False
        )
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
            # exactly what `demo_video` blurred: its OWN detector threshold, plus the
            # static-plate instrument. Neither is re-derived here; both are imported.
            b_blur, _ = person_boxes(frame, model, device, size, person_label, render_blur_thr)
            rects = [r for bb in b_blur if (r := blur_rect(src_w, src_h, *bb)) is not None]
            if plate_arr is not None:
                rects += [
                    r
                    for bb in plate_person_boxes(frame, plate_arr)
                    if (r := blur_rect(src_w, src_h, *bb)) is not None
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
        # **Written whether it passed or failed, and that is the whole point.** A verdict
        # file that only appears on success makes `failing == 0` true by construction, so
        # a guard reading it would be green about the act of writing rather than about the
        # frames. Written either way, a figure committed by hand after a refused audit
        # carries the refusal next to it.
        verdict.write_text(
            json.dumps(
                {
                    "camera": a.camera,
                    "gif": out.name,
                    "render": video.name,
                    "source_clip": meta["clip"],
                    # **The recipe, carried out of the track log into the tracked file.**
                    # `runs/` is gitignored, so a record that stays there is a record CI
                    # cannot read and a re-cutter on a fresh checkout does not have. The
                    # figure a second session re-cut without `--staff-colours` on
                    # 2026-08-29 passed every check here because the flags were nowhere
                    # in this file. `.get` rather than `[]`: a render from before the
                    # track log carried them writes `null`, and
                    # `tests/test_figures_are_audited.py` refuses that rather than
                    # reading a missing key as agreement.
                    "render_args": meta.get("args"),
                    "staff_model": meta.get("staff_model"),
                    "commit": subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    ).stdout.strip()
                    or None,
                    "start_frame": start,
                    "frames": a.count,
                    "stride": a.stride,
                    "audit_score_thr": AUDIT_THR,
                    "blur_score_thr": render_blur_thr,
                    "head_coverage_required": HEAD_COVERAGE,
                    "person_boxes_checked": n_people,
                    "person_boxes_failing": len(naked),
                    "failing_examples": [
                        {"source_frame": i, "box": bb, "score": round(s0, 3)}
                        for i, bb, s0, _ in naked[:10]
                    ],
                    "contact_sheets": str(sheets_dir.relative_to(ROOT)),
                },
                indent=1,
            )
            + "\n"
        )
        print(f"  verdict -> {verdict.relative_to(ROOT)}")
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
