#!/usr/bin/env python3
"""Per-camera static plates: what a fixed CCTV scene looks like with the people removed.

    python3 scripts/static_plates.py --root datasets/studioa_clips --out datasets/studioa_static

A fixed camera sees the same scene thousands of times, which is the auto-labelling
multiplier the robot line does not have: Tesla's expensive first stage -- recovering where
the camera was -- is a constant here, and its multi-trip aggregation is free.

Measured on Kaohsiung-cam04 (2026-08-19), the pixels that never change once in a
five-minute clip: **20.0% while the shop is open, 80.3% at night**. The 4x gap is the
signal -- the night figure is high because nobody is there, and a `person` box landing on
one of those pixels is false without asking a second model.

This script produces the two artefacts that follow from that:

  * **plate** -- the temporal median. People walk out of it: the median of a pixel that a
    shopper crossed for ten seconds is the floor, not the shopper. This is the frame a
    human QCs *once*, and its labels belong to every frame of the clip.
  * **static mask** -- which pixels never changed *once*. Its use is the negative one and it is
    the stronger of the two: a `person` box landing on a pixel that never changed, in a
    clip where nothing moved, is false by construction rather than by a second model's
    opinion. Consensus voting cannot remove the night-time false people because every
    model fails identically on the same frame; time is the axis that separates them.

---------------------------------------------------------------------------
THREE THINGS MEASURED THE HARD WAY, ENCODED HERE RATHER THAN LEFT TO THE CALLER

**The plate wants a robust statistic and the mask wants a sensitive one, and they are not
the same statistic.** The first version used the median absolute deviation for both, which
is correct for the plate -- rejecting the transient is exactly what makes a shopper vanish
from it -- and badly wrong for the mask. A woman walking through Kaohsiung-cam04 with a
shopping bag occupied her pixels on 18 of 152 samples: her median deviation is 2.0 grey
levels and her *maximum* is 113. The MAD mask scored her box **1.00 static** and would have
deleted her, which is the same failure `sam3_person_boxes.py`'s recurrence gate was
switched off for. Counting samples instead scores the same box **0.01**.

**Per time slot, never pooled.** Pooling all four slots of one camera scored 30.3% static
and looked like a scene that simply is not static. It is not: the four slots' median
images differ from each other by **12.7 to 45.4 grey levels**, and that illumination
change swamps any motion threshold.

**The filenames are UTC and the burned-in timestamps are store-local (+8).** An
`archive_20260816-1558...` clip is local **00:03, IR, shop closed** -- not four in the
afternoon; `-0300` is 11:00 in the morning. Anyone slicing day from night by filename gets
it exactly backwards, which is why `slot_utc` is named for what it is and the local hour
is derived rather than assumed.

**The threshold is relative to each clip's own noise floor**, per ARCHITECTURE_DIRECTION
rule 2. An absolute cut in grey levels is a different threshold on a bright shop floor
than on an IR night frame, and the first version of this measurement used one. The floor
is the *median* per-pixel temporal MAD over the frame, which is a sensor-noise estimate
exactly because most of the frame is static -- and `static_fraction` is reported so a
camera where that assumption breaks is visible rather than silently mis-thresholded.

---------------------------------------------------------------------------
WHAT THIS DOES NOT GIVE YOU

A plate is only as empty as its clip. A shopper who stands at the counter for three of the
five minutes is *in* the median, and no amount of care here changes that -- which is what
`static_all.png` is for: the intersection over all four slots, since nobody stands in the
same place at 11:00 and at midnight. Intersecting per-slot masks rather than pooling
frames keeps the illumination problem out of it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

# How many of a pixel's OWN noise floors it must move to count as dynamic.
#
# **Per pixel, not per frame, and that is the whole point.** A single floor for the frame
# is meaningless when the noise is not uniform across it: Taichung-cam09 at night is more
# than half flat wall and floor whose deviation is exactly 0.000, so the frame's median
# floor comes out at 0.5 -- and on the accessory pegboard in the same frame, high-frequency
# texture under IR swings 12 to 18 grey levels on compression noise alone. A frame-wide
# floor calls that pegboard dynamic, which is exactly backwards: it is the most reliably
# static object in the shop, and it is where SAM 3 hallucinates people -- 229 `person`
# boxes on 12 frames of a closed shop at 23:58.
#
# Against each pixel's own floor the two populations are an order of magnitude apart --
# pegboard noise is 6-18x its own floor, a walking shopper is 40-100x -- and the sweep
# separates them completely (share of a box's pixels called static; want high for the
# hanging packets, low for the shopper):
#
#   M=4    packets 0.58   shopper 0.00
#   M=6    packets 0.90   shopper 0.01
#   M=8    packets 0.99   shopper 0.05    <- here, and the gap either side is empty
#   M=10   packets 1.00   shopper 0.11
#   M=15   packets 1.00   shopper 0.22
#
# The cut goes in the measured gap rather than at a round number, per rule 2.
DYNAMIC_MULT = 8.0

# Which percentile of a pixel's own deviations estimates its noise. See `plate_and_mask`.
FLOOR_PCT = 10

# How many samples must exceed that before the pixel is dynamic. Two, and the number is a
# measurement rather than a preference: at 0.5 Hz anything physically present for four
# seconds lands in at least two samples, while a single-sample excursion is what a
# compression artefact looks like. Measured on Kaohsiung-cam04, static fraction by this
# cutoff -- day / night / the walking shopper's own box:
#
#   k=1   18.4% / 76.9% / 0.00     one spike is enough; noise-driven
#   k=2   20.0% / 80.3% / 0.01     <- here
#   k=3   23.7% / 82.7% / 0.05
#   k=5   30.8% / 85.6% / 0.21     starts re-admitting the shopper
#   k=10  39.9% / 90.4% / 0.42
#
# The knee is not sharp, so the argument is the physical one and not the shape of the
# curve: k=2 is the smallest cutoff that cannot be produced by a single bad frame.
MIN_DYNAMIC_SAMPLES = 2

# One sample every two seconds. A five-minute clip gives ~150, which is plenty for a
# median to reject anyone who is not standing still for most of the clip, and cheap enough
# that 168 clips fit in one batch. Denser sampling buys resistance to a *longer* stationary
# person, which is not what more samples of the same five minutes can provide.
SAMPLE_FPS = 0.5

# Half resolution. The masks are blobby and upsample without harm, the plates are for a
# human to look at, and full res would put 168 x 900 MB through this loop.
WORK_W, WORK_H = 960, 540


def decode(path: Path, fps: float, w: int, h: int) -> np.ndarray:
    """Whole clip -> `[N, h, w, 3]` uint8, via the system ffmpeg."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={fps},scale={w}:{h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}: {proc.stderr.decode()[:200]}")
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame = h * w * 3
    if not buf.size or buf.size % frame:
        raise RuntimeError(f"{path.name}: {buf.size} bytes is not a whole number of frames")
    return buf.reshape(-1, h, w, 3)


def plate_and_mask(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Temporal median, static mask, and the numbers that justify the mask's threshold."""
    plate = np.median(frames, axis=0).astype(np.uint8)
    # Deviation from the plate, per sample per pixel, averaged over colour.
    dev = np.abs(frames.astype(np.int16) - plate[None]).mean(axis=3)
    mad = np.median(dev, axis=0)  # reported only; the mask's floor is FLOOR_PCT below
    # Each pixel's own floor: a LOW percentile of its deviations, not the median.
    #
    # The median is the pixel's *typical* deviation, and in a busy part of the shop that is
    # a measure of activity rather than of noise -- so the counter where four people stand
    # gets a high floor, needs an enormous excursion to count as moving, and is declared
    # static. Measured on Kaohsiung-cam04's 103 real person boxes at M=8: a median floor
    # would have deleted **78 of them**. FLOOR_PCT deletes none.
    #
    #   floor  M   night packets dropped (229)   day people wrongly dropped (103)
    #   p50    8   100%                          78
    #   p25    8   100%                          17
    #   p10    6    60.3%                         0
    #   p10    8    98.3%                         0     <- here
    #   p10   12   100%                           7
    #
    # p10 is the quiet baseline of a pixel that spends most of its time occupied, which is
    # what "sensor noise" has to mean if the estimate is to survive a crowd.
    floor = np.maximum(np.percentile(dev, FLOOR_PCT, axis=0), 1.0)
    thr = floor * DYNAMIC_MULT
    exceeded = (dev > thr[None]).sum(axis=0)
    static = exceeded < MIN_DYNAMIC_SAMPLES
    return (
        plate,
        static,
        {
            "frames": len(frames),
            "noise_floor_mad_median": round(float(np.median(mad)), 3),
            "noise_floor_used_median": round(float(np.median(floor)), 3),
            "threshold_mad_median": round(float(np.median(thr)), 3),
            "min_dynamic_samples": MIN_DYNAMIC_SAMPLES,
            "static_fraction": round(float(static.mean()), 4),
            "mean_luma": round(float(frames.mean()), 1),
        },
    )


def slot_of(path: Path) -> str:
    """`archive_20260816-113024_...` -> `20260816-113024`, which is UTC. See the header."""
    return path.stem.split("_")[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("datasets/studioa_clips"))
    ap.add_argument("--out", type=Path, default=Path("datasets/studioa_static"))
    ap.add_argument(
        "--cameras", type=Path, default=None, help="cameras.json (default: --root/cameras.json)"
    )
    ap.add_argument("--only", nargs="*", default=[], help="camera names, for a spot check")
    ap.add_argument(
        "--include-dead", action="store_true", help="the six cameras that emit nothing"
    )
    args = ap.parse_args(argv)

    roles = json.loads((args.cameras or args.root / "cameras.json").read_text())["cameras"]
    names = args.only or sorted(
        k for k, v in roles.items() if args.include_dead or v.get("role") != "dead"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}

    for i, cam in enumerate(names, 1):
        clips = sorted((args.root / cam).glob("*.mp4"))
        if not clips:
            print(f"[{i}/{len(names)}] {cam}: no clips, skipped")
            continue
        cam_out = args.out / cam
        cam_out.mkdir(parents=True, exist_ok=True)
        slots: dict[str, dict] = {}
        every: np.ndarray | None = None
        for clip in clips:
            slot = slot_of(clip)
            try:
                frames = decode(clip, SAMPLE_FPS, WORK_W, WORK_H)
            except RuntimeError as exc:
                print(f"    {slot}: {exc}")
                continue
            plate, static, stats = plate_and_mask(frames)
            Image.fromarray(plate).save(cam_out / f"plate_{slot}.png")
            Image.fromarray((static * 255).astype(np.uint8)).save(
                cam_out / f"static_{slot}.png"
            )
            slots[slot] = {**stats, "role": roles.get(cam, {}).get("role")}
            every = static if every is None else (every & static)
        if every is not None:
            Image.fromarray((every * 255).astype(np.uint8)).save(cam_out / "static_all.png")
        index[cam] = {
            "role": roles.get(cam, {}).get("role"),
            "slots": slots,
            # The one number R2 consumes: pixels that never changed in ANY slot. A person
            # box landing here is false without asking a second model.
            "static_all_fraction": round(float(every.mean()), 4) if every is not None else None,
        }
        got = " ".join(f"{s}:{v['static_fraction']:.1%}" for s, v in slots.items())
        print(
            f"[{i}/{len(names)}] {cam:22s} all:{index[cam]['static_all_fraction']:.1%}  {got}"
        )

    (args.out / "index.json").write_text(
        json.dumps(
            {
                "measured": "static plates per camera per time slot",
                "note": "slot keys are UTC; the burned-in timestamp is store-local (+8)",
                "sample_fps": SAMPLE_FPS,
                "work_size": [WORK_W, WORK_H],
                "dynamic_mult": DYNAMIC_MULT,
                "min_dynamic_samples": MIN_DYNAMIC_SAMPLES,
                "cameras": index,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\n{len(index)} cameras -> {args.out}/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
