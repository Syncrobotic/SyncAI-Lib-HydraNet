#!/usr/bin/env python3
"""E-prep: does the monocular depth teacher agree with the robot's own range sensor?

    python3 scripts/robot/eprep_depth_residual.py --capture datasets/robot_eprep/stand01 \\
        --stride 60 --out runs/eprep/stand01

`docs/RESEARCH_OCCUPANCY.md` makes the whole occupancy programme turn on one number: run
Depth-Anything V2 over robot video, calibrate it against `ultrasound[2]` and odometry,
measure the residual. Below a threshold, the auto-labeller has a metric floor and E2/E3
have labels; above it, we are buying a depth sensor. This script is that measurement, on
a capture made by `robot_capture.py`.

It reports a residual. It also reports the thing that decides whether the residual means
anything, which is **where in the image the sonar reading actually comes from**.

---------------------------------------------------------------------------
WHAT THE SPEC SAYS THE SENSOR IS, AND WHAT IT TURNS OUT TO MEASURE

The comm spec (`assets/838272320-...V1-0-7.pdf`) defines
`ultrasound[2] = {forward_distance, backward_distance}` -- *forward* and **backward**, not
forward and down -- with an effective range of `[0.28 m, 4.50 m]` that saturates at both
ends. So of the "two metric range points per frame" the research direction counts on, the
backward one has no camera looking at it, and the usable anchor is **one scalar**.

That scalar is not a forward depth sample. On the first capture -- robot standing still in
an office corridor, scene provably static across 90 s -- `forward_distance` is **bimodal**,
flipping between 1.05 m and 1.45 m several times a second, each cluster tight to +/-0.02 m.
The sensor is precise; what changes is which surface it locks onto. Projecting both shells
into the depth map puts 86% of the 1.05 m pixels in the right-hand third of the frame and
the 1.45 m pixels split evenly left and right: they are the **corridor walls**, caught by a
wide sonar beam, while the corridor straight ahead is empty out to the glass doors at 5 m.

Two consequences, and they are the reason this script reports geometry and not just error:

  * **Never average the sonar over a window.** The mean of a bimodal echo is 1.3 m, a
    distance at which this scene contains nothing at all. A teacher calibrated to it is
    fitted to a surface that does not exist.
  * **A residual is only meaningful when the echo is explained by the forward cone.**
    Comparing a teacher's forward depth against a lateral return is not a scale error to be
    corrected, it is a different quantity. So every frame is scored for whether the echo
    lands centrally, and the aggregate residual is reported over that subset -- with the
    discarded fraction stated, because a residual computed over 6 usable frames out of 90
    is a number that should not travel on its own.

---------------------------------------------------------------------------
ASSUMPTIONS, NAMED

* **The forward cone is a centre column of the image.** There is no measured extrinsic
  between the sonar and the camera -- the sonar's mounting and beam width are not in the
  comm spec. `--cone` sets the column width; it is a stand-in for a calibration nobody has
  done, and it is the first thing to fix if these numbers are ever load-bearing.
* **Frame time comes from `EXT-X-PROGRAM-DATE-TIME` via `showinfo` PTS**, and telemetry
  time from the dashboard's `t`. Both are the robot's clock (see `robot_capture.py`), so
  they are compared directly. The local machine's clock is ~107 s off and is not used.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
SONAR_MIN, SONAR_MAX = 0.28, 4.50  # spec: effective range, saturates at both ends
TOL = 0.05  # +/- m: how close a pixel must be to count as "at" the echo distance


def frame_times(video: Path, stride: int, work: Path) -> list[tuple[Path, float]]:
    """Extract every `stride`-th frame and its media PTS, in seconds.

    `showinfo` is parsed rather than frame numbers multiplied by a nominal fps: the stream
    is a concatenation of live HLS segments whose durations vary (1.0 s and 2.0 s both
    appear), so a nominal rate would drift against the segment clock.
    """
    work.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(video),
            "-vf",
            f"select='not(mod(n\\,{stride}))',showinfo",
            "-vsync",
            "0",
            "-q:v",
            "2",
            str(work / "f%04d.jpg"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    pts = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]
    files = sorted(work.glob("f*.jpg"))
    return list(zip(files, pts, strict=False))


def echo_modes(values: list[float]) -> list[tuple[float, int]]:
    """Split a window of sonar readings into clusters, nearest first.

    A 5 cm gap separates clusters -- wider than the +/-0.02 m spread seen inside one, far
    narrower than the 0.40 m between the two this sensor alternates between. Returns
    (centre, count) so the caller can see a lopsided flip rather than assume a clean 50/50.
    """
    out: list[tuple[float, int]] = []
    run: list[float] = []
    for v in sorted(values):
        if run and v - run[-1] > TOL:
            out.append((float(np.mean(run)), len(run)))
            run = []
        run.append(v)
    if run:
        out.append((float(np.mean(run)), len(run)))
    return out


def locate(depth: np.ndarray, dist: float, cone: float) -> dict:
    """Where in the image sits the surface the sonar could have echoed off?

    Returns the left/centre/right split of the iso-distance shell and whether the forward
    cone contains anything at that range at all. `centre_frac` is the number that decides
    if a residual computed from this frame is comparing like with like.
    """
    _h, w = depth.shape
    shell = (depth >= dist - TOL) & (depth <= dist + TOL)
    c0, c1 = int((0.5 - cone / 2) * w), int((0.5 + cone / 2) * w)
    ys, xs = np.nonzero(shell)
    if len(xs) == 0:
        return {"px": 0, "centre_frac": 0.0, "in_cone": False, "left": 0.0, "right": 0.0}
    left = float((xs < c0).mean())
    right = float((xs >= c1).mean())
    return {
        "px": len(xs),
        "px_frac": round(float(len(xs)) / depth.size, 4),
        "centre_frac": round(1.0 - left - right, 3),
        "left": round(left, 3),
        "right": round(right, 3),
        "in_cone": bool(shell[:, c0:c1].any()),
        "y_med": int(np.median(ys)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True, help="robot_capture.py output dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=60, help="sample every Nth video frame")
    ap.add_argument("--cone", type=float, default=0.20, help="forward cone as image-width frac")
    ap.add_argument("--window", type=float, default=0.5, help="telemetry window per frame (s)")
    args = ap.parse_args()

    import torch
    from transformers import pipeline

    cap: Path = args.capture
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cap / "manifest.json").read_text())
    pdt0 = manifest["pdt_first"]
    tele = [
        json.loads(x) for x in (cap / "telemetry.jsonl").read_text().splitlines() if x.strip()
    ]
    tele = [r for r in tele if "robot" in r]

    frames = frame_times(cap / "raw.mp4", args.stride, args.out / "frames")
    print(f"{len(frames)} frames sampled, {len(tele)} telemetry samples")

    pipe = pipeline(
        "depth-estimation", model=MODEL, device=0 if torch.cuda.is_available() else -1
    )
    pts0 = frames[0][1]
    rows: list[dict] = []

    for path, pts in frames:
        wall = pdt0 + (pts - pts0)
        window = [r for r in tele if abs(r["t"] - wall) <= args.window]
        if not window:
            continue
        fwd = [r["robot"]["ultrasound"][0] for r in window]
        depth = np.asarray(pipe(Image.open(path))["predicted_depth"], dtype=float).squeeze()
        _h, w = depth.shape
        c0, c1 = int((0.5 - args.cone / 2) * w), int((0.5 + args.cone / 2) * w)
        cone_px = depth[:, c0:c1]
        modes = echo_modes(fwd)
        row = {
            "frame": path.name,
            "wall": round(wall, 3),
            "sonar_n": len(fwd),
            "sonar_modes": [{"m": round(m, 3), "n": n} for m, n in modes],
            "sonar_saturated": bool(
                min(fwd) <= SONAR_MIN + 1e-6 or max(fwd) >= SONAR_MAX - 1e-6
            ),
            "cone_min": round(float(cone_px.min()), 3),
            "cone_p05": round(float(np.percentile(cone_px, 5)), 3),
            "cone_med": round(float(np.median(cone_px)), 3),
            "echoes": [{"dist": round(m, 3), **locate(depth, m, args.cone)} for m, _ in modes],
        }
        # The residual the research doc asks for -- but only where the echo is a forward
        # one. Nearest echo, because that is what an obstacle sensor semantically returns.
        near = min(m for m, _ in modes)
        loc = locate(depth, near, args.cone)
        row["comparable"] = loc["in_cone"] and loc["centre_frac"] >= 0.5
        row["residual_m"] = round(float(cone_px.min()) - near, 3) if row["comparable"] else None
        rows.append(row)
        print(
            f"  {path.name} modes={[round(m, 2) for m, _ in modes]} "
            f"cone_min={row['cone_min']:.2f} comparable={row['comparable']}"
        )

    usable = [r for r in rows if r["comparable"]]
    res = [r["residual_m"] for r in usable]
    summary = {
        "capture": str(cap),
        "model": MODEL,
        "cone_frac": args.cone,
        "frames_scored": len(rows),
        "frames_comparable": len(usable),
        "comparable_frac": round(len(usable) / len(rows), 3) if rows else 0.0,
        "residual_mae_m": round(float(np.mean(np.abs(res))), 3) if res else None,
        "residual_bias_m": round(float(np.mean(res)), 3) if res else None,
        # A residual over a handful of frames is not a campaign result. Stated, not hidden.
        "verdict": (
            "insufficient comparable frames -- the sonar echo is lateral, not forward"
            if len(usable) < max(5, 0.2 * len(rows))
            else "residual computed over the comparable subset"
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "frames.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
