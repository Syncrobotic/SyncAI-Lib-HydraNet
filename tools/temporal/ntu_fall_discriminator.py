#!/usr/bin/env python3
"""Does head height separate a fall from a bend? NTU's ground truth, in real metres.

    uv run python tools/temporal/ntu_fall_discriminator.py \\
        --zip datasets/_incoming/ntu_rose/nturgbd_skeletons_s001_to_s017.zip \\
        --out runs/ntu_fall01

`analytics/events/pose.py`'s `fall` rule has two conditions. The first -- torso angle from
vertical -- was measured against NTU by `ntu_survey.py` and **fails**: `A43 falling down`
peaks at a 74.5 deg median against `A06 pick up` at 76.3, and 38/40 falls clear the 55 deg
threshold but so do 35/40 pick-ups. The second condition, `fall_head_height_m = 0.80`, was
added 2026-08-26 on the evidence of one 24-minute clip, and that function's own docstring
says of every threshold in it: "none is measured."

This measures it. Same classes, same ground truth, the same `A06 pick up` control -- a
shopper reaching a bottom shelf is the posture that must not alarm -- and real camera-frame
metres rather than a view-normalised redistribution, because a height in metres is exactly
what a normalisation rescales away.

---------------------------------------------------------------------------
WHAT "HEIGHT ABOVE THE FLOOR" MEANS WHEN THE SENSOR IS NOT LEVEL

`height_above_floor_m` in the serving path measures along the **commissioned** vertical.
NTU gives no such thing: the Kinect's +y is up only if that Kinect was level, and setups
vary. Taking camera y would fold each setup's tilt into the number and report it as a
difference between action classes.

So the vertical is estimated per clip, from the subject: the direction from mid-foot to
head over the opening frames, when every one of these actions starts from standing. That
is the same quantity the published redistributions impose globally, computed here per clip
so it can be reported rather than assumed -- `up_tilt_deg` in the output is how far it
sits from the camera's own +y, and if that were near zero the estimate would be doing
nothing.

The floor is the median foot position over the clip, projected onto that vertical. Feet
rest on the floor for most of every one of these actions, including a fall.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syncai_hydranet.data.ntu_skeletons import (  # noqa: E402
    JOINTS,
    members,
    read_clip,
    torso_from_vertical,
)

# `ntu_survey.py`'s set, kept identical so the two reports sit side by side.
CLASSES = {
    43: "A43 falling down",
    42: "A42 staggering",
    6: "A06 pick up",
    8: "A08 sit down",
    27: "A27 jump up",
    1: "A01 drink water",
}
FALL_ANGLE_DEG = 55.0  # events/pose.py default
FALL_HEAD_M = 0.80  # events/pose.py default -- the number this file exists to measure
FEET = ("l_foot", "r_foot", "l_ankle", "r_ankle")
NTU_FPS = 30.0
# `pose.py` applies its 1.0 s `sustained_seconds` to the ANGLE and not to the height, so
# how long the head stays low is the condition nobody has measured on either corpus. A
# fall stays down; a reach for a bottom shelf is a transit.
SUSTAIN_S = (0.0, 0.5, 1.0, 2.0)


def _longest_run(mask: np.ndarray) -> int:
    """Longest consecutive True run. A fall holds the pose; a pick-up passes through it."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def clip_vertical(body: np.ndarray, opening: int = 10) -> np.ndarray:
    """Up, from the subject's own standing pose at the start of the clip."""
    n = min(opening, len(body))
    head = np.nanmedian(body[:n, JOINTS["head"]], axis=0)
    foot = np.nanmedian(np.stack([body[:n, JOINTS[f]] for f in FEET]).reshape(-1, 3), axis=0)
    up = head - foot
    norm = np.linalg.norm(up)
    return up / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])


def head_above_floor(body: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Per frame: the head's height over the clip's own floor, along `up`, in metres."""
    feet = np.stack([body[:, JOINTS[f]] for f in FEET])
    floor = np.nanmedian(feet.reshape(-1, 3), axis=0)
    return (body[:, JOINTS["head"]] - floor) @ up


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--zip", nargs="+", required=True)
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--out")
    a = ap.parse_args()

    rows: dict[str, list[dict]] = {name: [] for name in CLASSES.values()}
    for path in a.zip:
        zf = zipfile.ZipFile(path)
        for action, label in CLASSES.items():
            want = a.per_class - len(rows[label])
            if want <= 0:
                continue
            for member in members(zf, actions={action})[:want]:
                clip = read_clip(zf, member)
                body = clip.joints[:, 0]
                if np.isnan(body).all():
                    continue
                up = clip_vertical(body)
                heights = head_above_floor(body, up)
                angles = torso_from_vertical(body, up=up)
                if not np.isfinite(heights).any():
                    continue
                low = np.asarray(heights < FALL_HEAD_M)
                rows[label].append(
                    {
                        "member": member.rsplit("/", 1)[-1],
                        "seconds_below": float(_longest_run(low) / NTU_FPS),
                        "setup": clip.setup,
                        "camera": clip.camera,
                        "standing_head_m": float(
                            np.nanmedian(heights[: min(10, len(heights))])
                        ),
                        "min_head_m": float(np.nanmin(heights)),
                        "peak_torso_deg": float(np.nanmax(angles)),
                        "up_tilt_deg": float(
                            np.degrees(
                                np.arccos(np.clip(up @ np.array([0.0, 1.0, 0.0]), -1, 1))
                            )
                        ),
                    }
                )

    print(
        f"{'class':22s} {'n':>4s} {'standing head':>14s} {'min head p50':>13s} "
        f"{'<0.80 m':>9s} {'peak torso p50':>15s} {'>55 deg':>9s}"
    )
    report: dict[str, dict] = {}
    for label, rs in rows.items():
        if not rs:
            continue
        mn = np.array([r["min_head_m"] for r in rs])
        st = np.array([r["standing_head_m"] for r in rs])
        pt = np.array([r["peak_torso_deg"] for r in rs])
        report[label] = {
            "n": len(rs),
            "standing_head_m_p50": float(np.median(st)),
            "min_head_m_p50": float(np.median(mn)),
            "min_head_m_p90": float(np.percentile(mn, 90)),
            "frac_below_fall_head": float((mn < FALL_HEAD_M).mean()),
            "peak_torso_deg_p50": float(np.median(pt)),
            "frac_over_fall_angle": float((pt > FALL_ANGLE_DEG).mean()),
            "up_tilt_deg_p50": float(np.median([r["up_tilt_deg"] for r in rs])),
            "seconds_below_p50": float(np.median([r["seconds_below"] for r in rs])),
            "frac_below_for": {
                f"{t:.1f}s": float(
                    np.mean([r["seconds_below"] >= t for r in rs])
                    if t > 0
                    else (mn < FALL_HEAD_M).mean()
                )
                for t in SUSTAIN_S
            },
        }
        r = report[label]
        print(
            f"{label:22s} {r['n']:4d} {r['standing_head_m_p50']:14.2f} "
            f"{r['min_head_m_p50']:13.2f} {r['frac_below_fall_head']:9.2f} "
            f"{r['peak_torso_deg_p50']:15.1f} {r['frac_over_fall_angle']:9.2f}"
        )
    print(
        f"\n{'class':22s} {'sec below p50':>14s} "
        + " ".join(f"{'>=' + f'{t:.1f}s':>8s}" for t in SUSTAIN_S)
    )
    for label, r in report.items():
        print(
            f"{label:22s} {r['seconds_below_p50']:14.2f} "
            + " ".join(f"{r['frac_below_for'][f'{t:.1f}s']:8.2f}" for t in SUSTAIN_S)
        )
    if report:
        tilts = [r["up_tilt_deg_p50"] for r in report.values()]
        print(
            f"\nEstimated vertical sits a median {np.median(tilts):.1f} deg off the camera's "
            f"+y, so the per-clip estimate is doing work rather than reproducing it."
        )
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fall_discriminator.json").write_text(
            json.dumps({"summary": report, "clips": rows}, indent=1) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
