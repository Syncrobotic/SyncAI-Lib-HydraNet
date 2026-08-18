#!/usr/bin/env python3
"""Put an error bar on the depth teacher using public ground truth, since the robot cannot.

    python3 scripts/eprep_teacher_nyuv2.py --data datasets/nyu_depth_v2/extracted \\
        --out runs/eprep/nyuv2

E-prep asks one question -- is the monocular depth teacher metric enough to bootstrap 3D
labels? -- and the first attempt to answer it on the robot came back blocked, not answered:
the ultrasound echo is lateral, so it never measures what the camera looks at
(`docs/RESEARCH_OCCUPANCY.md`, "E-prep, run once"). That measurement now waits on walking
the robot, which waits on the untested teleop.

Public ground truth does not wait on any of that. NYU Depth V2's 654-image official test
split carries dense metric depth from a calibrated Kinect, so the teacher can be scored
against it today, on this machine, with no robot involved.

---------------------------------------------------------------------------
WHAT THIS DOES AND DOES NOT SETTLE

It settles **"is this teacher metric at all"** -- a scale that is 30% out on public indoor
data is not going to be rescued by two sonar returns, and that is worth knowing before
anyone builds a pipeline on top of it.

It does not settle **"is it metric on our camera"**, and the gap is not small. NYUv2 is a
Kinect at roughly head height in furnished rooms. The Lite3's camera sits about 25 cm off
the floor under a wide lens, looking down a corridor -- a viewpoint no public indoor
dataset contains. This repo has been bitten by exactly that before, which is why the
distinction is spelled out rather than assumed away.

**And it cannot see the range the robot lives in.** Kinect returns start around 0.7 m; the
first robot capture put the floor in the forward cone at 0.34 m. So the band that matters
most for a legged robot's foothold and near-obstacle work is *below this dataset's floor*.
Metrics are therefore reported per depth band, not just as one aggregate, and the bands
below 1 m are marked for what they are: thin, and thinner still than the robot needs.

Protocol is the standard one -- Eigen centre crop, valid where GT > 0, capped at 10 m --
so the numbers are comparable to published NYUv2 results rather than to this file alone.

**Both scaled and unscaled numbers are reported, and the gap between them is the point.**
This checkpoint is the Hypersim-trained "Metric Indoor" one, so NYUv2 is a *zero-shot*
domain for it, exactly as the robot's corridor will be. Measured that way it over-predicts
depth by a near-constant factor -- structure intact, scale wrong. Median scaling is the
standard way to separate those two failures: what survives it is the model getting geometry
wrong, what it removes is the model getting metres wrong. For an auto-labeller, the second
is the one a weak anchor was always meant to fix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
MIN_D, MAX_D = 1e-3, 10.0
# Eigen crop: the standard NYUv2 evaluation region, where the Kinect actually returns.
CROP = (45, 471, 41, 601)
BANDS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 10.0)]


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """The published NYUv2 set. `delta1` is the one papers lead with; `abs_rel` is the one
    that tells you whether a *label* built from this depth would be usable."""
    thresh = np.maximum(gt / pred, pred / gt)
    return {
        "n_px": int(gt.size),
        "abs_rel": float(np.mean(np.abs(pred - gt) / gt)),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "log10": float(np.mean(np.abs(np.log10(pred) - np.log10(gt)))),
        "delta1": float(np.mean(thresh < 1.25)),
        "delta2": float(np.mean(thresh < 1.25**2)),
        "delta3": float(np.mean(thresh < 1.25**3)),
        # Absolute error in metres: what a 3D label is actually wrong by. The ratio
        # metrics hide this -- 10% at 5 m is half a metre of voxel in the wrong place.
        "mae_m": float(np.mean(np.abs(pred - gt))),
        "median_err_m": float(np.median(pred - gt)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="dir of NYUv2 .h5 files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N (0 = all)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import pipeline

    files = sorted(args.data.rglob("*.h5"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .h5 under {args.data}")
    args.out.mkdir(parents=True, exist_ok=True)

    pipe = pipeline(
        "depth-estimation", model=MODEL, device=0 if torch.cuda.is_available() else -1
    )
    t, b, l, r = CROP
    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []

    for i, f in enumerate(files):
        with h5py.File(f, "r") as h:
            rgb = np.asarray(h["rgb"][:]).transpose(1, 2, 0)
            gt = np.asarray(h["depth"][:], dtype=np.float64)
        out = pipe(Image.fromarray(rgb))["predicted_depth"]
        pred = np.asarray(out, dtype=np.float64).squeeze()
        gt, pred = gt[t:b, l:r], pred[t:b, l:r]
        ok = (gt > MIN_D) & (gt < MAX_D)
        preds.append(np.clip(pred[ok], MIN_D, MAX_D))
        gts.append(gt[ok])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)}")

    pred_all, gt_all = np.concatenate(preds), np.concatenate(gts)
    # One global factor, the standard median alignment. Not fitted per image: a per-image
    # scale would flatter the model by hiding that the error is a fixed domain offset,
    # which is the finding.
    scale = float(np.median(gt_all / pred_all))
    result = {
        "model": MODEL,
        "dataset": str(args.data),
        "images": len(files),
        "protocol": "Eigen crop, GT in (0, 10] m, no median scaling (model is metric)",
        "overall": metrics(pred_all, gt_all),
        "median_scale_factor": round(scale, 4),
        "overall_median_scaled": metrics(pred_all * scale, gt_all),
        "by_band": {},
    }
    for lo, hi in BANDS:
        m = (gt_all >= lo) & (gt_all < hi)
        band = metrics(pred_all[m], gt_all[m]) if m.sum() > 1000 else {"n_px": int(m.sum())}
        if m.sum() > 1000:
            band["delta1_median_scaled"] = metrics(pred_all[m] * scale, gt_all[m])["delta1"]
            band["abs_rel_median_scaled"] = metrics(pred_all[m] * scale, gt_all[m])["abs_rel"]
        band["gt_px_share"] = round(float(m.mean()), 4)
        result["by_band"][f"{lo:.0f}-{hi:.0f}m"] = band
    # The band the robot actually works in, and the reason it is nearly empty here.
    result["robot_range_note"] = (
        "Lite3's forward cone put floor at 0.34 m; Kinect GT starts near 0.7 m, so the "
        "0-1 m band is thin and the sub-0.7 m range this robot needs is absent entirely."
    )
    (args.out / "nyuv2_teacher.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
