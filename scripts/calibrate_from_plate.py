#!/usr/bin/env python3
"""One-shot camera geometry from a static background plate, cross-checked, not trusted.

    nice -n 10 .venv/bin/python scripts/calibrate_from_plate.py \\
        --camera Taichung-cam01 --slot 20260816-062728 --k1 -0.225 \\
        --vfovs 55,70.4,85 --out runs/calib01

    .venv/bin/python scripts/calibrate_from_plate.py --aggregate runs/calib01

A fixed CCTV camera's geometry is a constant, so it should be a one-time calibration
computed from the temporal-median plate (`scripts/static_plates.py`), not a per-frame
depth head. This script measures whether Depth-Anything V2 Metric-Indoor, run **once** on
the daytime plate, yields a usable floor plane -- camera height, pitch, pixel->metre
scale -- when checked against constraints it did not see.

**What is deliberately NOT fitted.**

* **vfov is an input, never a free parameter.** `fit_camera_from_people.py` documents the
  trap: sweeping vfov 40->100 slid the fitted height 3.53->1.85 m while the residual only
  improved, because a free field of view absorbs lens distortion. Here every requested
  vfov is evaluated and reported side by side; the spread across them IS the result. A
  single number from this script without its vfov row is a choice, not a measurement.
* **Lens distortion is applied first, from the tile-grid fit** (`geometry/calibrate.py`,
  Taichung-cam01: k1 = -0.225, division model, r normalised by the half-diagonal). The
  plate is resampled to the undistorted frame before the depth model sees it, so pinhole
  intrinsics mean what they claim. `--k1 0` measures what skipping this costs.

**Floor selection is "lowest horizontal surface", not "biggest plane".** On a selling
floor the counter top is also a large horizontal plane, and a wall is a large plane full
stop. RANSAC candidates are collected over several seed regions; those whose normal is
not plausibly a floor seen by a down-tilted ceiling camera (pitch outside 15-85 deg,
|roll| > 20 deg) are discarded, and among the survivors with a competitive inlier count
the plane *farthest below the camera* wins -- the floor is by definition beneath every
other horizontal surface. All candidates are recorded so the choice is auditable.

**Cross-checks, in the JSON, per vfov:**

* fitted height/pitch vs an anchor pose where one exists (Taichung-cam01: 2.38 m /
  50.2 deg from the tile grid, `hm3d_render.py` "cctv" preset). The ratio
  anchor_height / fitted_height is the depth model's scale error, comparable to the
  0.847 global factor measured on NYUv2 (see NYUV2_SCALE below for the derivation --
  the script that produced it belonged to the quadruped line and was removed with it).
* person boxes (`datasets/retail_person_batch01`) pushed through the fitted plane via
  `box_extents`: if the plane is right, standing adults should median ~1.70 m. The ratio
  1.70 / median is a second, independent estimate of the same scale factor.
* the people-only pose (`geometry.plate_calibration.fit_pose_from_people`, no depth
  model involved) at the same vfov, as a sanity band around the height.
* `horizon_row` from `geometry/calibrate.py`: a horizon inside the frame rejects the
  pose outright, whatever the residuals say.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_bev3d.calibrate import horizon_row  # noqa: E402

# The pipeline itself lives in the package (moved there 2026-08-19, when
# onboard_camera.py became its second script consumer): the wheel, the type ratchet and
# the coverage floor reach it there, and there is exactly one copy of every formula.
from syncai_bev3d.plate_calibration import (  # noqa: E402
    MODEL,
    PLATES,
    choose_floor,
    column_health,
    floor_candidates,
    floor_scale,
    load_person_boxes,
    person_checks,
    pick_daytime_slot,
    run_depth,
    undistort_image,
)
from syncai_hydranet.geometry.ground import Camera  # noqa: E402

# Zero-shot NYUv2, 654-image official test split, Eigen crop: the global median of
# gt/pred over the whole split. Applying it lifts delta-1 from 0.687 to 0.919 and drops
# AbsRel 0.212 -> 0.101, which is what makes it a scale error rather than a shape error.
#
# The derivation is written out here because the script that produced it
# (`eprep_teacher_nyuv2.py`) was part of the quadruped line and was removed with it on
# 2026-08-19. A constant whose provenance is a dead path is a magic number one commit
# later; re-deriving it needs only NYUv2 and this checkpoint.
NYUV2_SCALE = 0.847


# ---------------------------------------------------------------------------
# visualisation (PIL only: this box has no matplotlib and does not need one)

_TURBO = np.array(
    [(48, 18, 59), (70, 107, 227), (40, 178, 251), (36, 226, 156), (147, 245, 55),
     (232, 197, 26), (250, 112, 26), (200, 35, 15), (122, 4, 3)], dtype=float)  # fmt: skip


def colormap(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1) * (len(_TURBO) - 1)
    i = np.clip(t.astype(int), 0, len(_TURBO) - 2)
    f = (t - i)[..., None]
    return (_TURBO[i] * (1 - f) + _TURBO[i + 1] * f).astype(np.uint8)


def save_depth_png(depth: np.ndarray, path: Path) -> None:
    Image.fromarray(colormap(depth)).save(path)


def save_plane_png(rgb, residual, inlier_m, hrow, path: Path) -> None:
    """Plate dimmed; green where on the fitted plane, red well above it, horizon if inside."""
    im = rgb.astype(float) * 0.45
    on = np.abs(residual) < inlier_m
    above = residual < -4 * inlier_m  # towards the camera: things standing on the floor
    im[on] += np.array([30, 150, 30])
    im[above] += np.array([120, 20, 20])
    im = np.clip(im, 0, 255).astype(np.uint8)
    if 0 <= hrow < im.shape[0]:
        im[int(hrow), :] = (255, 255, 0)
    Image.fromarray(im).save(path)


# ---------------------------------------------------------------------------


def aggregate(out_dir: Path) -> int:
    """Merge every per-camera calib json into report.json with the cross-camera checks."""
    per = {}
    for f in sorted(out_dir.glob("*/calib_*.json")):
        d = json.loads(f.read_text())
        per.setdefault(d["camera"], {})[d["tag"]] = d
    scales = []
    for cam, runs in per.items():
        for tag, d in runs.items():
            for row in d["by_vfov"]:
                for key in ("scale_anchor", "scale_person"):
                    s = row.get(key) or row.get("person", {}).get(key)
                    if s:
                        scales.append(
                            {
                                "camera": cam,
                                "tag": tag,
                                "vfov": row["vfov_deg"],
                                "source": key,
                                "scale": s,
                            }
                        )
    vals = [float(s["scale"]) for s in scales]
    report = {
        "experiment": "per-camera one-time geometry calibration cross-check",
        "model": MODEL,
        "nyuv2_reference_scale": NYUV2_SCALE,
        "cameras": per,
        "scale_factors": scales,
        "scale_spread": {
            "n": len(vals),
            "min": round(min(vals), 3) if vals else None,
            "max": round(max(vals), 3) if vals else None,
            "median": round(float(np.median(vals)), 3) if vals else None,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"report.json written over {len(per)} cameras, {len(scales)} scale estimates")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", help="camera name under datasets/studioa_static")
    ap.add_argument("--slot", help="plate slot key; default: brightest daytime plate")
    ap.add_argument("--plates-root", type=Path, default=PLATES)
    ap.add_argument("--k1", type=float, default=0.0, help="division-model k1; 0 = raw plate")
    ap.add_argument("--vfovs", default="55,70.4,85", help="candidate vfov list, degrees")
    ap.add_argument("--inlier-m", type=float, default=0.05)
    ap.add_argument("--anchor-height", type=float, help="independent height for this camera, m")
    ap.add_argument("--anchor-pitch", type=float, help="independent pitch, deg down")
    ap.add_argument(
        "--person-anns",
        type=Path,
        default=Path("datasets/retail_person_batch01/annotations/instances_all.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("runs/calib01"))
    ap.add_argument(
        "--aggregate", type=Path, help="merge <dir>/*/calib_*.json into report.json"
    )
    args = ap.parse_args(argv)

    if args.aggregate:
        return aggregate(args.aggregate)
    if not args.camera:
        ap.error("--camera is required unless --aggregate")

    cam_dir = args.plates_root / args.camera
    slot = args.slot or pick_daytime_slot(cam_dir)
    plate_path = cam_dir / f"plate_{slot}.png"
    rgb = np.asarray(Image.open(plate_path).convert("RGB"))
    h, w = rgb.shape[:2]
    tag = f"{slot}_k1{args.k1:+.3f}".replace(".", "p")
    out_dir = args.out / args.camera
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_u = undistort_image(rgb, args.k1)
    depth = run_depth(rgb_u)
    save_depth_png(depth, out_dir / f"depth_{tag}.png")
    print(f"{args.camera} {slot}  k1={args.k1:+.3f}  depth "
          f"[{np.nanmin(depth):.2f}, {np.nanmax(depth):.2f}] m")  # fmt: skip

    boxes = None
    if args.person_anns.exists():
        boxes = load_person_boxes(args.person_anns, args.camera, w, h, args.k1)
        print(f"  {len(boxes)} person boxes usable for the cross-check")

    result = {
        "camera": args.camera,
        "slot": slot,
        "tag": tag,
        "plate": str(plate_path),
        "k1": args.k1,
        "model": MODEL,
        "frame_hw": [h, w],
        "inlier_m": args.inlier_m,
        "anchor": (
            {"height_m": args.anchor_height, "pitch_deg": args.anchor_pitch}
            if args.anchor_height
            else None
        ),
        "by_vfov": [],
    }

    for vfov in [float(x) for x in args.vfovs.split(",")]:
        cam = Camera.from_vfov(h, w, vfov)
        cands = floor_candidates(depth, cam, args.inlier_m)
        plane, residual, cand_rows = choose_floor(cands, args.inlier_m)
        row: dict = {"vfov_deg": vfov, "candidates": cand_rows}
        if plane is None:
            row["failed"] = "no plausible floor plane among RANSAC candidates"
            result["by_vfov"].append(row)
            print(f"  vfov {vfov:5.1f}  NO FLOOR")
            continue
        pitch_deg = math.degrees(plane.pitch)
        hrow = horizon_row(pitch_deg, cam.fy, h)
        finite = np.isfinite(residual)
        row.update(
            {
                "height_m": round(plane.height, 3),
                "pitch_deg": round(pitch_deg, 2),
                "roll_deg": round(math.degrees(plane.roll), 2),
                "inlier_frac_frame": round(
                    float((np.abs(residual[finite]) < args.inlier_m).mean()), 4
                ),
                "horizon_row": round(hrow, 1),
                "horizon_inside_frame": bool(0 <= hrow < h),
                "column": column_health(cam, plane, h, w),
                "floor_scale": floor_scale(cam, plane, h, w),
            }
        )
        if args.anchor_height:
            row["scale_anchor"] = round(args.anchor_height / plane.height, 4)
            if args.anchor_pitch is not None:
                row["pitch_err_deg"] = round(pitch_deg - args.anchor_pitch, 2)
        if boxes is not None and len(boxes):
            row["person"] = person_checks(boxes, cam, plane, (h, w), vfov)
        result["by_vfov"].append(row)
        save_plane_png(
            rgb_u, residual, args.inlier_m, hrow, out_dir / f"plane_{tag}_vfov{vfov:g}.png"
        )
        anchor_bit = (
            f"  scale_anchor {row.get('scale_anchor', '-')}" if args.anchor_height else ""
        )
        person_bit = (
            f"  scale_person {row['person'].get('scale_person', '-')}"
            if "person" in row
            else ""
        )
        print(
            f"  vfov {vfov:5.1f}  h {plane.height:5.2f} m  pitch {pitch_deg:5.1f}  "
            f"roll {math.degrees(plane.roll):+5.1f}  horizon row {hrow:7.1f}"
            f"{anchor_bit}{person_bit}"
        )

    out_path = out_dir / f"calib_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
