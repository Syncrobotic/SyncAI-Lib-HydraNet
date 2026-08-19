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
  0.847 global factor measured on NYUv2 (`eprep_teacher_nyuv2.py`).
* person boxes (`datasets/retail_person_batch01`) pushed through the fitted plane via
  `box_extents`: if the plane is right, standing adults should median ~1.70 m. The ratio
  1.70 / median is a second, independent estimate of the same scale factor.
* the people-only pose (`fit_camera_from_people.fit`, no depth model involved) at the
  same vfov, as a sanity band around the height.
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
sys.path.insert(0, str(HERE))
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.geometry.bev import box_extents  # noqa: E402
from syncai_hydranet.geometry.calibrate import horizon_row, undistort_points  # noqa: E402
from syncai_hydranet.geometry.ground import (  # noqa: E402
    Camera,
    GroundPlane,
    fit_ground_plane,
    pixel_to_ground,
)

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
ADULT_M = 1.70  # same prior as fit_camera_from_people.py, same caveats
NYUV2_SCALE = 0.847  # eprep_teacher_nyuv2.py: global gt/pred median on zero-shot NYUv2
PLATES = Path("datasets/studioa_static")


# ---------------------------------------------------------------------------
# plate handling


def pick_daytime_slot(cam_dir: Path) -> str:
    """Brightest plate among the daytime slots (slot keys are UTC; store-local is +8)."""
    best, best_luma = None, -1.0
    for p in sorted(cam_dir.glob("plate_*.png")):
        slot = p.stem.split("_", 1)[1]
        hour_local = (int(slot[9:11]) + 8) % 24
        if not (8 <= hour_local <= 18):
            continue
        luma = float(np.asarray(Image.open(p).convert("L"), dtype=float).mean())
        if luma > best_luma:
            best, best_luma = slot, luma
    if best is None:
        raise SystemExit(f"no daytime plate under {cam_dir}")
    return best


def undistort_image(img: np.ndarray, k1: float) -> np.ndarray:
    """Resample to the undistorted frame of the division model, same size and centre.

    `undistort_points` maps distorted->undistorted; an image warp needs the inverse, and
    for the one-parameter division model it is closed-form: with radii normalised by the
    half-diagonal (the `geometry/calibrate.py` convention, which is what makes this k1
    the same k1 the tile fit measured), r_u = r_d / (1 + k1 r_d^2) inverts to
    r_d = (1 - sqrt(1 - 4 k1 r_u^2)) / (2 k1 r_u).
    """
    if abs(k1) < 1e-12:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    radius = math.hypot(h, w) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ru = np.hypot(xx - cx, yy - cy) / radius
    disc = 1.0 - 4.0 * k1 * ru**2
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(
            ru > 1e-9, (1.0 - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * k1 * ru**2), 1.0
        )
    scale = np.where(disc > 0, scale, np.nan)
    xs = np.clip((xx - cx) * scale + cx, 0, w - 1)
    ys = np.clip((yy - cy) * scale + cy, 0, h - 1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (xs - x0)[..., None], (ys - y0)[..., None]
    im = img.astype(np.float64)
    if im.ndim == 2:
        im, fx, fy = im[..., None], fx, fy
    out = (
        im[y0, x0] * (1 - fx) * (1 - fy)
        + im[y0, x1] * fx * (1 - fy)
        + im[y1, x0] * (1 - fx) * fy
        + im[y1, x1] * fx * fy
    )
    return out.squeeze().astype(img.dtype)


def run_depth(rgb: np.ndarray) -> np.ndarray:
    """DA-V2 metric depth, resized to the input frame if the pipeline returns raw size."""
    import torch
    from transformers import pipeline

    pipe = pipeline(
        "depth-estimation", model=MODEL, device=0 if torch.cuda.is_available() else -1
    )
    pred = pipe(Image.fromarray(rgb))["predicted_depth"]
    depth = np.asarray(pred, dtype=np.float64).squeeze()
    if depth.shape != rgb.shape[:2]:
        depth = np.asarray(
            Image.fromarray(depth.astype(np.float32), mode="F").resize(
                (rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        )
    return depth


# ---------------------------------------------------------------------------
# floor plane


def floor_candidates(depth: np.ndarray, cam: Camera, inlier_m: float):
    """RANSAC candidates over several seed regions; each with a full-frame inlier count.

    Counts are recomputed over the whole frame so candidates seeded from different bands
    are comparable -- fit_ground_plane's own count is per-band and would make a small
    band's plane look weak for no geometric reason.
    """
    cands = []
    for lower in (0.35, 0.55, 0.75, 1.0):
        for seed in (0, 1, 2):
            plane, residual = fit_ground_plane(
                depth, cam, lower_fraction=lower, iterations=200, inlier_m=inlier_m, seed=seed
            )
            if plane is None:
                continue
            finite = np.isfinite(residual)
            count = int((np.abs(residual[finite]) < inlier_m).sum())
            cands.append((plane, residual, count, lower, seed))
    return cands


def choose_floor(cands, _inlier_m: float = 0.0):
    """Lowest plausible horizontal surface with a competitive inlier count."""
    rows = []
    for plane, _residual, count, lower, seed in cands:
        pitch = math.degrees(plane.pitch)
        roll = math.degrees(plane.roll)
        plausible = (15.0 <= pitch <= 85.0) and abs(roll) <= 20.0
        rows.append(
            {
                "height_m": round(plane.height, 3),
                "pitch_deg": round(pitch, 2),
                "roll_deg": round(roll, 2),
                "inliers_px": count,
                "seed_band": lower,
                "seed": seed,
                "plausible_floor": plausible,
            }
        )
    ok = [c for c, r in zip(cands, rows, strict=True) if r["plausible_floor"]]
    if not ok:
        return None, None, rows
    best_count = max(c[2] for c in ok)
    ok = [c for c in ok if c[2] >= 0.25 * best_count]
    plane, residual, *_ = max(ok, key=lambda c: c[0].height)
    return plane, residual, rows


# ---------------------------------------------------------------------------
# derived numbers


def column_health(cam: Camera, plane: GroundPlane, h: int, w: int) -> dict:
    """The fit_camera_from_people.py acceptance numbers: centre column cast to the floor."""
    v = np.arange(h, dtype=float)
    u = np.full_like(v, w / 2.0)
    _, z = pixel_to_ground(u, v, cam, plane)
    ok = np.isfinite(z) & (z > 0)
    if ok.sum() < 2:
        return {"floor_rows": int(ok.sum())}
    zf = z[ok]
    dz = np.abs(np.diff(zf))
    return {
        "floor_rows": int(ok.sum()),
        "first_floor_row": int(np.nonzero(ok)[0][0]),
        "range_near_m": round(float(zf.min()), 3),
        "range_far_m": round(float(zf.max()), 2),
        "worst_m_per_row": round(float(dz.max()), 3),
        "rows_over_0p3m": int((dz > 0.3).sum()),
    }


def floor_scale(cam: Camera, plane: GroundPlane, h: int, w: int) -> dict:
    """Metres per pixel on the floor at a bottom-centre reference pixel."""
    u0, v0 = w / 2.0, 0.85 * h
    pts = pixel_to_ground(
        np.array([u0, u0 + 1.0, u0]), np.array([v0, v0, v0 - 1.0]), cam, plane
    )
    x, z = pts
    if not np.isfinite(x).all():
        return {"ref_px": [u0, v0], "on_floor": False}
    du = math.hypot(x[1] - x[0], z[1] - z[0])
    dv = math.hypot(x[2] - x[0], z[2] - z[0])
    return {
        "ref_px": [u0, v0],
        "on_floor": True,
        "range_m": round(float(z[0]), 3),
        "m_per_px_u": round(du, 4),
        "m_per_px_v": round(dv, 4),
    }


def load_person_boxes(anns: Path, camera: str, plate_w: int, plate_h: int, k1: float):
    """COCO boxes for this camera, scaled to the plate frame and undistorted with it.

    Same gates as fit_camera_from_people.collect: edge boxes are crops, extreme aspect
    ratios are not standing people.
    """
    d = json.loads(anns.read_text())
    person = next(c["id"] for c in d["categories"] if c["name"] == "person")
    by_img = {im["id"]: im for im in d["images"] if im["file_name"].startswith(camera + "__")}
    keep = []
    for a in d["annotations"]:
        im = by_img.get(a["image_id"])
        if im is None or a["category_id"] != person or a.get("score", 1.0) < 0.5:
            continue
        sx, sy = plate_w / im["width"], plate_h / im["height"]
        x, y, bw, bh = a["bbox"]
        box = np.array([x * sx, y * sy, (x + bw) * sx, (y + bh) * sy])
        if abs(k1) > 1e-12:
            centre, radius = (plate_w / 2.0, plate_h / 2.0), math.hypot(plate_h, plate_w) / 2.0
            corners = undistort_points(
                box.reshape(2, 2), k1, centre, radius
            )  # (x0,y0),(x1,y1): foot and head columns move together enough for a check
            box = corners.reshape(4)
        m = 3.0  # 6 px at 1080 rows, halved with the frame
        if box[0] < m or box[1] < m or box[2] > plate_w - m or box[3] > plate_h - m:
            continue
        bw2, bh2 = box[2] - box[0], box[3] - box[1]
        if bh2 <= 0 or not (1.4 <= bh2 / max(bw2, 1e-6) <= 6.0):
            continue
        keep.append(box)
    return np.asarray(keep, dtype=float).reshape(-1, 4)


def person_checks(boxes: np.ndarray, cam: Camera, plane: GroundPlane, shape, vfov: float):
    out: dict = {"boxes_used": len(boxes)}
    if len(boxes) >= 5:
        hs = box_extents(boxes, cam, plane)[:, 1]
        hs = hs[np.isfinite(hs) & (hs > 0)]
        if len(hs) >= 5:
            med = float(np.median(hs))
            out["implied_person_height_med_m"] = round(med, 3)
            out["implied_person_height_mad_m"] = round(float(np.median(np.abs(hs - med))), 3)
            out["n_heights"] = len(hs)
            out["scale_person"] = round(ADULT_M / med, 4)
    # People-only pose at the same vfov: no depth model involved, so it bounds the height
    # independently. Reused from the script that owns its caveats.
    try:
        import fit_camera_from_people as fcp

        spread, pitch, cam_h, n = fcp.fit(
            boxes, shape, vfov, heights=(1.0, 8.0), pitches=np.arange(5.0, 80.1, 0.5)
        )
        out["people_fit"] = {
            "pitch_deg": round(pitch, 1),
            "height_m": round(cam_h, 2),
            "residual": round(spread, 4),
            "n": n,
        }
    except SystemExit as e:
        out["people_fit"] = {"failed": str(e)}
    except ImportError as e:
        out["people_fit"] = {"failed": f"import: {e}"}
    return out


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
