"""One-shot camera geometry from a static background plate: the shared pipeline.

Extracted from ``scripts/calibrate_from_plate.py`` on 2026-08-19, when
``scripts/onboard_camera.py`` became its second script consumer and tripped
``tests/test_scripts_are_not_libraries.py`` -- shared code belongs where the wheel, the
type ratchet and the coverage floor reach it, not behind a ``sys.path`` insert. The CLI
keeps the experiment's full rationale (what is deliberately not fitted, the vfov trap,
the cross-check design); this module holds the arithmetic both callers must agree on.

The pipeline, in order:

1. **Undistort the plate** with the one-parameter division model
   (:func:`undistort_image`), using the same half-diagonal-normalised ``k1`` the
   tile-grid fit in ``geometry/calibrate.py`` measures, so the two stay one number.
2. **Depth once** (:func:`run_depth`, Depth-Anything V2 Metric-Indoor) on the
   undistorted plate -- a fixed camera's geometry is a constant, not a per-frame head.
3. **Floor selection is "lowest plausible horizontal surface", not "biggest plane"**
   (:func:`floor_candidates` / :func:`choose_floor`): RANSAC candidates over several
   seed regions, implausible normals discarded, and among competitive survivors the
   plane farthest below the camera wins -- the floor is by definition beneath every
   other horizontal surface.
4. **Derived checks** a caller can gate on: :func:`column_health` (the
   ``fit_camera_from_people`` acceptance numbers), :func:`floor_scale` (metres per pixel
   at a reference), and :func:`person_checks` (person boxes pushed through the fitted
   plane, plus the depth-free people-only pose as an independent bound).

:func:`fit_pose_from_people` moved here with :func:`person_checks`, from
``scripts/fit_camera_from_people.py``: the package cannot import a script, and the
1.70 m prior (:data:`ADULT_M`) must exist exactly once. That script's docstring records
the method's measured limitation -- with a free vfov every pinhole parameter absorbs
lens distortion -- which is why the vfov is an input everywhere in this module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from syncai_hydranet.geometry.ground import (
    Camera,
    GroundPlane,
    fit_ground_plane,
    pixel_to_ground,
    undistort_points,
)

from .bev import box_extents

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
ADULT_M = 1.70  # the standing-adult prior; pose bias makes it a ±11% systematic term
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


# ---------------------------------------------------------------------------
# person cross-checks


def load_person_boxes(anns: Path, camera: str, plate_w: int, plate_h: int, k1: float):
    """COCO boxes for this camera, scaled to the plate frame and undistorted with it.

    Same gates as the video-based collector in fit_camera_from_people.py: edge boxes are
    crops, extreme aspect ratios are not standing people.
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


def fit_pose_from_people(boxes: np.ndarray, shape, vfov: float, heights, pitches):
    """(spread, pitch_deg, height_m, n) fitted from person boxes alone -- no depth model.

    Moved verbatim from ``scripts/fit_camera_from_people.py``, whose docstring records
    the method's measured limitation: it recovers a *family* of poses, and the vfov must
    be an input because a free one absorbs lens distortion. Here it serves as the
    depth-free sanity bound in :func:`person_checks`.

    Raises ``SystemExit`` (not ``ValueError``) when no pitch fits, preserved from the
    script so its CLI exit behaviour and :func:`person_checks`'s ``failed`` branch both
    stay byte-identical.
    """
    cam = Camera.from_vfov(shape[0], shape[1], vfov)
    best = None
    for pitch in pitches:
        plane_unit = GroundPlane(height=1.0, pitch=np.radians(pitch))
        ext = box_extents(boxes, cam, plane_unit)
        hs = ext[:, 1]
        hs = hs[np.isfinite(hs) & (hs > 0)]
        if len(hs) < 8:
            continue
        # Heights scale linearly with camera height, so the *relative* spread is a
        # function of pitch alone -- which is what makes this a one-dimensional search
        # per pitch instead of a two-dimensional one.
        med = float(np.median(hs))
        spread = float(np.median(np.abs(hs - med))) / max(med, 1e-9)
        cam_h = ADULT_M / med  # the height that puts the median person at 1.70 m
        if not (heights[0] <= cam_h <= heights[-1]):
            continue
        if best is None or spread < best[0]:
            best = (spread, float(pitch), float(cam_h), len(hs))
    if best is None:
        raise SystemExit("no pitch produced a usable fit within the allowed height range")
    return best


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
    # independently.
    try:
        spread, pitch, cam_h, n = fit_pose_from_people(
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
    return out
