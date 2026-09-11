"""Refine commissioning geometry against surveyed floor points, with held-out checks.

Survey coordinates may have any origin and rotation; the fitted rigid transform is
reported separately. Runtime still consumes Camera + GroundPlane in its existing frame.
The lens stays fixed: fitting distortion and focal length together from sparse floor
clicks can trade one error for another. No teacher or person-height prior is used here.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import (
    GroundPlane,
    distort_points,
    ground_to_pixel,
    undistort_points,
)


def floor_pixels(cf: CameraFile, points_m: np.ndarray) -> np.ndarray:
    """Camera-local floor metres to raw pixels; invisible points retain NaN rows."""
    pts = np.asarray(points_m, dtype=float)
    u, v, depth = ground_to_pixel(pts[:, 0], pts[:, 1], cf.camera, cf.plane)
    out = np.column_stack((u, v))
    if cf.lens is not None:
        lens = cf.lens
        out = distort_points(out, lens.k1, lens.centre_px, lens.radius_px)
    out[depth <= 0] = np.nan
    return out


def transfer_zones(existing: CameraFile, fresh: CameraFile) -> CameraFile:
    """Keep the raw-image location of every zone when the ground model changes.

    This preserves observed boundaries, not an independent survey. Reconfirm policy
    zones after refinement and regenerate depth-derived meshes from the new calibration.
    """
    if existing.camera_id != fresh.camera_id:
        raise ValueError("cannot transfer zones between different cameras")
    if existing.image_size_px != fresh.image_size_px or existing.lens != fresh.lens:
        raise ValueError("image size or lens changed; rederive pixel artefacts first")
    zones = []
    for zone in existing.zones:
        px = floor_pixels(existing, np.asarray(zone.points_m))
        if not np.isfinite(px).all():
            raise ValueError(f"zone {zone.name}: old boundary cannot be projected")
        points = fresh.ground_points(px, above_horizon="raise", what=zone.name)
        zones.append(replace(zone, points_m=tuple(map(tuple, points.tolist()))))
    return replace(fresh, zones=tuple(zones))


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _points(values, name: str, minimum: int) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f"{name} must be N x 2, with at least {minimum} points")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"{name} contains duplicate points")
    return points


def refine_ground_control(
    cf: CameraFile,
    controls: dict,
    *,
    fit_focal: bool = False,
    pixel_threshold: float = 4.0,
    validation_threshold_m: float = 0.15,
) -> tuple[CameraFile, dict]:
    """Fit pose and survey alignment; reserve validation points entirely for scoring.

    At least six fitting points are needed (eight when fitting focal length), spread
    over a two-dimensional patch. A candidate is accepted only with two or more held-out
    points, adequate numerical rank, no bound hits, and errors below explicit limits.
    Limits are commissioning policy, not a claim of accuracy on an unmeasured store.
    """
    from scipy.optimize import least_squares

    cf.validate()
    if controls.get("camera_id") != cf.camera_id:
        raise ValueError("control camera_id does not match camera.json")
    if tuple(controls.get("image_size_px", ())) != cf.image_size_px:
        raise ValueError("control image_size_px must match the raw calibrated frame")
    if controls.get("pixel_space") != "raw":
        raise ValueError("control pixel_space must be 'raw', before lens undistortion")
    if not str(controls.get("measurement_source", "")).strip():
        raise ValueError("measurement_source must describe the physical survey")
    if not all(math.isfinite(x) and x > 0 for x in (pixel_threshold, validation_threshold_m)):
        raise ValueError("error thresholds must be positive and finite")
    minimum = 8 if fit_focal else 6
    px = _points(controls["points_px"], "points_px", minimum)
    survey = _points(controls["points_m"], "points_m", minimum)
    if px.shape != survey.shape:
        raise ValueError("points_px and points_m must have the same shape")
    w, h = cf.image_size_px
    if ((px < 0) | (px >= [w, h])).any():
        raise ValueError("control pixels fall outside image_size_px")
    validation = np.asarray(controls.get("validation", [False] * len(px)))
    if validation.shape != (len(px),) or validation.dtype != np.bool_:
        raise ValueError("validation must contain one boolean per point")
    train = ~validation
    if train.sum() < minimum:
        raise ValueError(f"need at least {minimum} fitting points, excluding validation")
    for points, name in ((px[train], "pixels"), (survey[train], "survey")):
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if singular[1] < 0.05 * singular[0]:
            raise ValueError(f"{name} fitting points are collinear or too narrowly spread")

    ideal = px.copy()
    if cf.lens is not None:
        lens = cf.lens
        ideal = undistort_points(px, lens.k1, lens.centre_px, lens.radius_px)
    # A rigid alignment of the initial floor projections gives an origin/yaw seed.
    initial_ground = cf.ground_points(px[train], above_horizon="raise", what="controls")
    a = survey[train] - survey[train].mean(axis=0)
    b = initial_ground - initial_ground.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.diag([1.0, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    translation = initial_ground.mean(axis=0) - survey[train].mean(axis=0) @ rotation.T
    x0 = np.array([cf.plane.height, cf.plane.pitch, cf.plane.roll, yaw, *translation])
    lower = [0.5, math.radians(5), math.radians(-30), -np.inf, -np.inf, -np.inf]
    upper = [12.0, math.radians(85), math.radians(30), np.inf, np.inf, np.inf]
    if fit_focal:
        x0 = np.append(x0, 0.0)
        lower.append(math.log(0.5))
        upper.append(math.log(2.0))

    def unpack(params):
        factor = math.exp(params[6]) if fit_focal else 1.0
        cam = replace(cf.camera, fx=cf.camera.fx * factor, fy=cf.camera.fy * factor)
        candidate = replace(cf, camera=cam, plane=GroundPlane(*params[:3]))
        local = survey @ _rotation(params[3]).T + params[4:6]
        return candidate, local

    def residual(params, selection):
        candidate, local = unpack(params)
        level = np.column_stack((local[:, 0], np.full(len(local), params[0]), local[:, 1]))
        camera_pts = level @ candidate.plane.rotation.T
        # Finite trial residuals allow the solver to leave a behind-camera trial.
        depth = np.maximum(camera_pts[:, 2], 1e-3)
        pred = camera_pts[:, :2] / depth[:, None]
        pred = pred * [candidate.camera.fx, candidate.camera.fy]
        pred += [candidate.camera.cx, candidate.camera.cy]
        return (pred[selection] - ideal[selection]).ravel()

    def solve(start, selection):
        return least_squares(
            residual,
            start,
            args=(selection,),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=pixel_threshold / 2,
            x_scale="jac",
            max_nfev=2000,
        )

    result = solve(x0, train)
    candidate, local = unpack(result.x)
    errors = np.linalg.norm(floor_pixels(candidate, local) - px, axis=1)
    inliers = train & np.isfinite(errors) & (errors <= pixel_threshold)
    if inliers.sum() >= minimum and not np.array_equal(inliers, train):
        result = solve(result.x, inliers)
        candidate, local = unpack(result.x)
        errors = np.linalg.norm(floor_pixels(candidate, local) - px, axis=1)
        inliers = train & np.isfinite(errors) & (errors <= pixel_threshold)

    # Column-normalisation makes the condition estimate independent of m/rad units.
    jac = result.jac
    norms = np.linalg.norm(jac, axis=0)
    singular = np.linalg.svd(jac / np.maximum(norms, 1e-15), compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-15))
    reasons = []
    if not result.success:
        reasons.append("optimizer did not converge")
    if np.any(result.active_mask) or any(
        min(value - lo, hi - value) < 1e-4
        for value, lo, hi in zip(result.x, lower, upper, strict=True)
    ):
        reasons.append("fit reached a parameter bound")
    if condition > 1e4:
        reasons.append("fit is poorly constrained; spread the surveyed points further")
    if inliers.sum() < minimum or inliers.sum() < 0.75 * train.sum():
        reasons.append("too few consistent fitting points")
    if validation.sum() < 2:
        reasons.append("need at least two independent validation points")

    def score(model, local_points):
        reproj = np.linalg.norm(floor_pixels(model, local_points) - px, axis=1)
        # Keep correspondence indices even for rays that miss the ground.
        ground = np.full_like(local_points, np.nan)
        for i, point in enumerate(px):
            value = model.ground_points(point[None])
            if len(value):
                ground[i] = value[0]
        metric = np.linalg.norm(ground - local_points, axis=1)
        return reproj, metric

    # Baseline gets its own rigid alignment, but no validation point sets that alignment.
    before_local = survey @ rotation.T + translation
    before_px, before_m = score(cf, before_local)
    after_px, after_m = score(candidate, local)
    if validation.any() and (
        not np.isfinite(after_m[validation]).all()
        or np.max(after_m[validation]) > validation_threshold_m
        or not np.isfinite(after_px[validation]).all()
        or np.max(after_px[validation]) > pixel_threshold
    ):
        reasons.append("held-out error exceeds the configured limits")
    if not np.isfinite(after_m[train]).all():
        reasons.append("a fitting ray misses the floor")

    def finite_list(values):
        return [float(x) if np.isfinite(x) else None for x in values]

    report = {
        "camera_id": cf.camera_id,
        "measurement_source": controls["measurement_source"],
        "accepted": not reasons,
        "reasons": reasons,
        "fit_focal": fit_focal,
        "lens_fitted": False,
        "fit_count": int(train.sum()),
        "inlier_count": int(inliers.sum()),
        "validation_count": int(validation.sum()),
        "pixel_threshold": pixel_threshold,
        "validation_threshold_m": validation_threshold_m,
        "jacobian_condition": condition,
        "survey_to_camera_floor": {
            "yaw_rad": float(result.x[3]),
            "translation_m": result.x[4:6].tolist(),
        },
        "validation": validation.tolist(),
        "inliers": inliers.tolist(),
        "before": {"error_px": finite_list(before_px), "error_m": finite_list(before_m)},
        "after": {"error_px": finite_list(after_px), "error_m": finite_list(after_m)},
        "scope": "floor geometry only; not object-height or unseen-room validation",
        "requires": ["reconfirm zones", "rebuild depth geometry caches and scene meshes"],
    }
    candidate = transfer_zones(cf, candidate)
    candidate.validate()
    return candidate, report
