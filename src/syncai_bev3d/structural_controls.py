"""Conditional calibration from vertical edges and two orthogonal floor directions.

Line directions constrain focal length, pose and division distortion only under the
fixed principal point, square pixels and Manhattan assumptions. They do not measure
height. Entire lines are withheld, and no old metre-space zone survives a lens change.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import least_squares, minimize_scalar

from syncai_bev3d.render_provenance import sha256
from syncai_bev3d.vertical_controls import load_vertical_controls
from syncai_hydranet.geometry.ground import GroundPlane, distort_points, undistort_points

AXES = ("vertical", "floor_u", "floor_v")


def load_structural_controls(path: Path, root: Path, camera: str) -> dict:
    """Bind both the commissioned image and the separate annotation frame."""
    data = load_vertical_controls(path, root, camera)
    source = data.get("annotation_source", {})
    frame = root / source.get("path", "")
    if not frame.is_file() or sha256(frame) != source.get("sha256"):
        raise ValueError("stale or missing annotation frame")
    with Image.open(frame) as im:
        size = np.array(im.size)
    scale = np.asarray(source.get("scale_to_camera", []), float)
    target = np.asarray(data.get("image_size_px", []), float)
    if (
        source.get("original_size_px") != size.tolist()
        or scale.shape != (2,)
        or not np.isfinite(scale).all()
        or target.shape != (2,)
        or not np.allclose(size * scale, target)
        or scale[0] != scale[1]
    ):
        raise ValueError("annotation frame must have a recorded uniform scale to raw pixels")
    return data


def _line(raw, cf, params, axis):
    log_f, pitch, roll, yaw, k1 = params
    f = math.exp(log_f)
    rotation = GroundPlane(cf.plane.height, pitch, roll).rotation
    directions = {
        "vertical": [0, 1, 0],
        "floor_u": [math.cos(yaw), 0, math.sin(yaw)],
        "floor_v": [-math.sin(yaw), 0, math.cos(yaw)],
    }
    d = rotation @ directions[axis]
    vp = np.array([f * d[0] + cf.camera.cx * d[2], f * d[1] + cf.camera.cy * d[2], d[2]])
    ideal = undistort_points(raw, k1, cf.lens.centre_px, cf.lens.radius_px)
    line = np.cross(np.r_[ideal.mean(axis=0), 1], vp)
    norm = np.linalg.norm(line[:2])
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("degenerate structural line")
    return ideal, line / norm


def _fit_residual(raw, cf, params, axis):
    ideal, line = _line(raw, cf, params, axis)
    # First-order raw-normal distances for fitting; scoring uses the exact curve.
    vec = raw - cf.lens.centre_px
    k = params[4] / cf.lens.radius_px**2
    den = 1 + k * (vec**2).sum(axis=1)
    jac = np.eye(2)[None] / den[:, None, None]
    jac -= 2 * k * vec[:, :, None] * vec[:, None, :] / den[:, None, None] ** 2
    normal_scale = np.linalg.norm(jac @ line[:2], axis=1)
    return (np.c_[ideal, np.ones(len(ideal))] @ line) / normal_scale


def _score(raw, cf, params, axis):
    ideal, line = _line(raw, cf, params, axis)
    return _curve_score(raw, ideal, line, cf, params[4])


def _curve_score(raw, ideal, line, cf, k1):
    """Score a fixed ideal line in native pixels; never optimise camera parameters."""
    origin = -line[2] * line[:2]
    tangent = np.array([-line[1], line[0]])
    errors = []
    for point, corrected in zip(raw, ideal, strict=True):
        t = float((corrected - origin) @ tangent)

        def squared_distance(offset, point=point):
            ideal_point = (origin + offset * tangent)[None]
            predicted = distort_points(ideal_point, k1, cf.lens.centre_px, cf.lens.radius_px)[0]
            return float(np.sum((predicted - point) ** 2))

        nearest = minimize_scalar(squared_distance, bounds=(t - 50, t + 50), method="bounded")
        errors.append(math.sqrt(nearest.fun))
    return errors


def _direction_diagnostic(raw, cf, params, assigned_axis, limit):
    """Post-fit explanation only; no line is removed, reassigned or refitted."""
    ideal, _ = _line(raw, cf, params, assigned_axis)
    centre = ideal.mean(axis=0)
    _, _, vt = np.linalg.svd(ideal - centre, full_matrices=False)
    normal = vt[-1]
    free_line = np.r_[normal, -centre @ normal]
    errors = {}
    unavailable = {}
    for axis in AXES:
        try:
            error = max(_score(raw, cf, params, axis))
            errors[axis] = error if math.isfinite(error) else None
            if errors[axis] is None:
                unavailable[axis] = "non-finite curve distance"
        except ValueError as exc:
            # An alternative VP may coincide with the line centre. Optional
            # explanation must not make an otherwise scoreable fit fail.
            errors[axis] = None
            unavailable[axis] = str(exc)
    return {
        "axis_max_raw_px": errors,
        "axes_within_limit": [
            axis for axis in AXES if errors[axis] is not None and errors[axis] <= limit
        ],
        "unavailable_axes": unavailable,
        # TLS minimises ideal squared error, not the exact maximum raw error.
        # This is a diagnostic, not a certified lower bound or a new gate.
        "tls_curve_max_raw_px": max(_curve_score(raw, ideal, free_line, cf, params[4])),
        "scope": "fixed_fitted_camera; diagnostic_only; does_not_select_or_remove_lines",
    }


def refine_structural_camera(cf, controls, *, validation_limit_px=4.0):
    """Return a diagnostic proposal; even passing line checks cannot commission it."""
    cf.validate()
    if (
        controls.get("schema") != 1
        or controls.get("camera") != cf.camera_id
        or controls.get("image_size_px") != list(cf.image_size_px)
        or controls.get("pixel_space") != "raw"
    ):
        raise ValueError("structural controls require matching camera and raw pixel dimensions")
    if cf.lens is None or not math.isclose(cf.camera.fx, cf.camera.fy):
        raise ValueError("structural fit requires a division lens and square pixels")
    width, height = cf.image_size_px
    if cf.lens.centre_px != (width / 2, height / 2) or not math.isclose(
        cf.lens.radius_px, math.hypot(width, height) / 2
    ):
        raise ValueError("structural fit requires a centred half-diagonal division lens")
    if not math.isfinite(validation_limit_px) or validation_limit_px <= 0:
        raise ValueError("validation limit must be positive and finite")
    rows = controls.get("lines", [])
    seen = set()
    for row in rows:
        points = np.asarray(row["points_px"], float)
        if not row.get("id") or row["id"] in seen:
            raise ValueError("structural lines require unique ids")
        seen.add(row["id"])
        if row.get("axis") not in AXES or type(row.get("validation")) is not bool:
            raise ValueError("lines need an axis and whole-line validation assignment")
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
            or (points < 0).any()
            or (points >= cf.image_size_px).any()
            or np.linalg.norm(points[-1] - points[0]) < 40
        ):
            raise ValueError("structural lines need three in-frame points spanning 40 pixels")
    for axis in AXES:
        fit_count = sum(r["axis"] == axis and not r["validation"] for r in rows)
        held_count = sum(r["axis"] == axis and r["validation"] for r in rows)
        if fit_count < (3 if axis == "vertical" else 2) or held_count < 1:
            raise ValueError(
                "each direction needs fitting lines and an independent held-out line"
            )
    train = [r for r in rows if not r["validation"]]
    weights = np.concatenate(
        [np.full(len(r["points_px"]), 1 / len(r["points_px"])) for r in train]
    )

    def equal_line_loss(z):
        # Average robust point costs within each line, then sum over lines.
        # Weight the loss and its derivatives, not the residual: scaling the
        # residual would also change the physical 2 px robust transition.
        root = np.sqrt(1 + z)
        return np.array([2 * (root - 1), 1 / root, -0.5 / root**3]) * weights

    def residual(params):
        return np.concatenate(
            [_fit_residual(np.asarray(r["points_px"]), cf, params, r["axis"]) for r in train]
        )

    yaw = math.radians(float(controls["initial_floor_axis_deg"]))
    if not math.isfinite(yaw) or not -math.pi / 2 < yaw < math.pi / 2:
        raise ValueError("initial floor axis must be finite and between -90 and 90 degrees")
    bounds = (
        [math.log(0.18 * width), 0.01, -0.6, -math.pi / 2, -0.6],
        [math.log(2 * width), 1.45, 0.6, math.pi / 2, 0.0],
    )
    fits = []
    for focal in (0.4 * width, 0.8 * width, 1.4 * width):
        initial = [math.log(focal), 0.45, -0.1, yaw, -0.2]
        fits.append(
            least_squares(
                residual, initial, bounds=bounds, loss=equal_line_loss, f_scale=2, max_nfev=500
            )
        )
    fit = min(fits, key=lambda f: f.cost)  # Held-out lines never select the solution.
    params = fit.x
    old = np.array([math.log(cf.camera.fx), cf.plane.pitch, cf.plane.roll, yaw, cf.lens.k1])

    # Give the baseline its own best train-only floor azimuth under the original K/lens.
    def baseline_residual(y):
        probe = old.copy()
        probe[3] = y[0]
        return residual(probe)

    old[3] = least_squares(
        baseline_residual,
        [yaw],
        bounds=(-math.pi / 2, math.pi / 2),
        loss=equal_line_loss,
        f_scale=2,
    ).x[0]
    scores = []
    for row in rows:
        raw = np.asarray(row["points_px"])
        before = _score(raw, cf, old, row["axis"])
        after = _score(raw, cf, params, row["axis"])
        scores.append(
            {
                "id": row["id"],
                "axis": row["axis"],
                "validation": row["validation"],
                "before_raw_px": before,
                "after_raw_px": after,
                "before_max_raw_px": max(before),
                "after_max_raw_px": max(after),
                "direction_diagnostic": _direction_diagnostic(
                    raw, cf, params, row["axis"], validation_limit_px
                ),
            }
        )
    singular = np.linalg.svd(
        fit.jac / np.maximum(np.linalg.norm(fit.jac, axis=0), 1e-12), compute_uv=False
    )
    condition = float(singular[0] / max(singular[-1], 1e-12))
    reasons = []
    if not fit.success or np.any(fit.active_mask):
        reasons.append("solver failed or reached parameter bounds")
    if condition > 1000:
        reasons.append("structural directions do not constrain all fitted parameters")
    if not all(np.isfinite(r["after_raw_px"]).all() for r in scores):
        reasons.append("projection contains non-finite raw curve distances")
    for held, label in ((True, "held-out"), (False, "fitting")):
        if (
            max(r["after_max_raw_px"] for r in scores if r["validation"] == held)
            > validation_limit_px
        ):
            reasons.append(f"{label} structural lines exceed the raw-pixel limit")
    proposal = replace(
        cf,
        camera=replace(cf.camera, fx=math.exp(params[0]), fy=math.exp(params[0])),
        plane=replace(cf.plane, pitch=float(params[1]), roll=float(params[2])),
        lens=replace(cf.lens, k1=float(params[4])),
        zones=(),
    )
    report = {
        "schema": 1,
        "camera": cf.camera_id,
        "line_checks_passed": not reasons,
        "deployment_ready": False,
        "independent_metric_accuracy": False,
        "reasons": reasons,
        "validation_limit_px": validation_limit_px,
        "floor_axis_deg": math.degrees(params[3]),
        "condition": condition,
        "line_scores": scores,
        "fit_weighting": "sum_of_per_line_mean_soft_l1; f_scale=2_raw_px",
        "parameters": params.tolist(),
        "height_m_unchanged": cf.plane.height,
        "zones_removed": len(cf.zones),
        "requires_depth_rebuild": True,
        "scope": (
            "Conditional on fixed principal point, square pixels, radial lens and "
            "orthogonal structure; height remains unmeasured. Re-derive and review zones."
        ),
    }
    return proposal, report
