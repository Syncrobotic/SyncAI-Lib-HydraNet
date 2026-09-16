"""Check camera attitude with observed vertical edges, keeping intrinsics and scale fixed.

A vertical image line back-projects to a plane containing gravity. Several separated
lines determine gravity's direction conditional on K and the lens model. They do not
determine focal length, distortion or camera height. Whole lines, not their individual
points, are held out from the fit.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from syncai_bev3d.ground_control import transfer_zones
from syncai_bev3d.opening_controls import source_identity
from syncai_hydranet.geometry.ground import (
    GroundPlane,
    distort_points,
    pixel_to_ground,
    undistort_points,
)


def load_vertical_controls(path: Path, root: Path, camera: str) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema") != 1 or data.get("camera") != camera:
        raise ValueError("vertical controls have unsupported schema or camera")
    if data.get("source_identity") != source_identity(root, camera):
        raise ValueError("stale vertical controls: source, camera or masks changed")
    if not data.get("reviewer") or not data.get("evidence"):
        raise ValueError("vertical controls need reviewer and evidence")
    return data


def _ideal(points, cf):
    if cf.lens is None:
        return np.asarray(points, float)
    lens = cf.lens
    return undistort_points(points, lens.k1, lens.centre_px, lens.radius_px)


def _line(points, cf, attitude):
    down = GroundPlane(cf.plane.height, *attitude).rotation[:, 1]
    vp = np.array(
        [
            cf.camera.fx * down[0] + cf.camera.cx * down[2],
            cf.camera.fy * down[1] + cf.camera.cy * down[2],
            down[2],
        ]
    )
    line = np.cross(np.r_[points.mean(axis=0), 1.0], vp)
    norm = np.linalg.norm(line[:2])
    if norm < 1e-10:
        raise ValueError("vertical edge is degenerate at the vanishing point")
    return line / norm


def _raw_distances(raw, ideal, cf, attitude):
    line = _line(ideal, cf, attitude)
    # Sample the predicted straight ideal line and apply the actual raw lens. A raw
    # curved edge cannot be checked with ideal pixel distances and labelled raw error.
    if abs(line[0]) >= abs(line[1]):
        y = np.linspace(ideal[:, 1].min() - 20, ideal[:, 1].max() + 20, 2001)
        curve = np.c_[-(line[1] * y + line[2]) / line[0], y]
    else:
        x = np.linspace(ideal[:, 0].min() - 20, ideal[:, 0].max() + 20, 2001)
        curve = np.c_[x, -(line[0] * x + line[2]) / line[1]]
    if cf.lens is not None:
        lens = cf.lens
        curve = distort_points(curve, lens.k1, lens.centre_px, lens.radius_px)
    return np.sqrt(((raw[:, None] - curve[None]) ** 2).sum(axis=2).min(axis=1))


def refine_vertical_pose(cf, controls, *, walkable=None, validation_limit_px=4.0):
    """Return an isolated attitude proposal and its limitations; never a deployment gate."""
    if controls.get("camera") != cf.camera_id:
        raise ValueError("vertical controls camera does not match")
    if controls.get("image_size_px") != list(cf.image_size_px):
        raise ValueError("vertical controls require raw camera dimensions")
    if controls.get("pixel_space") != "raw":
        raise ValueError("vertical controls require raw pixel space")
    if not math.isfinite(validation_limit_px) or validation_limit_px <= 0:
        raise ValueError("validation limit must be positive and finite")
    rows = controls.get("lines", [])
    ids, raw, validation = set(), [], []
    for row in rows:
        if not row.get("id") or row["id"] in ids:
            raise ValueError("vertical lines need unique ids")
        ids.add(row["id"])
        points = np.asarray(row["points_px"], float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
            or (points < 0).any()
            or (points >= cf.image_size_px).any()
            or np.linalg.norm(points[-1] - points[0]) < 40
        ):
            raise ValueError("each vertical line needs three in-frame points spanning 40 px")
        if type(row.get("validation")) is not bool:
            raise ValueError("each whole line needs a boolean validation assignment")
        raw.append(points)
        validation.append(row["validation"])
    validation = np.asarray(validation, bool)
    train = ~validation
    if train.sum() < 3 or validation.sum() < 2:
        raise ValueError("need at least three fitting lines and two held-out lines")
    centres = np.array([p.mean(axis=0) for p in raw])
    if np.ptp(centres[train, 0]) < 0.2 * cf.image_size_px[0]:
        raise ValueError("fitting lines must span at least 20 percent of image width")
    ideal = [_ideal(p, cf) for p in raw]

    def residual(attitude):
        return np.concatenate(
            [
                np.c_[p, np.ones(len(p))] @ _line(p, cf, attitude)
                for p, selected in zip(ideal, train, strict=True)
                if selected
            ]
        )

    before = [cf.plane.pitch, cf.plane.roll]
    result = least_squares(
        residual,
        before,
        bounds=([0.0, math.radians(-30)], [math.radians(85), math.radians(30)]),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=500,
    )
    proposal = replace(
        cf, plane=replace(cf.plane, pitch=float(result.x[0]), roll=float(result.x[1]))
    )
    scores = []
    for row, points, corrected in zip(rows, raw, ideal, strict=True):
        first = _raw_distances(points, corrected, cf, before)
        after = _raw_distances(points, corrected, cf, result.x)
        scores.append(
            {
                "id": row["id"],
                "validation": row["validation"],
                "before_raw_px": first.tolist(),
                "after_raw_px": after.tolist(),
                "after_max_raw_px": float(after.max()),
            }
        )
    jac = result.jac / np.maximum(np.linalg.norm(result.jac, axis=0), 1e-12)
    singular = np.linalg.svd(jac, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    reasons = []
    if not result.success or np.any(result.active_mask):
        reasons.append("attitude solver did not converge away from bounds")
    if condition > 1e3:
        reasons.append("vertical directions are poorly constrained")
    if max(r["after_max_raw_px"] for r in scores if r["validation"]) > validation_limit_px:
        reasons.append("held-out vertical edges exceed raw-pixel limit")
    if (
        max(r["after_max_raw_px"] for r in scores if not r["validation"])
        > 2 * validation_limit_px
    ):
        reasons.append("fitting edges remain inconsistent; check annotations or fixed lens")
    zones_transferred = True
    try:
        proposal = transfer_zones(cf, proposal)
    except ValueError as error:
        # Do not return an apparently valid camera with stale metre-space zones.
        proposal = replace(proposal, zones=())
        zones_transferred = False
        reasons.append(f"zone transfer failed: {error}")
    floor = None
    if walkable is not None:
        mask = np.asarray(walkable, bool)
        if mask.shape != cf.image_size_px[::-1] or mask.sum() < 100:
            raise ValueError("walkable check needs a raw mask with 100 pixels")
        v, u = np.nonzero(mask)
        step = max(1, len(v) // 10000)
        px = _ideal(np.c_[u[::step], v[::step]], cf)
        x, z = pixel_to_ground(px[:, 0], px[:, 1], proposal.camera, proposal.plane)
        finite = np.isfinite(x) & np.isfinite(z)
        fraction = float((~finite).mean())
        floor = {
            "sampled_pixels": len(px),
            "above_horizon_fraction": fraction,
            "p95_range_m": float(np.percentile(np.hypot(x[finite], z[finite]), 95))
            if finite.any()
            else None,
            "scope": "same observed floor; inherits unverified focal length and camera height",
        }
        if fraction > 0.01:
            reasons.append("over one percent of observed floor misses the new ground plane")
    report = {
        "camera": cf.camera_id,
        "orientation_checks_passed": not reasons,
        "deployment_ready": False,
        "reasons": reasons,
        "validation_limit_raw_px": validation_limit_px,
        "fit_lines": int(train.sum()),
        "held_out_lines": int(validation.sum()),
        "jacobian_condition": condition,
        "before_deg": np.degrees(before).tolist(),
        "after_deg": np.degrees(result.x).tolist(),
        "line_scores": scores,
        "floor_check": floor,
        "zones_transferred": zones_transferred,
        "scope": (
            "attitude conditional on existing intrinsics/lens; no independent metric scale"
        ),
        "requires": [
            "independent focal/scale checks",
            "rebuild and check depth geometry",
            "reconfirm zones and whole scene",
        ],
    }
    return proposal, report
