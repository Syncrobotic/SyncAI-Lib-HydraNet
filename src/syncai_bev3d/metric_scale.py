"""Validate absolute scale from measured floor distances, with held-out segments.

A distance survey needs no room coordinate origin. It changes only metric scale;
camera intrinsics, lens distortion and viewing angles remain fixed. Catalog tile sizes
and assumed device dimensions are not independent measurements.
"""

from dataclasses import replace

import numpy as np

from syncai_bev3d.ground_control import transfer_zones


def refine_metric_scale(cf, controls, *, absolute_tolerance_m=0.05, relative_tolerance=0.03):
    """Return a review candidate; never install an unvalidated calibration."""
    cf.validate()
    if controls.get("camera_id") != cf.camera_id:
        raise ValueError("distance control camera_id mismatch")
    if tuple(controls.get("image_size_px", ())) != cf.image_size_px:
        raise ValueError("distance control image_size_px mismatch")
    if controls.get("pixel_space") != "raw" or controls.get("surface") != "floor":
        raise ValueError("distance controls must use raw pixels on the floor")
    if not str(controls.get("measurement_source", "")).strip():
        raise ValueError("measurement_source must describe an actual physical measurement")
    if (
        not np.isfinite([absolute_tolerance_m, relative_tolerance]).all()
        or min(absolute_tolerance_m, relative_tolerance) <= 0
    ):
        raise ValueError("distance tolerances must be finite and positive")
    segments = controls.get("segments", [])
    if len(segments) < 3:
        raise ValueError("need at least two fitting distances and one held-out distance")
    lengths, measured, validation, vectors = [], [], [], []
    seen = set()
    for segment in segments:
        px = np.asarray(segment["points_px"], float)
        length = float(segment["length_m"])
        held_out = segment.get("validation", False)
        if type(held_out) is not bool:
            raise ValueError("validation must be boolean")
        if px.shape != (2, 2) or not np.isfinite(px).all():
            raise ValueError("each distance needs two finite raw pixel endpoints")
        if ((px < 0) | (px >= cf.image_size_px)).any():
            raise ValueError("distance endpoints lie outside the calibrated frame")
        if not np.isfinite(length) or length <= 0 or np.linalg.norm(px[1] - px[0]) < 8:
            raise ValueError(
                "distance must be positive and endpoints at least eight pixels apart"
            )
        key = tuple(sorted(map(tuple, px.tolist())))
        if key in seen:
            raise ValueError("duplicate segment cannot serve as independent validation")
        seen.add(key)
        world = cf.ground_points(px, above_horizon="raise", what="measured floor distance")
        delta = world[1] - world[0]
        current = float(np.linalg.norm(delta))
        if not np.isfinite(current) or current < 1e-6:
            raise ValueError("distance does not project to a valid floor segment")
        lengths.append(current)
        measured.append(length)
        validation.append(held_out)
        vectors.append(delta / current)
    lengths, measured = np.asarray(lengths), np.asarray(measured)
    validation = np.asarray(validation, bool)
    train = ~validation
    if train.sum() < 2 or validation.sum() < 1:
        raise ValueError("need two fitting distances and an independent validation distance")
    # Only training distances choose the factor. Bad held-out lengths cannot alter it.
    factor = float(np.median(measured[train] / lengths[train]))
    residual = lengths * factor - measured
    tolerance = np.maximum(absolute_tolerance_m, measured * relative_tolerance)
    consistent = np.abs(residual) <= tolerance
    reasons = []
    if not consistent[validation].all():
        reasons.append("held-out distances disagree with the fitted scale")
    if not consistent[train].all():
        reasons.append("fitting distances disagree; check endpoints, units and camera geometry")
    singular = np.linalg.svd(np.asarray(vectors), compute_uv=False)
    if singular[1] < 0.15 * singular[0]:
        reasons.append(
            "all distances have nearly the same direction; include a crosswise segment"
        )
    height = cf.plane.height * factor
    if not 0.5 <= height <= 12:
        reasons.append("scaled camera height falls outside 0.5-12 metres")
    candidate = replace(cf, plane=replace(cf.plane, height=height))
    candidate = transfer_zones(cf, candidate)
    report = {
        "camera_id": cf.camera_id,
        "accepted": not reasons,
        "reasons": reasons,
        "method": "physically measured floor distances; scale only",
        "measurement_source": controls["measurement_source"],
        "factor": factor,
        "height_before_m": cf.plane.height,
        "height_after_m": height,
        "validation": validation.tolist(),
        "before_lengths_m": lengths.tolist(),
        "after_lengths_m": (lengths * factor).tolist(),
        "measured_lengths_m": measured.tolist(),
        "error_m": residual.tolist(),
        "tolerance_m": tolerance.tolist(),
        "next_step": "rebuild depth geometry and scenes before using the candidate camera",
    }
    return candidate, report
