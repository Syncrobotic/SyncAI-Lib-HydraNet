"""Conditional Manhattan directions from automatically observed, grouped raw edges.

Grouping is conservative geometric equivalence, not verified object identity. The
partition is frozen before lens/direction selection. Floor semantics orient the
otherwise unordered axes; all geometry remains conditional on those semantics,
centred square-pixel intrinsics and a one-parameter division lens.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from syncai_bev3d.structural_controls import AXES, refine_structural_camera
from syncai_hydranet.geometry.camera_json import CameraFile, Lens
from syncai_hydranet.geometry.ground import Camera, GroundPlane, undistort_points

LENS_GRID = tuple(np.linspace(-0.6, 0, 13).tolist())


def add_direction_checks(proposals: dict, groups_path: Path):
    """Persist the partition before fitting, then propagate the actual capability state."""
    groups = freeze_line_groups(proposals)
    with groups_path.open("x") as handle:
        json.dump(groups, handle, indent=2, allow_nan=False)
        handle.write("\n")
    directions = infer_directions(groups)
    report = deepcopy(proposals)
    report["directions"] = directions
    report["groups_file"] = str(groups_path.resolve())
    report["status"] = (
        "partial" if directions["direction_checks_passed"] else directions["status"]
    )
    report["reasons"] = directions["reasons"] + [
        "semantic_predictions_not_geometric_truth",
        "no_metric_scale_estimated",
    ]
    return report


def _samples(points):
    raw = np.asarray(points, dtype=float)
    return raw[np.unique(np.linspace(0, len(raw) - 1, min(16, len(raw)), dtype=int))]


def _ideal(raw, size, k1):
    centre = np.asarray(size) / 2
    radius = float(np.linalg.norm(centre))
    return (undistort_points(raw, k1, centre, radius) - centre) / radius


def _tls(points):
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    normal = vt[-1]
    return np.r_[normal, -centre @ normal]


def _coherent(raw, size, tolerance):
    radius = float(np.linalg.norm(np.asarray(size) / 2))
    for k1 in LENS_GRID:
        points = _ideal(raw, size, k1)
        line = _tls(points)
        if np.max(np.abs(np.c_[points, np.ones(len(points))] @ line)) * radius <= tolerance:
            return True
    return False


def freeze_line_groups(proposals: dict):
    """Merge possibly identical persistent traces before any direction fit or split.

    A union under ANY allowed lens keeps fragments on the same side of holdout.
    Non-complete unions are retained in the manifest but excluded from fitting.
    The result is still a geometric hypothesis, not a proof of physical identity.
    """
    if (
        proposals.get("schema") != "structural-edge-proposals-v1"
        or proposals.get("pixel_space") != "raw"
    ):
        raise ValueError("expected raw structural proposals")
    size = proposals["image_size_px"]
    observations = {row["id"]: row for row in proposals["observations"]}
    if len(observations) != len(proposals["observations"]):
        raise ValueError("duplicate observation ids")
    radius = float(np.linalg.norm(np.asarray(size) / 2))
    tolerance = radius * 0.005
    frame_count = proposals.get("summary", {}).get(
        "selected_frames", len({row["frame_id"] for row in observations.values()})
    )
    minimum_frames = max(
        2, math.ceil(frame_count * proposals.get("config", {}).get("min_persistence", 0.5))
    )
    excluded = []
    rows: list[dict] = []
    for track in proposals["tracks"]:
        if track["state"] not in {"persistent", "ambiguous_match"}:
            continue
        members = [observations[key] for key in track["observation_ids"]]
        frames = {row["frame_id"] for row in members}
        if len(frames) < minimum_frames:
            excluded.append({"track_id": track["id"], "reason": "insufficient_distinct_frames"})
            continue
        # Selection uses native span only, never a calibration residual.
        representative = max(members, key=lambda row: (row["span_px"], row["id"]))
        for member in members:
            points = np.asarray(member["points_px"], dtype=float)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or len(points) < 3
                or not np.isfinite(points).all()
                or (points < 0).any()
                or (points >= size).any()
            ):
                raise ValueError("invalid raw observation points")
        support = np.concatenate([_samples(member["points_px"]) for member in members])
        if not _coherent(support, size, tolerance):
            excluded.append({"track_id": track["id"], "reason": "inconsistent_division_curve"})
            continue
        floor_frames = {
            row["frame_id"] for row in members if row.get("floor_fraction", 0) >= 0.8
        }
        rows.append(
            {
                "track_id": track["id"],
                "observation_ids": track["observation_ids"],
                "points_px": representative["points_px"],
                "floor_hint": len(floor_frames) >= max(2, math.ceil(len(frames) * 2 / 3)),
                "support": support,
                "resolved_ambiguity": track["state"] == "ambiguous_match",
            }
        )
    parents = list(range(len(rows)))
    compatible = np.eye(len(rows), dtype=bool)
    corrected = [
        [_ideal(_samples(row["points_px"]), size, k) for row in rows] for k in LENS_GRID
    ]

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, j in itertools.combinations(range(len(rows)), 2):
        for frame in corrected:
            joint = np.concatenate([frame[i], frame[j]])
            line = _tls(joint)
            error = np.max(np.abs(np.c_[joint, np.ones(len(joint))] @ line)) * radius
            if error <= tolerance:
                compatible[i, j] = compatible[j, i] = True
                parents[root(i)] = root(j)
                break
    components: dict[int, list[int]] = {}
    for i in range(len(rows)):
        components.setdefault(root(i), []).append(i)
    groups = []
    for indices in components.values():
        tracks = sorted(rows[i]["track_id"] for i in indices)
        digest = hashlib.sha256(
            ("structural-line-group-v1:" + ":".join(tracks)).encode()
        ).hexdigest()
        representative = max(
            indices,
            key=lambda i: np.linalg.norm(
                np.diff(np.asarray(rows[i]["points_px"])[[0, -1]], axis=0)
            ),
        )
        coherent = bool(compatible[np.ix_(indices, indices)].all()) and _coherent(
            np.concatenate([rows[i]["support"] for i in indices]), size, tolerance
        )
        groups.append(
            {
                "id": "line-" + digest[:12],
                "track_ids": tracks,
                "observation_ids": sorted(
                    set(
                        itertools.chain.from_iterable(
                            rows[i]["observation_ids"] for i in indices
                        )
                    )
                ),
                "points_px": rows[representative]["points_px"],
                "floor_hint": all(rows[i]["floor_hint"] for i in indices),
                "state": "geometric_candidate" if coherent else "ambiguous_union",
                "validation": int(digest, 16) % 4 == 0,
                "physical_identity_verified": False,
                "resolved_ambiguous_tracks": [
                    rows[i]["track_id"] for i in indices if rows[i]["resolved_ambiguity"]
                ],
            }
        )
    result = {
        "schema": "structural-line-groups-v1",
        "camera": proposals["camera"],
        "proposals_sha256": hashlib.sha256(
            json.dumps(proposals, sort_keys=True).encode()
        ).hexdigest(),
        "image_size_px": size,
        "pixel_space": "raw",
        "lens_grid": list(LENS_GRID),
        "groups": groups,
        "excluded_tracks": excluded,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "partition_policy": "sha256-track-component-mod4-v1",
    }
    result["freeze_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()
    return result


def _vp_errors(points, vp, radius):
    centres = points.mean(axis=1)
    lines = np.cross(np.c_[centres, np.ones(len(centres))], vp)
    norms = np.linalg.norm(lines[:, :2], axis=1)
    values = np.einsum(
        "nij,nj->ni", np.concatenate([points, np.ones((*points.shape[:2], 1))], axis=2), lines
    )
    errors = np.max(np.abs(values), axis=1) * radius / np.maximum(norms, 1e-12)
    errors[norms < 1e-10] = np.inf
    return errors


def _floor_vps(points, floor, radius, limit):
    lines = np.array([_tls(p) for p in points])
    candidates = []
    seen = set()
    for i, j in itertools.combinations(np.flatnonzero(floor), 2):
        vp = np.cross(lines[i], lines[j])
        if np.linalg.norm(vp) < 1e-9:
            continue
        vp /= np.linalg.norm(vp)
        members = floor & (_vp_errors(points, vp, radius) <= limit)
        if members.sum() < 2:
            continue
        key = tuple(np.flatnonzero(members).tolist())
        if key in seen:
            continue
        seen.add(key)
        _, _, vt = np.linalg.svd(lines[members], full_matrices=True)
        vp = vt[-1]
        errors = _vp_errors(points, vp, radius)
        members = floor & (errors <= limit)
        if members.sum() >= 2:
            candidates.append(
                (int(members.sum()), float(np.mean(errors[members])), vp, members)
            )
    candidates.sort(key=lambda entry: (-entry[0], entry[1]))
    return candidates[:12]


def _axis_support(errors, floor_hint, limit):
    """Distinguish nearest-axis guesses from uniquely supported directions."""
    compatible = [AXES[i] for i, error in enumerate(errors) if error <= limit]
    if not compatible:
        status = "unsupported"
    elif len(compatible) > 1:
        status = "ambiguous"
    elif floor_hint and compatible == ["vertical"]:
        status = "semantic_conflict"
    else:
        status = "supported"
    return {
        "status": status,
        "compatible_axes": compatible,
        "resolved_axis": compatible[0] if status == "supported" else None,
        "errors_px": {
            axis: float(error) if math.isfinite(error) else None
            for axis, error in zip(AXES, errors, strict=True)
        },
    }


def infer_directions(frozen: dict):
    """Select lens/axes on fitting groups ONLY, then test every held-out group.

    The unit-height CameraFile exists only to reuse the existing direction solver;
    it is never saved, returned or interpreted as a metric measurement.
    """
    expected = dict(frozen)
    digest = expected.pop("freeze_sha256", None)
    if digest != hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest():
        raise ValueError("line groups changed after partition freeze")
    size = frozen["image_size_px"]
    radius = float(np.linalg.norm(np.asarray(size) / 2))
    limit = radius / math.hypot(480, 270) * 4
    rows = [g for g in frozen["groups"] if g["state"] == "geometric_candidate"]
    train = [g for g in rows if not g["validation"]]
    result = {
        "schema": "structural-directions-v1",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "groups_sha256": digest,
        "status": "insufficient_evidence",
        "direction_checks_passed": False,
        "world_direction_assignments_complete": False,
        "axis_label_policy": "axis_is_nominal; use_axis_support_for_unique_direction",
        "semantic_hypotheses_rejected": 0,
        "deployment_ready": False,
        "scale_status": "unknown",
        "groups": [],
        "assumptions": [
            "predicted_floor_is_horizontal",
            "Manhattan_structure",
            "centred_square_pixel_intrinsics",
            "division_lens",
        ],
        "reasons": [],
    }
    if sum(g["floor_hint"] for g in train) < 4:
        result["reasons"] = ["fewer_than_four_fitting_floor_groups"]
        return result
    floor = np.array([g["floor_hint"] for g in train], dtype=bool)
    # Equal samples per physical-line candidate: repeated frames/longer traces
    # cannot multiply a line's vote. Interpolation here is for hypothesis scoring
    # only; the existing solver below receives the original measured pixels.
    raw = []
    for row in train:
        pts = np.asarray(row["points_px"], dtype=float)
        indices = np.linspace(0, len(pts) - 1, 16)
        raw.append(
            np.column_stack(
                [np.interp(indices, np.arange(len(pts)), pts[:, k]) for k in (0, 1)]
            )
        )
    best = None
    for k1 in LENS_GRID:
        points = np.array([_ideal(p, size, k1) for p in raw])
        candidates = _floor_vps(points, floor, radius, limit)
        for first, second in itertools.combinations(candidates, 2):
            vp1, vp2 = first[2], second[2]
            if (first[3] & second[3]).any() or abs(vp1[2] * vp2[2]) < 1e-9:
                continue
            f2 = -float(vp1[:2] @ vp2[:2]) / (vp1[2] * vp2[2])
            if not (0.18 * size[0] / radius) ** 2 < f2 < (2 * size[0] / radius) ** 2:
                continue
            f = math.sqrt(f2)
            ray1, ray2 = vp1 / [f, f, 1], vp2 / [f, f, 1]
            normal = np.cross(ray1, ray2)
            normal /= np.linalg.norm(normal)
            if normal[2] < 0:
                normal = -normal
            pitch = math.asin(float(normal[2]))
            roll = math.atan2(-normal[0], normal[1])
            if not 0.01 < pitch < 1.45 or not -0.6 < roll < 0.6:
                continue
            vps = np.array([normal * [f, f, 1], vp1, vp2])
            errors = np.stack([_vp_errors(points, vp, radius) for vp in vps], axis=1)
            assignments = errors.argmin(axis=1)
            minima = errors.min(axis=1)
            # A predicted floor direction cannot support the vertical family.
            # Apply this train-only constraint before choosing a hypothesis,
            # rather than selecting a contradictory winner and rejecting later.
            if np.any(floor & (assignments == 0)):
                result["semantic_hypotheses_rejected"] += 1
                continue
            unique = np.sum(errors <= limit, axis=1) == 1
            if any(
                np.sum(
                    (assignments == axis)
                    & unique
                    & (minima <= limit)
                    & (~floor if axis == 0 else floor)
                )
                < (3 if axis == 0 else 2)
                for axis in range(3)
            ):
                continue
            score = float(np.mean(np.minimum(minima / limit, 3) ** 2))
            if best is None or score < best[0]:
                direction = GroundPlane(1, pitch, roll).rotation.T @ ray1
                yaw = (
                    math.atan2(direction[2], direction[0]) + math.pi / 2
                ) % math.pi - math.pi / 2
                best = (score, k1, f * radius, pitch, roll, yaw, vps)
    if best is None:
        result["reasons"] = ["no_supported_floor_pair_and_orthogonal_vertical"]
        return result
    score, k1, focal, pitch, roll, yaw, vps = best
    result["train_hypothesis"] = {
        "score": score,
        "k1": k1,
        "focal_px": focal,
        "pitch_rad": pitch,
        "roll_rad": roll,
        "yaw_rad": yaw,
    }
    controls = {
        "schema": 1,
        "camera": frozen["camera"],
        "image_size_px": size,
        "pixel_space": "raw",
        "initial_floor_axis_deg": math.degrees(yaw),
        "lines": [],
    }
    semantic_conflicts = []
    for row in rows:
        ideal = _ideal(np.asarray(row["points_px"]), size, k1)[None]
        errors = np.array([_vp_errors(ideal, vp, radius)[0] for vp in vps])
        axis = AXES[int(errors.argmin(axis=0))]
        if row["floor_hint"] and axis == "vertical":
            semantic_conflicts.append(row["id"])
        result["groups"].append(
            {
                "id": row["id"],
                "axis": axis,
                "validation": row["validation"],
                "hypothesis_error_px": float(errors.min()),
                "axis_support": _axis_support(errors, row["floor_hint"], limit),
                "axis_support_basis": "hypothesis_ideal_line_px",
            }
        )
        # No held-out line is removed for having a large error or a wrong semantic hint.
        controls["lines"].append(
            {
                "id": row["id"],
                "axis": axis,
                "points_px": row["points_px"],
                "validation": row["validation"],
            }
        )
    cf = CameraFile(
        camera_id=frozen["camera"],
        image_size_px=tuple(size),
        camera=Camera(focal, focal, size[0] / 2, size[1] / 2),
        plane=GroundPlane(1, pitch, roll),
        lens=Lens(k1, (size[0] / 2, size[1] / 2), radius),
        plate_file="",
    )
    try:
        proposal, checks = refine_structural_camera(cf, controls, validation_limit_px=limit)
    except ValueError as exc:
        result["reasons"] = [str(exc)]
        return result
    result["relative_camera_candidate"] = {
        "focal_px": proposal.camera.fx,
        "pitch_rad": proposal.plane.pitch,
        "roll_rad": proposal.plane.roll,
        "k1": proposal.lens.k1,
        "height_m": None,
    }
    checks["height_m_unchanged"] = None
    checks["baseline_scope"] = "train_only_direction_seed; not a commissioned camera"
    result["line_checks"] = checks
    result["semantic_axis_conflicts"] = semantic_conflicts
    scores = {row["id"]: row for row in checks["line_scores"]}
    source_rows = {row["id"]: row for row in rows}
    coverage = {axis: {"fitting": 0, "held_out": 0} for axis in AXES}
    for row in result["groups"]:
        source = source_rows[row["id"]]
        errors_by_axis = scores[row["id"]]["direction_diagnostic"]["axis_max_raw_px"]
        errors = [
            errors_by_axis[axis] if errors_by_axis[axis] is not None else math.inf
            for axis in AXES
        ]
        row["axis_support"] = _axis_support(errors, source["floor_hint"], limit)
        row["axis_support_basis"] = "refined_raw_curve_px"
        resolved = row["axis_support"]["resolved_axis"]
        if resolved == row["axis"] and (
            row["validation"] or resolved == "vertical" or source["floor_hint"]
        ):
            coverage[resolved]["held_out" if row["validation"] else "fitting"] += 1
    result["direction_coverage"] = coverage
    coverage_passed = all(
        counts["fitting"] >= (3 if axis == "vertical" else 2) and counts["held_out"] >= 1
        for axis, counts in coverage.items()
    )
    result["world_direction_assignments_complete"] = all(
        row["axis_support"]["status"] == "supported" for row in result["groups"]
    )
    result["direction_checks_passed"] = (
        checks["line_checks_passed"] and not semantic_conflicts and coverage_passed
    )
    result["status"] = (
        "conditional_candidate" if result["direction_checks_passed"] else "rejected"
    )
    result["reasons"] = (
        checks["reasons"]
        + (["floor_semantics_conflict_with_vertical_axis"] if semantic_conflicts else [])
        + (
            ["each direction needs uniquely supported fitting and held-out lines"]
            if not coverage_passed
            else []
        )
    )
    return result
