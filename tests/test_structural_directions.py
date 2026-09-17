"""Direction recovery, partition isolation and conservative raw-line grouping."""

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from syncai_bev3d.structural_directions import (
    add_direction_checks,
    freeze_line_groups,
    infer_directions,
)
from syncai_hydranet.geometry.ground import GroundPlane, distort_points


def seal(groups):
    groups.pop("freeze_sha256", None)
    groups["freeze_sha256"] = hashlib.sha256(
        json.dumps(groups, sort_keys=True).encode()
    ).hexdigest()
    return groups


def synthetic_groups():
    rng = np.random.default_rng(17)
    rotation = GroundPlane(1, 0.5, -0.15).rotation
    directions = [
        [0, 1, 0],
        [math.cos(0.45), 0, math.sin(0.45)],
        [-math.sin(0.45), 0, math.cos(0.45)],
    ]
    groups = []
    for axis, direction in enumerate(directions):
        ray = rotation @ direction
        vp = np.array([720 * ray[0] + 480 * ray[2], 720 * ray[1] + 270 * ray[2], ray[2]])
        for i in range(12):
            centre = rng.uniform([140, 130], [820, 410])
            line = np.cross([*centre, 1], vp)
            tangent = np.array([-line[1], line[0]]) / np.linalg.norm(line[:2])
            ideal = centre + np.linspace(-80, 80, 12)[:, None] * tangent
            raw = distort_points(ideal, -0.3, (480, 270), math.hypot(480, 270))
            groups.append(
                {
                    "id": f"axis-{axis}-{i}",
                    "points_px": raw.tolist(),
                    "validation": i % 4 == 0,
                    "state": "geometric_candidate",
                    "floor_hint": axis != 0,
                }
            )
    return seal(
        {
            "schema": "structural-line-groups-v1",
            "camera": "unseen",
            "image_size_px": [960, 540],
            "pixel_space": "raw",
            "groups": groups,
        }
    )


def test_recovers_directions_without_inventing_metric_scale():
    result = infer_directions(synthetic_groups())
    assert result["direction_checks_passed"], result["reasons"]
    candidate = result["relative_camera_candidate"]
    assert candidate["focal_px"] == pytest.approx(720, abs=1)
    assert candidate["k1"] == pytest.approx(-0.3, abs=0.001)
    assert candidate["pitch_rad"] == pytest.approx(0.5, abs=0.001)
    assert candidate["roll_rad"] == pytest.approx(-0.15, abs=0.001)
    assert candidate["height_m"] is None
    assert result["line_checks"]["height_m_unchanged"] is None
    assert not result["deployment_ready"] and result["scale_status"] == "unknown"


def test_bad_held_out_points_cannot_select_lens_or_change_fitted_camera():
    groups = synthetic_groups()
    first = infer_directions(groups)
    changed = copy.deepcopy(groups)
    changed["groups"][0]["points_px"][3][0] += 40
    second = infer_directions(seal(changed))
    assert first["train_hypothesis"] == second["train_hypothesis"]
    assert first["relative_camera_candidate"] == second["relative_camera_candidate"]
    assert not second["direction_checks_passed"]
    assert any("held-out" in reason for reason in second["reasons"])
    row = next(g for g in second["groups"] if g["id"] == groups["groups"][0]["id"])
    assert row["axis_support"]["status"] == "unsupported"
    assert row["axis_support"]["resolved_axis"] is None


@pytest.mark.parametrize("only_vertical_holdout", [True, False])
def test_line_through_two_vanishing_points_cannot_claim_a_unique_direction(
    only_vertical_holdout,
):
    groups = synthetic_groups()
    original = infer_directions(groups)
    rotation = GroundPlane(1, 0.5, -0.15).rotation
    vps = []
    for direction in ([0, 1, 0], [-math.sin(0.45), 0, math.cos(0.45)]):
        ray = rotation @ direction
        vps.append(np.array([720 * ray[0] + 480 * ray[2], 720 * ray[1] + 270 * ray[2], ray[2]]))
    line = np.cross(*vps)
    line /= np.linalg.norm(line[:2])
    centre = np.array([480.0, 270.0])
    centre -= (line[:2] @ centre + line[2]) * line[:2]
    tangent = np.array([-line[1], line[0]])
    ideal = centre + np.linspace(-60, 60, 12)[:, None] * tangent
    raw = distort_points(ideal, -0.3, (480, 270), math.hypot(480, 270))
    assert (raw >= 0).all() and (raw < [960, 540]).all()
    # Leave this ambiguous line as the only nominal vertical holdout. Other
    # clear held-out families and all fitting observations remain unchanged.
    if only_vertical_holdout:
        groups["groups"] = [
            row
            for row in groups["groups"]
            if not (row["validation"] and row["id"].startswith("axis-0-"))
        ]
    groups["groups"].append(
        {
            "id": "two-directions",
            "points_px": raw.tolist(),
            "validation": True,
            "state": "geometric_candidate",
            "floor_hint": False,
        }
    )
    result = infer_directions(seal(groups))
    assert result["train_hypothesis"] == original["train_hypothesis"]
    assert result["relative_camera_candidate"] == original["relative_camera_candidate"]
    assert result["line_checks"]["line_checks_passed"]
    assert result["direction_checks_passed"] is not only_vertical_holdout
    if only_vertical_holdout:
        assert (
            "each direction needs uniquely supported fitting and held-out lines"
            in result["reasons"]
        )
        assert result["direction_coverage"]["vertical"]["held_out"] == 0
    else:
        assert result["direction_coverage"]["vertical"]["held_out"] == 3
    assert not result["world_direction_assignments_complete"]
    row = next(g for g in result["groups"] if g["id"] == "two-directions")
    assert row["axis_support"]["status"] == "ambiguous"
    assert row["axis_support"]["resolved_axis"] is None


def test_fitting_floor_conflict_is_checked_before_hypothesis_selection():
    groups = synthetic_groups()
    groups["groups"][1]["floor_hint"] = True  # Known synthetic vertical, in fitting side.
    result = infer_directions(seal(groups))
    assert result["semantic_hypotheses_rejected"] > 0
    assert not result["direction_checks_passed"]
    floor_ids = {g["id"] for g in groups["groups"] if g["floor_hint"] and not g["validation"]}
    assert not any(g["axis"] == "vertical" and g["id"] in floor_ids for g in result["groups"])


def test_no_floor_evidence_cannot_be_renamed_into_world_directions():
    groups = synthetic_groups()
    for row in groups["groups"]:
        row["floor_hint"] = False
    report = infer_directions(seal(groups))
    assert report["status"] == "insufficient_evidence"
    assert "relative_camera_candidate" not in report


def test_wrong_held_out_floor_semantics_fail_without_changing_the_fit():
    groups = synthetic_groups()
    first = infer_directions(groups)
    groups["groups"][0]["floor_hint"] = True  # The synthetic truth is a vertical edge.
    second = infer_directions(seal(groups))
    assert first["train_hypothesis"] == second["train_hypothesis"]
    assert first["relative_camera_candidate"] == second["relative_camera_candidate"]
    assert second["line_checks"]["line_checks_passed"]
    assert not second["direction_checks_passed"]
    assert "floor_semantics_conflict_with_vertical_axis" in second["reasons"]


def test_frozen_groups_reject_silent_edit():
    groups = synthetic_groups()
    groups["groups"][0]["validation"] = False
    with pytest.raises(ValueError, match="changed after partition freeze"):
        infer_directions(groups)


def test_pipeline_persists_partition_and_propagates_refusal(tmp_path):
    raw = {
        "schema": "structural-edge-proposals-v1",
        "pixel_space": "raw",
        "camera": "new",
        "image_size_px": [960, 540],
        "observations": [],
        "tracks": [],
        "status": "partial",
        "reasons": ["direction_hypotheses_pending"],
    }
    target = tmp_path / "groups.freeze.json"
    report = add_direction_checks(raw, target)
    assert report["status"] == "insufficient_evidence"
    assert "direction_hypotheses_pending" not in report["reasons"]
    frozen = json.loads(target.read_text())
    assert report["directions"]["groups_sha256"] == frozen["freeze_sha256"]
    assert raw["status"] == "partial"
    with pytest.raises(FileExistsError):
        add_direction_checks(raw, target)


def test_disjoint_fragments_share_one_partition_but_parallel_lines_do_not():
    observations, tracks = [], []
    # Two disjoint fragments on the same edge, plus a separate parallel edge.
    for i, points in enumerate(
        [
            [[100, 20], [100, 80], [100, 140]],
            [[100, 260], [100, 320], [100, 380]],
            [[170, 20], [170, 80], [170, 140]],
        ]
    ):
        ids = []
        for frame in range(3):
            key = f"edge-{i}-{frame}"
            ids.append(key)
            observations.append(
                {
                    "id": key,
                    "frame_id": f"frame-{frame}",
                    "points_px": points,
                    "span_px": 120,
                    "floor_fraction": 0,
                }
            )
        tracks.append({"id": f"track-{i}", "state": "persistent", "observation_ids": ids})
    report = {
        "schema": "structural-edge-proposals-v1",
        "pixel_space": "raw",
        "camera": "new",
        "image_size_px": [960, 540],
        "observations": observations,
        "tracks": tracks,
    }
    groups = freeze_line_groups(report)
    assert len(groups["groups"]) == 2
    merged = next(g for g in groups["groups"] if len(g["track_ids"]) == 2)
    assert merged["track_ids"] == ["track-0", "track-1"]
    assert len(merged["observation_ids"]) == 6
    assert not merged["physical_identity_verified"]
    assert freeze_line_groups(report) == groups


def test_collinear_ambiguous_fragments_are_resolved_but_crossed_edges_are_not():
    observations, tracks = [], []
    for track_id, x_end in [("fragmented", 100), ("crossed", 180)]:
        ids = []
        for frame in range(3):
            for segment, points in enumerate(
                [
                    [[100, 20], [100, 80], [100, 140]],
                    [[x_end, 260], [x_end, 320], [x_end, 380]],
                ]
            ):
                key = f"{track_id}-{frame}-{segment}"
                ids.append(key)
                observations.append(
                    {
                        "id": key,
                        "frame_id": f"frame-{frame}",
                        "points_px": points,
                        "span_px": 120,
                        "floor_fraction": 1,
                    }
                )
        tracks.append({"id": track_id, "state": "ambiguous_match", "observation_ids": ids})
    report = {
        "schema": "structural-edge-proposals-v1",
        "pixel_space": "raw",
        "camera": "new",
        "image_size_px": [960, 540],
        "observations": observations,
        "tracks": tracks,
    }
    groups = freeze_line_groups(report)
    assert len(groups["groups"]) == 1
    assert groups["groups"][0]["track_ids"] == ["fragmented"]
    assert groups["groups"][0]["floor_hint"]  # Three unique floor frames, not six votes.
    assert any(row["track_id"] == "crossed" for row in groups["excluded_tracks"])
