"""Whole-line holdout and scope limits for vertical attitude observations."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from syncai_bev3d.opening_controls import source_identity
from syncai_bev3d.surfaces import _project
from syncai_bev3d.vertical_controls import load_vertical_controls, refine_vertical_pose
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, GroundPlane


def example():
    truth = CameraFile(
        camera_id="sample",
        image_size_px=(640, 480),
        camera=Camera(420, 420, 320, 240),
        plane=GroundPlane(2.8, math.radians(25), math.radians(2)),
        plate_file="plate.png",
    )
    controls = {
        "schema": 1,
        "camera": "sample",
        "image_size_px": [640, 480],
        "pixel_space": "raw",
        "lines": [],
    }
    for i, (x, z, held) in enumerate(
        [(-2, 5, False), (0, 6, False), (2, 5, False), (-1, 7, True), (1, 6, True)]
    ):
        points = _project(np.array([[x, y, z] for y in [0, 1, 2.1]]), truth, (480, 640))
        controls["lines"].append(
            {"id": str(i), "points_px": points.tolist(), "validation": held}
        )
    current = replace(truth, plane=replace(truth.plane, pitch=math.radians(40), roll=0))
    return truth, current, controls


def test_recovers_attitude_without_claiming_focal_or_metric_accuracy():
    truth, current, controls = example()
    candidate, report = refine_vertical_pose(current, controls)
    assert report["orientation_checks_passed"]
    assert not report["deployment_ready"]
    assert abs(candidate.plane.pitch - truth.plane.pitch) < 1e-5
    assert abs(candidate.plane.roll - truth.plane.roll) < 1e-5
    assert candidate.camera == current.camera and candidate.lens == current.lens
    assert candidate.plane.height == current.plane.height
    assert max(r["after_max_raw_px"] for r in report["line_scores"]) < 0.2


def test_held_out_line_does_not_select_or_change_the_fit():
    _, current, controls = example()
    first, _ = refine_vertical_pose(current, controls)
    controls["lines"][-1]["points_px"][0][0] += 30
    second, report = refine_vertical_pose(current, controls)
    assert first.plane == second.plane
    assert not report["orientation_checks_passed"]
    assert any("held-out" in r for r in report["reasons"])


def test_observed_floor_above_new_horizon_blocks_attitude_use():
    _, current, controls = example()
    floor = np.zeros((480, 640), bool)
    floor[5:20, 200:440] = True
    _, report = refine_vertical_pose(current, controls, walkable=floor)
    assert report["floor_check"]["above_horizon_fraction"] == 1.0
    assert not report["orientation_checks_passed"]


@pytest.mark.parametrize("problem", ["no_holdout", "short", "nan", "duplicate", "ideal_pixels"])
def test_insufficient_or_malformed_evidence_is_rejected(problem):
    _, current, controls = example()
    if problem == "no_holdout":
        for row in controls["lines"]:
            row["validation"] = False
    elif problem == "short":
        controls["lines"][0]["points_px"] = [[20, 20], [20, 21], [20, 22]]
    elif problem == "nan":
        controls["lines"][0]["points_px"][0][0] = float("nan")
    elif problem == "duplicate":
        controls["lines"][1]["id"] = controls["lines"][0]["id"]
    else:
        controls["pixel_space"] = "undistorted"
    with pytest.raises(ValueError):
        refine_vertical_pose(current, controls)


def test_source_plate_change_invalidates_vertical_observations(tmp_path):
    _, current, controls = example()
    (tmp_path / "runs/commission01").mkdir(parents=True)
    current.save(tmp_path / "runs/commission01/sample.camera.json")
    Image.new("RGB", (640, 480)).save(tmp_path / "plate.png")
    controls.update(
        source_identity=source_identity(tmp_path, "sample"),
        reviewer="test",
        evidence="synthetic",
    )
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls))
    assert load_vertical_controls(path, tmp_path, "sample") == controls
    Image.new("RGB", (640, 480), "white").save(tmp_path / "plate.png")
    with pytest.raises(ValueError, match="stale vertical"):
        load_vertical_controls(path, tmp_path, "sample")
