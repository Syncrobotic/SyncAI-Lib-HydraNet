"""Check recovery, holdout isolation and evidence binding for line calibration."""

import copy
import json
import math
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from syncai_bev3d.opening_controls import source_identity
from syncai_bev3d.render_provenance import sha256
from syncai_bev3d.structural_controls import load_structural_controls, refine_structural_camera
from syncai_hydranet.geometry.camera_json import CameraFile, Lens
from syncai_hydranet.geometry.ground import Camera, GroundPlane, distort_points


def example():
    truth = CameraFile(
        camera_id="sample",
        image_size_px=(960, 540),
        camera=Camera(720, 720, 480, 270),
        plane=GroundPlane(2.8, 0.5, -0.15),
        lens=Lens(-0.3, (480, 270), math.hypot(960, 540) / 2),
        plate_file="plate.png",
    )
    assert truth.lens is not None
    yaw = 0.45
    rows = []
    directions = {
        "vertical": [0, 1, 0],
        "floor_u": [math.cos(yaw), 0, math.sin(yaw)],
        "floor_v": [-math.sin(yaw), 0, math.cos(yaw)],
    }
    for axis, direction in directions.items():
        d = truth.plane.rotation @ direction
        vp = np.array([720 * d[0] + 480 * d[2], 720 * d[1] + 270 * d[2], d[2]])
        for i, centre in enumerate([(160, 210), (460, 170), (780, 230), (650, 360)]):
            line = np.cross([*centre, 1], vp)
            tangent = np.array([-line[1], line[0]]) / np.linalg.norm(line[:2])
            ideal = centre + np.linspace(-75, 75, 5)[:, None] * tangent
            raw = distort_points(ideal, -0.3, (480, 270), truth.lens.radius_px)
            rows.append(
                {
                    "id": f"{axis}_{i}",
                    "axis": axis,
                    "points_px": raw.tolist(),
                    "validation": i == 3,
                }
            )
    controls = {
        "schema": 1,
        "camera": "sample",
        "image_size_px": [960, 540],
        "pixel_space": "raw",
        "initial_floor_axis_deg": 30,
        "lines": rows,
    }
    current = replace(
        truth,
        camera=replace(truth.camera, fx=390, fy=390),
        plane=replace(truth.plane, pitch=0.7, roll=-0.1),
        lens=replace(truth.lens, k1=-0.2),
    )
    return truth, current, controls


def test_recovers_joint_geometry_without_claiming_metric_accuracy():
    truth, current, controls = example()
    proposal, report = refine_structural_camera(current, controls)
    assert report["line_checks_passed"]
    assert not report["deployment_ready"] and not report["independent_metric_accuracy"]
    assert proposal.camera.fx == pytest.approx(truth.camera.fx, abs=0.01)
    assert proposal.plane.pitch == pytest.approx(truth.plane.pitch, abs=1e-5)
    assert proposal.plane.roll == pytest.approx(truth.plane.roll, abs=1e-5)
    assert proposal.lens.k1 == pytest.approx(truth.lens.k1, abs=1e-5)
    assert proposal.plane.height == current.plane.height and proposal.zones == ()
    assert max(r["after_max_raw_px"] for r in report["line_scores"]) < 0.01


def test_corrupt_holdout_cannot_change_fit_or_select_a_start():
    _, current, controls = example()
    first, _ = refine_structural_camera(current, controls)
    controls["lines"][3]["points_px"][0][0] += 40
    second, report = refine_structural_camera(current, controls)
    assert first == second
    assert not report["line_checks_passed"]
    assert any("held-out" in r for r in report["reasons"])


@pytest.mark.parametrize("problem", ["axis", "holdout", "size", "nan", "duplicate", "short"])
def test_malformed_and_underconstrained_controls_fail(problem):
    _, current, controls = example()
    if problem == "axis":
        controls["lines"][0]["axis"] = "unknown"
    elif problem == "holdout":
        controls["lines"] = [r for r in controls["lines"] if r["axis"] != "floor_v"]
    elif problem == "size":
        controls["image_size_px"] = [1920, 1080]
    elif problem == "nan":
        controls["lines"][0]["points_px"][0][0] = float("nan")
    elif problem == "duplicate":
        controls["lines"][1]["id"] = controls["lines"][0]["id"]
    else:
        controls["lines"][0]["points_px"] = [[20, 20], [21, 21], [22, 22]]
    with pytest.raises(ValueError):
        refine_structural_camera(current, controls)


def test_changed_original_frame_or_unrecorded_scaling_is_rejected(tmp_path):
    _, cf, controls = example()
    (tmp_path / "runs/commission01").mkdir(parents=True)
    cf.save(tmp_path / "runs/commission01/sample.camera.json")
    Image.new("RGB", cf.image_size_px).save(tmp_path / "plate.png")
    Image.new("RGB", (1920, 1080)).save(tmp_path / "original.png")
    controls.update(
        source_identity=source_identity(tmp_path, "sample"),
        reviewer="test",
        evidence="synthetic",
        annotation_source={
            "path": "original.png",
            "sha256": sha256(tmp_path / "original.png"),
            "original_size_px": [1920, 1080],
            "scale_to_camera": [0.5, 0.5],
        },
    )
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls))
    assert load_structural_controls(path, tmp_path, "sample") == controls
    wrong = copy.deepcopy(controls)
    wrong["annotation_source"]["scale_to_camera"] = [1, 1]
    path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="scale"):
        load_structural_controls(path, tmp_path, "sample")
    path.write_text(json.dumps(controls))
    Image.new("RGB", (1920, 1080), "white").save(tmp_path / "original.png")
    with pytest.raises(ValueError, match="annotation frame"):
        load_structural_controls(path, tmp_path, "sample")
