"""Contact geometry, source invalidation and honest residuals for opening controls."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from syncai_bev3d.opening_controls import fit_controls, load_controls, source_identity
from syncai_bev3d.scene_audit import capture_inputs
from syncai_bev3d.surfaces import _project, scene_surfaces
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, GroundPlane


def fixture(root):
    cf = CameraFile(
        camera_id="sample",
        image_size_px=(640, 480),
        camera=Camera(fx=420, fy=420, cx=320, cy=240),
        plane=GroundPlane(height=2.8, pitch=math.radians(25)),
        plate_file="plate.png",
    )
    folder = root / "runs/commission01/sample/masks"
    folder.mkdir(parents=True)
    cf.save(root / "runs/commission01/sample.camera.json")
    Image.new("RGB", cf.image_size_px).save(root / "plate.png")

    def project(points):
        return _project(np.asarray(points, float), cf, (480, 640)).tolist()

    panels = []
    for i, (left, right) in enumerate([(-1.5, 0.0), (0.0, 1.5)]):
        panels.append(
            {
                "id": f"panel{i}",
                "kind": "glass_door",
                "left_edge_px": project([[left, y, 5] for y in [0, 1, 2.1]]),
                "right_edge_px": project([[right, y, 5] for y in [0, 1, 2.1]]),
                "top_edge_px": project([[left, 2.1, 5], [right, 2.1, 5]]),
            }
        )
    mask = Image.new("L", cf.image_size_px)
    ImageDraw.Draw(mask).polygon(
        [tuple(p) for p in project([[-1.5, 0, 5], [-1.5, 2.1, 5], [1.5, 2.1, 5], [1.5, 0, 5]])],
        fill=255,
    )
    mask.save(folder / "glass_door.png")
    data = {
        "schema": 1,
        "camera": "sample",
        "image_size_px": list(cf.image_size_px),
        "reviewer": "synthetic test",
        "evidence": "known vertical plane",
        "source_identity": source_identity(root, "sample"),
        "floor_contact_px": project([[-1.5, 0, 5], [1.5, 0, 5]]),
        "replaces_kinds": ["glass_door"],
        "panels": panels,
    }
    path = root / "controls.json"
    path.write_text(json.dumps(data))
    return cf, data, path, np.asarray(mask) > 127


def test_contacts_recover_plane_and_keep_individual_panels(tmp_path):
    cf, data, path, _ = fixture(tmp_path)
    controls = load_controls(path, tmp_path, "sample")
    surfaces, rows = fit_controls(controls, cf)
    assert len(surfaces) == 2
    for surface, row in zip(surfaces, rows, strict=True):
        assert np.allclose(surface.points[:, 1], 5)
        assert np.isclose(np.linalg.norm(surface.points[1] - surface.points[0]), 1.5)
        assert np.isclose(surface.height, 2.1)
        assert row["boundary_fit_residual_px"]["max"] < 0.2
        assert surface.iou is None  # a fitted corner is not mask IoU
        assert "not independent" in row["boundary_fit_residual_px"]["scope"]
        assert row["hinge_and_swing"].startswith("unknown")
    assert np.allclose(surfaces[0].points[1], surfaces[1].points[0])
    assert data == controls


@pytest.mark.parametrize("changed", ["plate", "camera", "mask", "new_mask"])
def test_changes_invalidate_controls_instead_of_reusing_stale_corners(tmp_path, changed):
    _, _, path, _ = fixture(tmp_path)
    if changed == "plate":
        Image.new("RGB", (640, 480), "white").save(tmp_path / "plate.png")
    elif changed == "camera":
        camera = tmp_path / "runs/commission01/sample.camera.json"
        data = json.loads(camera.read_text())
        data["plane"]["height"] += 0.1
        camera.write_text(json.dumps(data))
    else:
        name = "glass_door" if changed == "mask" else "glass"
        Image.new("L", (640, 480)).save(tmp_path / f"runs/commission01/sample/masks/{name}.png")
    with pytest.raises(ValueError, match="stale opening controls"):
        load_controls(path, tmp_path, "sample")


def test_cropped_lintel_is_a_convention_not_a_measurement(tmp_path):
    cf, data, _, _ = fixture(tmp_path)
    for panel in data["panels"]:
        panel.update(top_edge_px=None, top_visibility="cropped")
    surfaces, rows = fit_controls(data, cf)
    assert all(s.height == 2.4 for s in surfaces)
    assert all("drawing convention" in r["top_source"] for r in rows)


def test_incompatible_pose_remains_visible_in_frame_residuals(tmp_path):
    cf, data, _, _ = fixture(tmp_path)
    wrong = replace(cf, plane=replace(cf.plane, pitch=math.radians(45)))
    surfaces, rows = fit_controls(data, wrong)
    assert surfaces
    assert any(r["review_status"] == "calibration_or_observation_review" for r in rows)
    assert all(s.bottom == 0 for s in surfaces)


def test_overlapping_or_unsupported_panels_abstain_as_a_group(tmp_path):
    cf, data, _, _ = fixture(tmp_path)
    data["panels"][1]["left_edge_px"] = data["panels"][0]["left_edge_px"]
    surfaces, rows = fit_controls(data, cf)
    assert not surfaces and all(r["status"] == "abstained" for r in rows)
    assert all("overlapping" in r["reason"] for r in rows)


def test_invalid_or_degenerate_contact_is_not_used(tmp_path):
    cf, data, _, _ = fixture(tmp_path)
    data["floor_contact_px"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite in-frame"):
        fit_controls(data, cf)
    data["floor_contact_px"][0] = data["floor_contact_px"][1]
    surfaces, rows = fit_controls(data, cf)
    assert not surfaces and "baseline too short" in rows[0]["reason"]


def test_scene_replaces_merged_mask_with_two_bound_panels(tmp_path):
    cf, _, path, mask = fixture(tmp_path)
    ev = SimpleNamespace(cf=cf, walk=~mask)
    report = []
    surfaces = scene_surfaces("sample", ev, tmp_path, opening_controls=path, report=report)
    assert len(surfaces) == 2
    assert {s.source_id for s in surfaces} == {"control:panel0", "control:panel1"}
    assert any(r["status"] == "superseded_by_controls" for r in report)
    assert sum(len(r["final_surfaces"]) for r in report) == 2
    with pytest.raises(ValueError, match="unbound candidate masks"):
        scene_surfaces("sample", ev, tmp_path, opening_controls=path, mask_overrides={})


def test_manifest_tracks_external_control_file_edits(tmp_path):
    _, _, path, _ = fixture(tmp_path)
    before = capture_inputs(tmp_path, "sample", opening_controls=path)
    path.write_text(path.read_text() + "\n")
    after = capture_inputs(tmp_path, "sample", opening_controls=path)
    assert before["inputs"] != after["inputs"]
