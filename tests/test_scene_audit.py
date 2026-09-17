"""Final exports, missing stages and snapshot identity must not masquerade as fit scores."""

import json
import math

import numpy as np
import pytest
import trimesh
from PIL import Image

from syncai_bev3d.scene_audit import (
    audit_export,
    capture_inputs,
    mask_metrics,
    raw_silhouette,
)
from syncai_bev3d.surfaces import Surface, separate_glazing
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, GroundPlane


def fixture(root):
    folder = root / "runs/commission01/sample/masks"
    folder.mkdir(parents=True)
    cache = root / "runs/site30k_qa/geometry_cache/sample.npz"
    cache.parent.mkdir(parents=True)
    np.savez(cache, horiz=np.ones((48, 64)))
    cf = CameraFile(
        camera_id="sample",
        image_size_px=(64, 48),
        camera=Camera(fx=42, fy=42, cx=32, cy=24),
        plane=GroundPlane(height=2.8, pitch=math.radians(25)),
        mask_files={"objects": "sample/masks/objects.png"},
    )
    cf.save(root / "runs/commission01/sample.camera.json")
    Image.new("L", (64, 48)).save(folder / "objects.png")
    return cf, folder


def mesh_and_rows(root, cf):
    vertices = np.array([[-1, 0, 5], [-1, 2.1, 5], [1, 2.1, 5], [1, 0, 5]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False), node_name="glass_0"
    )
    glb = root / "scene.glb"
    scene.export(glb)
    reference = raw_silhouette(vertices, faces, cf)
    Image.fromarray(reference.astype(np.uint8) * 255).save(
        root / "runs/commission01/sample/masks/glass.png"
    )
    rows = [
        {
            "kind": "glass",
            "status": "supported",
            "iou": 0.123,
            "source_id": "glass:1",
            "final_surfaces": [{"surface_id": "glass:1:0", "mesh_nodes": ["glass_0"]}],
        }
    ]
    return glb, rows


def test_export_is_scored_from_glb_and_missing_devices_are_not_zero(tmp_path):
    cf, _ = fixture(tmp_path)
    glb, rows = mesh_and_rows(tmp_path, cf)
    report = audit_export(tmp_path, "sample", glb, rows, [], [], tmp_path)
    glass = next(r for r in report["material_coverage"] if r["kind"] == "glass")
    assert glass["iou"] > 0.95
    assert rows[0]["iou"] == 0.123  # never overwritten by a different metric
    assert report["devices"]["stage"] == "not_run"
    assert report["devices"]["candidates"] is None
    assert report["geometry_provenance"]["status"] == "legacy_unverified"
    assert (
        next(r for r in report["material_coverage"] if r["kind"] == "door")["status"]
        == "missing_reference"
    )
    json.dumps(report, allow_nan=False)


def test_missing_export_node_refuses_a_complete_audit(tmp_path):
    cf, _ = fixture(tmp_path)
    glb, rows = mesh_and_rows(tmp_path, cf)
    rows[0]["final_surfaces"][0]["mesh_nodes"] = ["glass_999"]
    with pytest.raises(ValueError, match="missing GLB nodes"):
        audit_export(tmp_path, "sample", glb, rows, [], [], tmp_path)


def test_control_boundary_residual_uses_exported_vertices(tmp_path):
    from syncai_bev3d.surfaces import _project

    cf, _ = fixture(tmp_path)
    glb, rows = mesh_and_rows(tmp_path, cf)
    edges = [
        _project(np.array([[x, 0, 5], [x, 1, 5], [x, 2.1, 5]]), cf, (48, 64)).tolist()
        for x in (-1, 1)
    ]
    rows[0].update(control_id="entrance", observed_edges_px=edges)
    first = audit_export(tmp_path, "sample", glb, rows, [], [], tmp_path)
    original_error = first["controlled_openings"][0]["final_glb_boundary_residual_px"]["mean"]
    scene = trimesh.load(glb, force="scene", process=False)
    assert isinstance(scene, trimesh.Scene)
    for mesh in scene.geometry.values():
        mesh.vertices += [1.0, 0, 0]
    scene.export(glb)
    changed = audit_export(tmp_path, "sample", glb, rows, [], [], tmp_path)
    assert (
        changed["controlled_openings"][0]["final_glb_boundary_residual_px"]["mean"]
        > original_error + 2
    )
    assert rows[0]["iou"] == 0.123  # changing export must not rewrite the fitting score


def test_unreported_surface_node_is_not_silently_omitted(tmp_path):
    cf, _ = fixture(tmp_path)
    glb, _ = mesh_and_rows(tmp_path, cf)
    with pytest.raises(ValueError, match="unmapped final surface"):
        audit_export(tmp_path, "sample", glb, [], [], [], tmp_path)


def test_snapshot_detects_new_optional_masks_and_code_edits(tmp_path):
    _, folder = fixture(tmp_path)
    source = tmp_path / "src/syncai_bev3d/scene_mesh.py"
    source.parent.mkdir(parents=True)
    source.write_text("first\n")
    before = capture_inputs(tmp_path, "sample")
    Image.new("L", (64, 48)).save(folder / "glass.png")
    after = capture_inputs(tmp_path, "sample")
    assert before["inputs"] != after["inputs"]
    source.write_text("second\n")
    assert capture_inputs(tmp_path, "sample")["code"] != after["code"]


def test_empty_reference_and_empty_prediction_are_undefined_not_perfect():
    empty = np.zeros((5, 5), bool)
    report = mask_metrics(empty, empty)
    assert report["iou"] is None and report["recall"] is None
    report = mask_metrics(empty, ~empty)
    assert report["iou"] == 0 and report["recall"] == 0
    assert report["precision"] is None


def test_conflicting_glass_references_are_reported_without_changing_scores(tmp_path):
    cf, folder = fixture(tmp_path)
    glb, rows = mesh_and_rows(tmp_path, cf)
    with Image.open(folder / "glass.png") as image:
        image.save(folder / "glass_door.png")
    report = audit_export(tmp_path, "sample", glb, rows, [], [], tmp_path)
    consistency = report["material_reference_consistency"]
    assert consistency["glass_door_overlap_fraction"] == 1.0
    glass = next(r for r in report["material_coverage"] if r["kind"] == "glass")
    assert glass["iou"] > 0.95


def test_split_pieces_preserve_source_identity():
    pane = Surface(
        np.array([[-2.0, 5.0], [2.0, 5.0]]), 0, 2.4, "glass", 0.8, "floor", "glass:3"
    )
    door = Surface(
        np.array([[-0.5, 5.0], [0.5, 5.0]]), 0, 2.4, "glass_door", 0.9, "floor", "glass_door:1"
    )
    pieces = separate_glazing([pane, door])
    assert [p.source_id for p in pieces] == ["glass:3", "glass:3", "glass_door:1"]
    assert all(not (p.points[0, 0] < 0 < p.points[1, 0]) for p in pieces if p.kind == "glass")


def test_triangle_gaps_are_not_filled_by_a_convex_hull(tmp_path):
    cf, _ = fixture(tmp_path)
    left = np.array([[-2.0, 0, 5], [-2, 2, 5], [-1, 0, 5]])
    right = left + np.array([3.0, 0, 0])
    mask = raw_silhouette(np.concatenate([left, right]), [[0, 1, 2], [3, 4, 5]], cf)
    assert mask.any()
    assert not mask[:, 32].any()


def test_support_audit_scores_final_vertices_not_the_fitter_claim(tmp_path):
    from syncai_bev3d.meshes import Placement, place
    from syncai_bev3d.object_instances import ObjectInstance, save_instances
    from syncai_bev3d.scene_audit import audit_fixture_geometry
    from syncai_bev3d.scene_mesh import counter

    cf, folder = fixture(tmp_path)
    mesh = place(counter(1.2, 0.8, 0.8), Placement(0, 4, 0))
    vertices, faces = mesh
    top_faces = faces[np.isclose(vertices[faces, 1], 0.8).all(1)]
    # Frozen reference is unchanged when the exported geometry is displaced.
    reference = raw_silhouette(vertices, top_faces, cf)
    source = tmp_path / "plate.png"
    Image.new("RGB", cf.image_size_px).save(source)
    save_instances(
        folder / "support_tops.npz",
        [ObjectInstance("table_top", reference, 0.99, "frozen top")],
        shape=reference.shape,
        source=source,
        categories=["table_top"],
    )
    report = [
        {"mesh_index": 0, "top_observation_index": 0, "accepted": True, "after_top_iou": 1.0}
    ]
    baseline = audit_fixture_geometry(
        tmp_path, "sample", cf, {"display_table_0": mesh}, report, tmp_path
    )
    displaced = (vertices + np.array([3, 0, 0]), faces)
    wrong = audit_fixture_geometry(
        tmp_path, "sample", cf, {"display_table_0": displaced}, report, tmp_path
    )
    a = baseline["support_checks"][0]["final_glb_top_hull"]["iou"]
    b = wrong["support_checks"][0]["final_glb_top_hull"]["iou"]
    assert a > 0.9 and b < 0.1
