"""Student semantics cannot silently inherit teacher labels or merge known instances."""

import numpy as np
import pytest

from syncai_hydranet.cli.scene_evidence import structural_instances
from syncai_hydranet.data.studioa_supervision import CLASSES


def test_touching_tables_keep_two_instances_without_expanding_student_mask():
    labels = np.full((20, 30), CLASSES["floor"], np.uint8)
    labels[4:16, 3:27] = CLASSES["display_table"]
    proposals = np.zeros(labels.shape, np.int32)
    proposals[6:14, 4:10] = 12
    proposals[6:14, 20:26] = 27
    objects = structural_instances(labels, np.ones(labels.shape), proposals, min_pixels=4)
    assert set(np.unique(objects)) == {0, 1, 2}
    assert objects[10, 5] != objects[10, 24]
    assert not objects[labels == CLASSES["floor"]].any()


def test_proposal_never_recovers_a_class_the_student_did_not_predict():
    labels = np.full((10, 10), CLASSES["floor"], np.uint8)
    proposals = np.ones(labels.shape, np.int32)
    assert not structural_instances(labels, np.ones(labels.shape), proposals).any()
    labels[:] = CLASSES["display_table"]
    assert not structural_instances(labels, np.full(labels.shape, 0.2), proposals).any()


def test_unknown_ids_and_bad_confidence_are_rejected():
    labels = np.zeros((4, 4), np.uint8)
    with pytest.raises(ValueError, match="align"):
        structural_instances(labels, np.ones((2, 2)), labels)
    with pytest.raises(ValueError, match="confidence"):
        structural_instances(labels, np.full(labels.shape, np.nan), labels)
    with pytest.raises(ValueError, match="unknown"):
        structural_instances(labels + 255, np.ones(labels.shape), labels)


def test_changed_training_config_is_rejected_before_loading_checkpoint(tmp_path):
    import json

    from syncai_hydranet.cli.scene_evidence import prepare_model_evidence
    from syncai_hydranet.data.studioa_review import digest, write_json

    (tmp_path / "model").mkdir()
    (tmp_path / "model/best.pt").write_bytes(b"not a checkpoint")
    write_json(tmp_path / "config.json", {"original": True})
    write_json(
        tmp_path / "job.json", {"files": {"config.json": digest(tmp_path / "config.json")}}
    )
    write_json(
        tmp_path / "report.json",
        {
            "status": "completed",
            "job_sha256": digest(tmp_path / "job.json"),
            "outputs": {"model/best.pt": digest(tmp_path / "model/best.pt")},
        },
    )
    (tmp_path / "config.json").write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError, match="configuration changed"):
        prepare_model_evidence(tmp_path, tmp_path, "camera", tmp_path / "candidate")


def test_offline_world_viewer_contains_the_exported_objects(tmp_path):
    import base64
    import io
    import re
    import runpy
    from pathlib import Path

    import trimesh

    tool = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/commissioning/scene_mesh.py")
    )
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=[2, 1, 1]), node_name="display_table_0")
    source = tmp_path / "scene.glb"
    scene.export(source)
    target = tmp_path / "scene.html"
    tool["export_html"](source, target)
    html = target.read_text()
    assert "<script src=" not in html
    payload = re.search(r'base64_data="([A-Za-z0-9+/=]+)"', html)
    assert payload is not None
    recovered = trimesh.load_scene(io.BytesIO(base64.b64decode(payload[1])), file_type="glb")
    assert "display_table_0" in recovered.graph.nodes_geometry
    np.testing.assert_allclose(recovered.bounds, scene.bounds)
    assert np.isfinite(recovered.camera_transform).all()


def test_support_geometry_is_bound_to_the_source_image(tmp_path):
    from PIL import Image

    from syncai_bev3d.object_instances import ObjectInstance, save_instances
    from syncai_hydranet.cli.scene_evidence import copy_support_tops

    plate = tmp_path / "plate.png"
    Image.new("RGB", (32, 24), "white").save(plate)
    source, target = tmp_path / "original.npz", tmp_path / "candidate.npz"
    mask = np.zeros((24, 32), bool)
    mask[8:18, 6:26] = True
    save_instances(
        source,
        [ObjectInstance("table_top", mask, 0.95, "table top")],
        shape=mask.shape,
        source=plate,
        categories=["table_top"],
    )
    result = copy_support_tops(source, target, plate)
    assert result["status"] == "retained" and result["observations"] == 1
    assert source.read_bytes() == target.read_bytes()
    Image.new("RGB", (32, 24), "black").save(plate)
    with pytest.raises(ValueError, match="different plate"):
        copy_support_tops(source, tmp_path / "foreign.npz", plate)
    assert not (tmp_path / "foreign.npz").exists()
