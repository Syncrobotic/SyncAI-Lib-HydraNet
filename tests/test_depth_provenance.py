"""Depth reuse must retain its actual source and reject changed image inputs."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from syncai_bev3d.depth_provenance import inference_record, load_depth_record
from syncai_bev3d.geometry_cache import build_geometry_cache, save_geometry_cache
from syncai_bev3d.geometry_review import load_geometry_cache
from syncai_bev3d.render_provenance import sha256
from syncai_hydranet.geometry.camera_json import CameraFile, Lens
from syncai_hydranet.geometry.ground import Camera, GroundPlane


def fixture(root):
    plate = root / "plate.png"
    Image.new("RGB", (64, 48)).save(plate)
    cf = CameraFile(
        camera_id="sample",
        image_size_px=(64, 48),
        camera=Camera(45, 45, 32, 24),
        plane=GroundPlane(2.8, np.radians(45)),
        lens=Lens(-0.2, (32, 24), 40),
        plate_file=str(plate),
    )
    depth = root / "source.npy"
    np.save(depth, np.full((48, 64), 3.0))
    manifest = depth.with_suffix(".json")
    record = inference_record(cf, plate, depth, model="synthetic", revision="test-v1")
    manifest.write_text(json.dumps(record))
    return cf, plate, depth, manifest


def cli():
    path = Path(__file__).resolve().parents[1] / "tools/commissioning/rebuild_geometry.py"
    spec = importlib.util.spec_from_file_location("rebuild_geometry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("changed", ["plate", "lens", "depth", "shape", "code"])
def test_reuse_rejects_changed_source_even_when_camera_pose_is_unchanged(tmp_path, changed):
    cf, plate, depth, manifest = fixture(tmp_path)
    if changed == "plate":
        Image.new("RGB", (64, 48), "white").save(plate)
    elif changed == "lens":
        cf = replace(cf, lens=replace(cf.lens, k1=-0.1))
    elif changed == "depth":
        np.save(depth, np.full((48, 64), 4.0))
    elif changed == "shape":
        np.save(depth, np.full((24, 32), 3.0))
        record = json.loads(manifest.read_text())
        record["depth_sha256"] = sha256(depth)
        manifest.write_text(json.dumps(record))
    else:
        record = json.loads(manifest.read_text())
        record["preprocessing"]["code_sha256"] = "old-code"
        manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="depth manifest"):
        load_depth_record(manifest, cf, plate, depth)


def test_raw_depth_remains_reusable_after_pose_focal_or_height_correction(tmp_path):
    cf, plate, depth, manifest = fixture(tmp_path)
    corrected = replace(
        cf, camera=replace(cf.camera, fx=60, fy=60), plane=GroundPlane(3.1, 0.6, 0.02)
    )
    assert load_depth_record(manifest, corrected, plate, depth)["model"] == "synthetic"


def test_external_depth_without_evidence_never_inherits_fixed_teacher_identity(tmp_path):
    cf, _, depth, manifest = fixture(tmp_path)
    manifest.unlink()
    camera = tmp_path / "camera.json"
    cf.save(camera)
    cache = tmp_path / "external.npz"
    assert (
        cli().main(
            [str(camera), "--depth", str(depth), "--depth-scale", "1", "--out", str(cache)]
        )
        == 0
    )
    with np.load(cache, allow_pickle=False) as saved:
        assert "depth_model" not in saved and "depth_revision" not in saved
        assert json.loads(str(saved["depth_source"]))["status"] == "external_unverified"


def test_fresh_inference_saves_binding_and_reuse_preserves_recorded_model(
    tmp_path, monkeypatch
):
    cf, _, _, _ = fixture(tmp_path)
    camera = tmp_path / "camera.json"
    cf.save(camera)
    tool = cli()
    monkeypatch.setattr(tool, "MODEL", "synthetic")
    monkeypatch.setattr(tool, "MODEL_REVISION", "test-v2")
    monkeypatch.setattr(tool, "run_depth", lambda image: np.full(image.shape[:2], 3.0))
    fresh, reused = tmp_path / "fresh.npz", tmp_path / "reused.npz"
    assert tool.main([str(camera), "--depth-scale", "1", "--out", str(fresh)]) == 0

    def never_run(_image):
        raise AssertionError("reuse must not run a teacher")

    monkeypatch.setattr(tool, "run_depth", never_run)
    monkeypatch.setattr(tool, "MODEL", "different-default")
    depth = fresh.with_suffix(".depth.npy")
    assert (
        tool.main(
            [str(camera), "--depth", str(depth), "--depth-scale", "1.1", "--out", str(reused)]
        )
        == 0
    )
    with np.load(reused, allow_pickle=False) as saved:
        assert str(saved["depth_model"]) == "synthetic"
        assert str(saved["depth_revision"]) == "test-v2"
        assert json.loads(str(saved["depth_source"]))["status"] == "bound"
        assert float(saved["depth_scale"]) == 1.1


def test_rebuild_rejects_plate_change_during_teacher_run(tmp_path, monkeypatch):
    cf, plate, _, _ = fixture(tmp_path)
    camera = tmp_path / "camera.json"
    cf.save(camera)
    tool = cli()

    def changed_image(image):
        Image.new("RGB", (64, 48), "white").save(plate)
        return np.full(image.shape[:2], 3.0)

    monkeypatch.setattr(tool, "run_depth", changed_image)
    cache = tmp_path / "changed.npz"
    with pytest.raises(ValueError, match="changed during"):
        tool.main([str(camera), "--depth-scale", "1", "--out", str(cache)])
    assert not cache.exists()


def test_cache_checks_actual_plate_with_explicit_isolated_root(tmp_path):
    cf, plate, _, _ = fixture(tmp_path)
    cf = replace(cf, plate_file="relative/plate.png")
    cache = tmp_path / "geometry.npz"
    arrays = build_geometry_cache(cf, np.full((48, 64), 3), depth_scale=1, frame_hw=(48, 64))
    arrays["plate_sha256"] = np.array(sha256(plate))
    save_geometry_cache(cache, arrays)
    assert load_geometry_cache(cache, cf, plate_path=plate)["gx"].shape == (48, 64)
    Image.new("RGB", (64, 48), "white").save(plate)
    with pytest.raises(ValueError, match="stale geometry cache plate"):
        load_geometry_cache(cache, cf, plate_path=plate)
