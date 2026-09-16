"""Different bucket timestamps and split ownership must not silently change intake."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def worker():
    path = Path(__file__).resolve().parents[1] / "tools/annotation/studioa_gcs.py"
    spec = importlib.util.spec_from_file_location("gcs_intake_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_bucket_filename_is_not_old_recording_utc(worker):
    uris = [
        f"gs://studioa/taichung-8/2026/09/10/taichung-8_20260910_{t}.mp4"
        for t in ("035003", "115003", "120003")
    ]
    uri, gap = worker.choose_clip(uris, "2026-09-10", "12:00")
    assert uri.endswith("120003.mp4") and gap == 3
    with pytest.raises(ValueError, match="15 filename-minutes"):
        worker.choose_clip(uris[:1], "2026-09-10", "12:00")


@pytest.mark.parametrize("camera", ["Taichung-cam01", "Tao-Hsin-cam04"])
def test_intake_refuses_validation_or_held_out_camera_before_network(
    worker, monkeypatch, tmp_path, camera
):
    manifest = {
        "frames": [{"id": "a", "camera": camera}],
        "folds": {"Tao-Hsin": {"assignments": {"a": "val"}}},
    }
    monkeypatch.setattr(worker, "check_supervision", lambda _p: manifest)

    def forbidden(*_args, **_kwargs):
        pytest.fail("network reached for disallowed camera")

    monkeypatch.setattr(worker, "command", forbidden)
    with pytest.raises(ValueError, match="source-train cameras"):
        worker.plan(tmp_path, tmp_path / "out", [camera], ["2026-09-10"], ["12:00"], 1000)


def test_camera_prefix_rejects_unknown_store(worker):
    assert worker.camera_prefix("Taichung-cam08") == "taichung-8"
    with pytest.raises(ValueError, match="source stores"):
        worker.camera_prefix("Tao-Hsin-cam08")
