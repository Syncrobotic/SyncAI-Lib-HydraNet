"""Different bucket timestamps and split ownership must not silently change intake."""

import importlib.util
import json
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


def test_raw_media_selection_needs_bound_camera_review_and_unchanged_inputs(
    tmp_path, monkeypatch
):
    from syncai_hydranet.data.studioa_review import digest, write_json

    path = Path(__file__).resolve().parents[1] / "tools/annotation/studioa_autolabel.py"
    spec = importlib.util.spec_from_file_location("media_prepare_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture source")
    frame = {
        "id": "a",
        "image": "image.jpg",
        "image_sha256": digest(image),
        "original_split": "train",
    }
    write_json(
        tmp_path / "manifest.json", {"schema": "studioa.gcs-intake.v1", "frames": [frame]}
    )
    write_json(
        tmp_path / "report.json",
        {
            "status": "completed",
            "outputs": {
                "image.jpg": digest(image),
                "manifest.json": digest(tmp_path / "manifest.json"),
            },
        },
    )
    selection = tmp_path / "selection.json"
    review = {
        "reviewer_kind": "ai",
        "source_manifest_sha256": digest(tmp_path / "manifest.json"),
        "frames": [{"id": "a", "camera_match_confirmed": True, "reason": "same table/view"}],
    }
    write_json(selection, review)
    calls = []
    monkeypatch.setattr(module, "freeze_frames", lambda *args: calls.append(args))
    module.prepare_media(tmp_path, selection, tmp_path / "out")
    assert calls[0][2][0]["files"] == {"image.jpg": digest(image)}
    review["frames"][0]["camera_match_confirmed"] = False
    selection.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="camera confirmation"):
        module.prepare_media(tmp_path, selection, tmp_path / "out")
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="intake file changed"):
        module.prepare_media(tmp_path, selection, tmp_path / "out")
