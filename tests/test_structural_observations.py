"""Raw evidence, temporal rejection, provenance and closed capability boundaries."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from syncai_bev3d.structural_observations import (
    ObservationConfig,
    extract_edges,
    group_tracks,
    observe_directory,
)


def scene(shift=0):
    image = Image.new("RGB", (480, 320), "#303030")
    draw = ImageDraw.Draw(image)
    # A curved boundary, a diagonal, and a moving short edge on a fixed camera.
    curve = [(int(90 + 0.0005 * (y - 160) ** 2), y) for y in range(30, 290)]
    draw.line(curve, fill="white", width=3)
    draw.line([(180, 50), (430, 270)], fill="white", width=3)
    draw.line([(250 + shift, 50), (250 + shift, 140)], fill="white", width=3)
    return np.asarray(image)


def test_native_curved_edges_are_measured_and_axis_is_unresolved():
    image = scene()
    rows, _ = extract_edges(image)
    curved = [row for row in rows if np.mean(np.asarray(row["points_px"])[:, 0]) < 120]
    assert curved, "must retain native curvature rather than only exact straight lines"
    for row in rows:
        points = np.asarray(row["points_px"])
        assert points.dtype.kind in "iu"
        assert (points >= 0).all() and (points < [480, 320]).all()
        assert row["axis"] is None and not row["calibration_eligible"]
        assert row["span_px"] > 40
    # Independent analytic boundary: local maxima stay near the actual drawn curve.
    points = np.concatenate([np.asarray(row["points_px"]) for row in curved])
    distance = np.abs(points[:, 0] - (90 + 0.0005 * (points[:, 1] - 160) ** 2))
    assert np.quantile(distance, 0.9) < 4


def track_row(frame, x, name):
    return {"id": name, "frame_id": frame, "points_px": [[x, 10], [x, 50], [x, 100]]}


def test_tracks_do_not_promote_transient_or_ambiguous_matches():
    rows = [track_row("a", 10, "a1"), track_row("b", 10, "b1")]
    rows += [track_row("a", 40, "moving"), track_row("b", 80, "moved")]
    tracks = group_tracks(rows, 2, 200, 0.5)
    assert [t["state"] for t in tracks].count("persistent") == 1
    assert [t["state"] for t in tracks].count("transient") == 2
    rows.append(track_row("a", 11, "a2"))
    tracks = group_tracks(rows, 2, 200, 0.5)
    assert tracks[0]["state"] == "ambiguous_match"
    assert all(t["split"] == "unassigned" for t in tracks)


def test_transitive_drift_is_not_a_stable_track():
    rows = [track_row("a", 10, "a"), track_row("b", 12, "b"), track_row("c", 14, "c")]
    tracks = group_tracks(rows, 3, 200, 0.5)
    assert len(tracks) == 1
    assert tracks[0]["state"] == "ambiguous_match"


def test_blank_and_duplicate_frames_never_supply_calibration_evidence(tmp_path):
    Image.new("RGB", (100, 80), "gray").save(tmp_path / "000.png")
    Image.new("RGB", (100, 80), "gray").save(tmp_path / "001.png", compress_level=0)
    report = observe_directory(tmp_path, "unseen")
    assert report["summary"]["selected_frames"] == 1
    assert report["summary"]["observations"] == 0
    assert report["status"] == "insufficient_evidence"
    assert report["frames"][1]["reason"] == "duplicate_content"
    assert not report["calibration_ready"] and not report["deployment_ready"]


def test_directory_is_source_bound_and_camera_name_cannot_change_result(tmp_path):
    for i in range(3):
        Image.fromarray(scene(i * 25)).save(tmp_path / f"{i:03}.png")
    first = observe_directory(tmp_path, "unseen-store-A")
    second = observe_directory(tmp_path, "Kaohsiung-cam04")
    first["camera"] = second["camera"]
    assert first == second
    assert first["summary"]["persistent_tracks"] > 0
    assert not first["calibration_ready"]
    for frame in first["frames"]:
        assert (
            frame["sha256"]
            == hashlib.sha256(
                (tmp_path / f"{int(frame['id'].split('-')[1]):03}.png").read_bytes()
            ).hexdigest()
        )
    assert all(row["axis"] is None for row in first["observations"])


def test_mixed_native_resolutions_are_rejected(tmp_path):
    Image.new("RGB", (100, 80)).save(tmp_path / "000.png")
    Image.new("RGB", (200, 160)).save(tmp_path / "001.png")
    with pytest.raises(ValueError, match="mixed native"):
        observe_directory(tmp_path, "camera")


def test_cli_never_overwrites_an_existing_report(tmp_path):
    path = Path(__file__).resolve().parents[1] / "tools/commissioning/structural_observe.py"
    spec = importlib.util.spec_from_file_location("structural_observe_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "report.json"
    output.write_text("existing")
    with pytest.raises(SystemExit):
        module.main(["camera", str(tmp_path), "--out", str(output)])
    assert output.read_text() == "existing"


def test_changed_selected_source_is_rejected(tmp_path, monkeypatch):
    import syncai_bev3d.structural_observations as module

    Image.fromarray(scene()).save(tmp_path / "000.png")
    original = module._read
    calls = 0

    def changing(path):
        nonlocal calls
        calls += 1
        rgb, digest = original(path)
        return rgb, digest if calls == 1 else "changed"

    monkeypatch.setattr(module, "_read", changing)
    with pytest.raises(ValueError, match="changed"):
        observe_directory(tmp_path, "camera")


def test_budgets_and_spatial_sampling_are_deterministic(tmp_path):
    for i in range(5):
        Image.new("RGB", (100, 80), (10 + i, 10, 10)).save(tmp_path / f"{i:03}.png")
    report = observe_directory(
        tmp_path, "camera", ObservationConfig(max_scan_frames=3, max_frames=2)
    )
    assert report["summary"]["scanned_frames"] == 3
    assert report["summary"]["selected_frames"] == 2
    assert [r["reason"] for r in report["frames"]].count("scan_budget") == 2
    with pytest.raises(ValueError):
        ObservationConfig(max_scan_frames=1, max_frames=2)
