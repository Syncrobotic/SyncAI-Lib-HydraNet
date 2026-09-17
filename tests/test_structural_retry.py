"""Acquisition limits, no holdout rerolls, duplicate evidence and provenance failures."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from syncai_bev3d.structural_retry import (
    RetryPolicy,
    _seal,
    decode_window,
    extend_ledger,
    may_retry,
    plan_windows,
    run_retry,
)


def group(key, x, held=False):
    return {
        "id": key,
        "points_px": [[x, 30], [x, 130], [x, 230]],
        "validation": held,
        "floor_hint": False,
        "state": "geometric_candidate",
    }


def frozen(rows):
    return _seal(
        {
            "schema": "structural-line-groups-v1",
            "camera": "new",
            "image_size_px": [960, 540],
            "pixel_space": "raw",
            "lens_grid": [-0.6, 0],
            "groups": rows,
        }
    )


def test_repeated_line_cannot_move_partition_or_replace_pixels():
    first, _ = extend_ledger(None, frozen([group("first", 100)]), "window-a")
    incoming = group("different-local-id", 100, held=True)
    incoming["points_px"] = [[100, 10], [100, 140], [100, 280]]
    incoming["floor_hint"] = True
    result, change = extend_ledger(first, frozen([incoming]), "window-b")
    assert len(result["groups"]) == 1
    old = result["groups"][0]
    assert old["id"] == "first" and old["validation"] is False
    assert old["points_px"] == first["groups"][0]["points_px"]
    assert old["floor_hint"] is False
    assert old["seen_windows"] == ["window-a", "window-b"]
    assert change["repeat_groups"] == 1 and change["new_groups"] == 0


def test_cross_partition_bridge_rejects_the_whole_update():
    first, _ = extend_ledger(None, frozen([group("fit", 100), group("held", 106, True)]), "a")
    before = copy.deepcopy(first)
    result, change = extend_ledger(
        first, frozen([group("bridge", 103), group("novel", 300)]), "b"
    )
    assert change["conflicts"]
    assert change["conflicts"][0]["partitions"] == [False, True]
    assert result == before and first == before


def test_novel_line_gets_one_assignment_and_stays_on_that_side():
    first, _ = extend_ledger(None, frozen([group("a", 100)]), "a")
    second, change = extend_ledger(first, frozen([group("local", 300)]), "b")
    assert change["new_groups"] == 1
    third, change = extend_ledger(second, frozen([group("local-again", 300)]), "c")
    assert change["new_groups"] == 0 and len(third["groups"]) == 2
    assert [(g["id"], g["validation"]) for g in third["groups"]] == [
        (g["id"], g["validation"]) for g in second["groups"]
    ]


@pytest.mark.parametrize("change", ["camera", "image_size_px", "freeze_sha256"])
def test_contract_changes_fail_closed(change):
    first, _ = extend_ledger(None, frozen([group("a", 100)]), "a")
    incoming = frozen([group("b", 300)])
    if change == "camera":
        incoming[change] = "another"
        incoming = _seal(incoming)
    elif change == "image_size_px":
        incoming[change] = [1920, 1080]
        incoming = _seal(incoming)
    else:
        incoming[change] = "bad"
    with pytest.raises(ValueError):
        extend_ledger(first, incoming, "b")


def test_catalog_is_frozen_ordered_bounded_and_excludes_known_seed(tmp_path):
    names = [f"archive_20260816-0{i}0000_20260816-0{i}0500" for i in range(4)]
    for i, name in enumerate(names):
        (tmp_path / f"{name}.mp4").write_bytes(bytes([i]))
    seed = {"frames": [{"path": f"/images/camera__{names[0]}/0000.jpg"}]}
    plan = plan_windows(tmp_path, RetryPolicy(), seed)
    assert [w["archive_interval"] for w in plan["windows"]] == names[1:3]
    assert [o["reason"] for o in plan["omitted"]] == ["seed_archive", "attempt_budget"]
    assert plan["seed_archive_interval_known"]


def test_duplicate_video_is_not_another_window(tmp_path):
    for name, data in [("a", b"same"), ("b", b"same"), ("c", b"different")]:
        (tmp_path / f"{name}.mp4").write_bytes(data)
    plan = plan_windows(tmp_path, RetryPolicy())
    assert len(plan["windows"]) == 2
    assert plan["omitted"][0]["reason"] == "duplicate_video_content"


def test_changed_video_is_refused_before_decode(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"original")
    window = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="planned source video changed"):
        decode_window(window, tmp_path / "frames", 3, set())


def test_decoder_preserves_native_pixels_and_deduplicates_across_windows(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixture")
    window = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    offsets = []

    def decoder(args, **kwargs):
        assert kwargs["timeout"] <= 60
        if args[0] == "ffprobe":
            return SimpleNamespace(
                stdout=json.dumps(
                    {"streams": [{"width": 80, "height": 60}], "format": {"duration": "100"}}
                )
            )
        assert "-noautorotate" in args and "-n" in args
        offset = float(args[args.index("-ss") + 1])
        offsets.append(offset)
        Image.new("RGB", (80, 60), (int(offset), 0, 0)).save(Path(args[-1]))
        return SimpleNamespace()

    monkeypatch.setattr("syncai_bev3d.structural_retry.subprocess.run", decoder)
    seen = set()
    first = decode_window(window, tmp_path / "first", 3, seen)
    second = decode_window(window, tmp_path / "second", 3, seen)
    assert offsets == [10, 50, 90, 10, 50, 90]
    assert first["unique_frames"] == 3 and second["unique_frames"] == 0
    assert len(list((tmp_path / "first").glob("*.png"))) == 3
    assert not list((tmp_path / "second").glob("*.png"))
    assert all(row["duplicate"] for row in second["frames"])


@pytest.mark.parametrize(
    "stream, duration, reason",
    [
        ({"sample_aspect_ratio": "2:1"}, "10", "non-square"),
        ({"side_data_list": [{"rotation": 90}]}, "10", "rotated video"),
        ({}, "nan", "duration"),
    ],
)
def test_decoder_refuses_ambiguous_pixel_contract(
    tmp_path, monkeypatch, stream, duration, reason
):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixture")
    window = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}

    def probe(args, **_kwargs):
        assert args[0] == "ffprobe"  # Invalid coordinates must stop before decoding.
        return SimpleNamespace(
            stdout=json.dumps({"streams": [stream], "format": {"duration": duration}})
        )

    monkeypatch.setattr("syncai_bev3d.structural_retry.subprocess.run", probe)
    with pytest.raises(ValueError, match=reason):
        decode_window(window, tmp_path / "frames", 3, set())


def fake_proposals(_directory, camera, _folder):
    observations = [
        {
            "id": f"edge-{i}",
            "frame_id": f"frame-{i}",
            "points_px": [[100, 30], [100, 130], [100, 230]],
            "span_px": 200,
            "floor_fraction": 0,
        }
        for i in range(3)
    ]
    return {
        "schema": "structural-edge-proposals-v1",
        "camera": camera,
        "pixel_space": "raw",
        "image_size_px": [960, 540],
        "observations": observations,
        "tracks": [
            {
                "id": "track-0",
                "state": "persistent",
                "observation_ids": [o["id"] for o in observations],
            }
        ],
    }


def fake_decode(_window, output, _count, _seen):
    output.mkdir()
    return {"unique_frames": 3, "frames": []}


def videos(tmp_path):
    directory = tmp_path / "videos"
    directory.mkdir()
    for i in range(5):
        (directory / f"{i}.mp4").write_bytes(bytes([i]))
    return directory


def test_campaign_hits_budget_without_rerolling_repeated_lines(tmp_path, monkeypatch):
    monkeypatch.setattr("syncai_bev3d.structural_retry.decode_window", fake_decode)
    result = run_retry("new", videos(tmp_path), tmp_path / "run", processor=fake_proposals)
    assert result["status"] == "insufficient_evidence"
    assert result["stop_reason"] == "attempt_budget_exhausted"
    assert len(result["attempts"]) == 3
    assert [a["merge"]["new_groups"] for a in result["attempts"]] == [1, 0, 0]
    ledger = json.loads((tmp_path / "run/ledger.final.json").read_text())
    assert len(ledger["groups"]) == 1
    assert not result["deployment_ready"]


@pytest.mark.parametrize("status", ["rejected", "conditional_candidate"])
def test_completed_geometric_check_is_terminal_even_when_more_sources_exist(
    tmp_path, monkeypatch, status
):
    monkeypatch.setattr("syncai_bev3d.structural_retry.decode_window", fake_decode)

    def solver(_groups):
        return {
            "status": status,
            "line_checks": {"line_checks_passed": status != "rejected"},
            "reasons": ["held-out structural lines exceed the raw-pixel limit"],
        }

    result = run_retry(
        "new", videos(tmp_path), tmp_path / "run", processor=fake_proposals, solver=solver
    )
    assert result["status"] == status
    assert len(result["attempts"]) == 1
    assert result["stop_reason"] == "terminal_direction_decision"


def test_execution_failure_has_a_durable_window_and_reason(tmp_path, monkeypatch):
    def fail(*_args):
        raise ValueError("source mutated")

    monkeypatch.setattr("syncai_bev3d.structural_retry.decode_window", fail)
    result = run_retry("new", videos(tmp_path), tmp_path / "run", processor=fake_proposals)
    assert result["status"] == "failed"
    assert result["failed_window"].startswith("window-")
    assert "source mutated" in result["reasons"][0]
    assert json.loads((tmp_path / "run/status.json").read_text()) == result


def test_campaign_stops_before_scoring_a_cross_window_identity_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr("syncai_bev3d.structural_retry.decode_window", fake_decode)
    calls = 0

    def processor(directory, camera, folder):
        nonlocal calls
        calls += 1
        report = fake_proposals(directory, camera, folder)
        if calls == 1:
            second = copy.deepcopy(report)
            for row in second["observations"]:
                row["id"] += "-parallel"
                row["points_px"] = [[106, 30], [106, 130], [106, 230]]
            second["tracks"][0]["id"] = "track-1"
            second["tracks"][0]["observation_ids"] = [r["id"] for r in second["observations"]]
            report["observations"] += second["observations"]
            report["tracks"] += second["tracks"]
        else:
            for row in report["observations"]:
                row["points_px"] = [[103, 30], [103, 130], [103, 230]]
        return report

    result = run_retry("new", videos(tmp_path), tmp_path / "run", processor=processor)
    assert result["status"] == "partition_conflict"
    assert result["stop_reason"] == "ambiguous_line_identity"
    assert calls == 2
    assert not (tmp_path / "run/attempt-02/directions.json").exists()
    ledger = json.loads((tmp_path / "run/ledger.final.json").read_text())
    assert len(ledger["groups"]) == 2


def test_insufficient_status_alone_cannot_trigger_retry_after_scoring():
    candidate = {
        "status": "insufficient_evidence",
        "reasons": ["fewer_than_four_fitting_floor_groups"],
    }
    assert may_retry(candidate)
    candidate["line_checks"] = {}
    assert not may_retry(candidate)
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
