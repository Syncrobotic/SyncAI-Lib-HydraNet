"""Overnight delivery survives camera failures and refuses stale output reuse."""

import runpy
import sys
from pathlib import Path

import pytest

from syncai_hydranet.data.studioa_review import digest, write_json

worker = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/annotation/studioa_overnight.py")
)


def test_selection_and_deadline_budget():
    with pytest.raises(ValueError, match="no completed"):
        worker["choose_model"]([])
    rows = [{"name": "a", "miou": 0.2}, {"name": "b", "miou": 0.3}]
    assert worker["choose_model"](rows)["name"] == "b"
    assert worker["render_budget"](1000, 900) == 100
    assert worker["render_budget"](1000, 1001) == 1


def test_child_reports_failure_and_terminates_only_its_own_timeout(tmp_path):
    with pytest.raises(RuntimeError, match="exit 7"):
        worker["child"](
            [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "failure.log", 5
        )
    with pytest.raises(TimeoutError, match="exceeded"):
        worker["child"](
            [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "timeout.log", 0.1
        )


def test_partial_render_delivery_and_tamper_refusal(tmp_path, monkeypatch):
    run = tmp_path / "jobs/new"
    (run / "model").mkdir(parents=True)
    (run / "model/best.pt").write_bytes(b"new checkpoint")
    write_json(run / "config.json", {"seed": 42})
    expected = digest(run / "model/best.pt")

    def fake_child(command, log, seconds):
        assert log.suffix == ".log" and 0 < seconds <= 600
        directory = Path(command[command.index("--out") + 1])
        if directory.name in {"camera4", "camera5"}:
            raise RuntimeError("simulated unavailable camera")
        directory.mkdir(parents=True)
        (directory / "world.png").write_bytes(b"preview")
        write_json(
            directory / "provenance.json",
            {
                "checkpoint_sha256": expected,
                "outputs": {"world.png": digest(directory / "world.png")},
            },
        )

    monkeypatch.setitem(worker["publish"].__globals__, "child", fake_child)
    plan = {"cameras": [f"camera{i}" for i in range(1, 6)]}
    selected = {"name": "new", "miou": 0.3}
    delivery = worker["publish"](tmp_path, plan, selected, 10**12)
    assert len(delivery["cameras"]) == 3 and len(delivery["render_errors"]) == 4
    assert (tmp_path / "index.html").exists()
    directory = tmp_path / "deliveries/new/camera1"
    assert worker["verified_render"](directory, expected)
    assert not worker["verified_render"](directory, "foreign model")
    (directory / "world.png").write_bytes(b"changed")
    assert not worker["verified_render"](directory, expected)
