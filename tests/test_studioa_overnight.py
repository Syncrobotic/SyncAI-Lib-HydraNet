"""Overnight training publishes checkpoints without claiming a reconstructed world."""

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


def test_child_reports_failure_and_terminates_only_its_own_timeout(tmp_path):
    with pytest.raises(RuntimeError, match="exit 7"):
        worker["child"](
            [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "failure.log", 5
        )
    with pytest.raises(TimeoutError, match="exceeded"):
        worker["child"](
            [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "timeout.log", 0.1
        )


def test_training_delivery_does_not_claim_a_world(tmp_path):
    run = tmp_path / "jobs/new"
    (run / "model").mkdir(parents=True)
    (run / "model/best.pt").write_bytes(b"new checkpoint")
    write_json(run / "config.json", {"seed": 42})
    selected = {"name": "new", "miou": 0.3}
    delivery = worker["publish"](tmp_path, {}, selected, 10**12)
    assert delivery["checkpoint_sha256"] == digest(run / "model/best.pt")
    assert delivery["cameras"] == []
    assert delivery["scene_status"].startswith("not_built")
    assert (tmp_path / "index.html").exists()
