"""Background failures must leave evidence and must not outlive the time budget."""

import json
import runpy
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools/commissioning"
TRAINING = runpy.run_path(str(TOOLS / "stage0_train.py"))
Worker, sha256, spread = (TRAINING[name] for name in ("Worker", "sha256", "spread"))
rates = runpy.run_path(str(TOOLS / "staff_store_probe.py"))["rates"]


def worker(tmp_path):
    (tmp_path / "snapshot").mkdir()
    source = tmp_path / "source.txt"
    source.write_text("frozen")
    plan = {
        "deadline": (datetime.now(UTC) + timedelta(minutes=2)).isoformat(),
        "jobs": [],
        "frozen_files": {"source.txt": sha256(source)},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    return Worker(tmp_path)


def test_timeout_terminates_child_and_records_status(tmp_path):
    job = worker(tmp_path)
    started = time.monotonic()
    code, timed_out = job.step(
        [sys.executable, "-c", "import time; time.sleep(60)"], "sleep", 0
    )
    assert timed_out and code != 0
    assert time.monotonic() - started < 10
    assert job.child is None
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["active_step"] == "sleep"
    assert "child_pid" not in status
    assert (tmp_path / "REPORT.zh-TW.md").is_file()


def test_changed_frozen_inputs_refuse_training(tmp_path):
    job = worker(tmp_path)
    job.verify()
    (tmp_path / "source.txt").write_text("changed")
    with pytest.raises(RuntimeError, match="frozen input changed"):
        job.verify()


def test_sampling_spreads_across_session_and_single_class_is_not_balanced_accuracy():
    assert spread(list(range(100)), 4) == [0, 33, 66, 99]
    assert spread([1, 2], 4) == [1, 2]
    report = rates(np.array([1, 1]), np.array([1, 0]))
    assert report["balanced_accuracy"] is None
    assert report["customer_recall"] is None
    assert report["staff_recall"] == 0.5
