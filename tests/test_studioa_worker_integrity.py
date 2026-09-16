"""Completed jobs and JSON progress are evidence only while their bytes are intact."""

import importlib.util
import json
from pathlib import Path

import pytest

from syncai_hydranet.data.studioa_review import digest, write_json

path = Path(__file__).resolve().parents[1] / "tools/annotation/studioa_train.py"
spec = importlib.util.spec_from_file_location("studioa_integrity_worker", path)
assert spec and spec.loader
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def completed(tmp_path):
    job = {"files": {}, "scene_comparison": True}
    write_json(tmp_path / "job.json", job)
    outputs = {}
    for name in [
        "model/best.pt",
        "model/last.pt",
        "model/metrics.jsonl",
        "baseline_val.json",
        "validation.json",
        "initial_state.pt",
    ]:
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"bound output")
        outputs[name] = digest(p)
    report = {
        "status": "completed",
        "job_sha256": digest(tmp_path / "job.json"),
        "test_evaluated": False,
        "outputs": outputs,
    }
    write_json(tmp_path / "report.json", report)
    return job, report


def test_completed_worker_checks_outputs_before_reporting_success(tmp_path):
    completed(tmp_path)
    worker.run(tmp_path)  # Completed fast path works without CUDA or a model load.
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "completed"
    (tmp_path / "model/best.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="output changed"):
        worker.run(tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("defect", ["foreign_job", "missing_initial", "wrong_scope", "escape"])
def test_bad_completion_provenance_is_rejected(tmp_path, defect):
    job, report = completed(tmp_path)
    if defect == "foreign_job":
        report["job_sha256"] = "foreign"
    elif defect == "missing_initial":
        report["outputs"].pop("initial_state.pt")
    elif defect == "wrong_scope":
        report["test_evaluated"] = True
    else:
        report["outputs"]["../outside"] = "foreign"
    write_json(tmp_path / "report.json", report)
    with pytest.raises(ValueError):
        worker.verify_completed(tmp_path, job)


def test_json_replace_failure_preserves_previous_status_and_removes_temporary(
    tmp_path, monkeypatch
):
    p = tmp_path / "status.json"
    write_json(p, {"status": "running"})
    original = p.read_bytes()

    def fail_replace(self, target):
        assert target == p and self != p
        assert json.loads(self.read_text())["status"] == "completed"
        raise OSError("simulated publication failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        write_json(p, {"status": "completed"})
    assert p.read_bytes() == original
    assert list(tmp_path.iterdir()) == [p]


def test_nonfinite_json_does_not_truncate_existing_status(tmp_path):
    p = tmp_path / "status.json"
    write_json(p, {"status": "running"})
    original = p.read_bytes()
    with pytest.raises(ValueError):
        write_json(p, {"metric": float("nan")})
    assert p.read_bytes() == original
