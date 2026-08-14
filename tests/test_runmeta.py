"""Run provenance: metadata, metrics files and output-directory collisions.

pytest tests/test_runmeta.py -v
"""

import json

import yaml

from syncai_hydranet.config import Config
from syncai_hydranet.utils.runmeta import (
    append_metrics,
    environment,
    git_state,
    resolve_out_dir,
    write_run_meta,
)

CFG = Config(
    {
        "experiment": "test_run",
        "output_dir": "runs/test_run",
        "seed": 42,
        "data": {"input_size": [512, 640], "datasets": [{"name": "ade20k"}]},
        "train": {"lr": 2.0e-4, "epochs": 3},
    }
)


# ------------------------------------------------------------ output directory


def test_fresh_directory_is_used_as_is(tmp_path):
    out = tmp_path / "run"
    assert resolve_out_dir(out) == out


def test_directory_with_a_checkpoint_is_never_reused(tmp_path):
    """Two runs sharing a directory means one silently overwrites the other's best.pt
    and interleaves its TensorBoard events."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "best.pt").write_bytes(b"")
    resolved = resolve_out_dir(out)
    assert resolved != out
    assert resolved.name.startswith("run-")


def test_resuming_writes_back_into_the_same_directory(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "last.pt").write_bytes(b"")
    assert resolve_out_dir(out, resuming=True) == out


def test_empty_leftover_directory_is_reused(tmp_path):
    """A directory holding only a stale log is not a run worth protecting."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "train.log").write_text("")
    assert resolve_out_dir(out) == out


# ------------------------------------------------------------------- metadata


def test_meta_records_code_version_and_config(tmp_path):
    meta = write_run_meta(tmp_path, CFG, device="cpu", steps_per_epoch=10)
    on_disk = json.loads((tmp_path / "meta.json").read_text())

    assert on_disk == meta
    assert on_disk["experiment"] == "test_run"
    assert on_disk["steps_per_epoch"] == 10
    assert on_disk["config"]["train"]["lr"] == 2.0e-4
    assert on_disk["environment"]["torch"]
    assert "started_at" in on_disk
    # Run inside this repo, so git state must be present rather than merely attempted.
    assert on_disk["git"]["available"] is True
    assert len(on_disk["git"]["commit"]) == 40


def test_config_snapshot_is_reloadable(tmp_path):
    """The snapshot is what you re-run from, so it must survive a YAML round trip."""
    write_run_meta(tmp_path, CFG)
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == dict(CFG)


def test_git_state_reports_unavailable_outside_a_repo(tmp_path):
    assert git_state(tmp_path).get("available") is False


def test_dirty_file_paths_are_reported_verbatim(tmp_path):
    """Provenance that names the wrong file is worse than none. porcelain v1 puts the
    status in the first two columns, so the path must be sliced, not stripped."""
    run = __import__("subprocess").run

    def git(*args):
        return run(["git", *args], cwd=tmp_path, capture_output=True, check=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "README.md").write_text("a\n")
    (tmp_path / "keep.py").write_text("a\n")
    git("add", "-A")
    git("commit", "-qm", "init")

    (tmp_path / "README.md").write_text("b\n")  # modified: status is " M"
    (tmp_path / "new.py").write_text("c\n")  # untracked: status is "??"

    state = git_state(tmp_path)
    assert state["dirty"] is True
    assert state["dirty_files"] == ["README.md", "new.py"]


def test_environment_is_json_serialisable():
    json.dumps(environment("cpu"))


# -------------------------------------------------------------------- metrics


def test_metrics_jsonl_is_one_object_per_validation(tmp_path):
    append_metrics(tmp_path, {"epoch": 1, "traversability_mIoU": 0.31})
    append_metrics(tmp_path, {"epoch": 2, "traversability_mIoU": 0.42})
    rows = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2]
    assert rows[1]["traversability_mIoU"] == 0.42


def test_metrics_accepts_numpy_scalars(tmp_path):
    np = __import__("numpy")
    append_metrics(tmp_path, {"epoch": 1, "mIoU": np.float32(0.5)})
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["mIoU"] == 0.5
