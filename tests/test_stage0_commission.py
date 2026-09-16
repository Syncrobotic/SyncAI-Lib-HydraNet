"""The one-camera Stage 0 chain: the README's steps, the bootstrap rule, the site root."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/commissioning/stage0_commission.py"


@pytest.fixture(scope="module")
def tool():
    return runpy.run_path(str(TOOL))


def test_the_slot_is_the_archive_name_or_the_files_utc_mtime(tool, tmp_path):
    named = tmp_path / "FTI-SMT-cam912_archive_20260910-112151_fti.mp4"
    named.write_bytes(b"")
    assert tool["slot_from_clip"](named) == "20260910-112151"
    plain = tmp_path / "nvr9_ch12.mp4"
    plain.write_bytes(b"")
    assert len(tool["slot_from_clip"](plain)) == len("20260910-112151")


def _calib(tmp_path, **fields):
    p = tmp_path / "cam.calib.json"
    d = {
        "pitch_deg": 39.6,
        "height_dav2_raw_m": 3.1,
        "scale_source": "unmeasured",
        "flags": ["scale_unmeasured"],
    }
    d.update(fields)
    p.write_text(json.dumps(d))
    return p


def test_no_floor_is_a_refusal_and_bootstrap_needs_permission(tool, tmp_path):
    with pytest.raises(SystemExit, match="2"):
        tool["bootstrap_unmeasured"](_calib(tmp_path, pitch_deg=None), allow=True)
    with pytest.raises(SystemExit, match="3"):
        tool["bootstrap_unmeasured"](_calib(tmp_path), allow=False)


def test_the_bootstrap_is_flagged_and_a_measured_scale_is_left_alone(tool, tmp_path):
    p = _calib(tmp_path)
    assert tool["bootstrap_unmeasured"](p, allow=True).startswith(
        "dav2_metric_indoor_raw_bootstrap"
    )
    d = json.loads(p.read_text())
    assert d["scale"] == 1.0 and d["height_m"] == 3.1
    assert tool["BOOTSTRAP_FLAG"] in d["flags"] and "scale_unmeasured" not in d["flags"]
    p = _calib(tmp_path, scale_source="person_height_median_vs_1.7m_prior_n38", flags=[])
    assert (
        tool["bootstrap_unmeasured"](p, allow=False) == "person_height_median_vs_1.7m_prior_n38"
    )
    assert "scale" not in json.loads(p.read_text())


def test_the_plan_is_the_readme_chain_ending_in_a_freeze(tool, tmp_path):
    steps = tool["plan"](
        tmp_path,
        "cam",
        Path("datasets/fti_clips"),
        Path("datasets/fti_static"),
        tmp_path / "out",
    )
    labels = [label for label, _ in steps]
    assert labels == [
        "plate",
        "person boxes",
        "onboard",
        "zones",
        "review bundle",
        "masks",
        "extras",
        "depth completion",
        "scene3d",
        "scene overlay",
        "scene mesh",
        "freeze",
    ]
    freeze = dict(steps)["freeze"]
    bundle = dict(steps)["review bundle"]
    assert bundle[-1] == "runs/commission_review/cam_batch"
    stamped = dict(tool["plan"](tmp_path, "cam", Path("c"), Path("p"), None, "20260910-102225"))
    assert stamped["review bundle"][-1] == "runs/commission_review/cam_20260910-102225"
    assert freeze[1].endswith("stage0_baseline.py") and str(tmp_path) in freeze
    assert len(tool["plan"](tmp_path, "cam", Path("c"), Path("p"), None)) == 11


def test_dry_run_prints_the_chain_without_touching_the_tree(
    tool, tmp_path, capsys, monkeypatch
):
    clip = tmp_path / "nvr9_ch12.mp4"
    clip.write_bytes(b"")
    monkeypatch.setitem(tool, "ROOT", tmp_path)
    assert tool["main"](["cam", "--clip", str(clip), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("[") == 11 and "[scene mesh]" in out
    assert not (tmp_path / "datasets").exists()
