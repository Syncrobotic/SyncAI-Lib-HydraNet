"""The settings block of a report, and what it is not allowed to say.

Three scripts write a JSON report and each opened it with `{"settings": vars(args)}`.
The block earns its place -- every row below it is a threshold crossing and the
thresholds are the argument list -- but the namespace also holds filesystem paths, so
each report named an operator, a home directory, a repository checkout and a dataset
root. Nobody notices, because the file it lands in looks like output rather than like a
document.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from syncai_hydranet.analytics.delivery import report_settings, tail

HOME = "/home/paul/SyncAI-Lib-HydraNet"


def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_an_absolute_clip_path_keeps_the_camera_and_the_clip_and_nothing_above():
    """Both surviving parts are already in the report as `camera` and `session`."""
    got = tail(f"{HOME}/datasets/studioa_clips/Taichung-cam01/archive_20260816-113012_x.mp4")
    assert got == "Taichung-cam01/archive_20260816-113012_x.mp4"


def test_a_checkpoint_keeps_the_run_that_produced_it():
    """A basename alone was tried and is worse than the leak: `best.pt` names nothing,
    there are forty of them under runs/, and a report that cannot say which weights
    produced it is not auditable."""
    assert tail(f"{HOME}/runs/hydranet_retail_security_b03/best.pt") == (
        "hydranet_retail_security_b03/best.pt"
    )


@pytest.mark.parametrize(
    "value",
    [f"{HOME}/datasets/x/y.mp4", Path(f"{HOME}/runs/a/b.pt"), f"{HOME}/configs/c.yaml"],
)
def test_nothing_keeps_the_home_directory_or_the_checkout(value):
    assert "/home/" not in str(tail(value))
    assert "SyncAI-Lib-HydraNet" not in str(tail(value))


def test_a_list_of_clips_reduces_elementwise():
    got = report_settings(_args(clips=[f"{HOME}/d/cam01/a.mp4", f"{HOME}/d/cam02/b.mp4"]))
    assert got["clips"] == ["cam01/a.mp4", "cam02/b.mp4"]


def test_thresholds_are_the_point_and_pass_through_untouched():
    """Redaction that ate the numbers would remove the reason the block exists."""
    args = _args(fps=5.0, max_occupancy=2, k1=-0.225, draw=True, zone=None)
    assert report_settings(args) == {
        "fps": 5.0,
        "max_occupancy": 2,
        "k1": -0.225,
        "draw": True,
        "zone": None,
    }


def test_prose_is_not_a_path():
    """The whitespace test. A basis reading "5.2 m / 4.0 m" must survive intact."""
    assert tail("5.2 m / 4.0 m of a counting line") == "5.2 m / 4.0 m of a counting line"


def test_a_bare_name_is_left_alone():
    assert tail("best.pt") == "best.pt"
    assert tail("cam01/best.pt") == "cam01/best.pt"


def test_extra_facts_append():
    """retail_flow passes the tracker's simplifications, which bound every row."""
    got = report_settings(_args(fps=5.0), simplifications=["one", "two"])
    assert got == {"fps": 5.0, "simplifications": ["one", "two"]}


def test_the_reports_that_prompted_this_all_use_it():
    """A helper two of three callers adopted is the situation that produced the copies
    of `frames()` and of `source_fps`. Grep, because the alternative is remembering."""
    repo = Path(__file__).resolve().parent.parent
    for name in ("site_events", "retail_flow", "mine_fall_candidates"):
        source = (repo / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert "report_settings(" in source, f"{name}.py builds its settings block by hand"
        assert "vars(args)" not in source, f"{name}.py still writes the raw namespace"
