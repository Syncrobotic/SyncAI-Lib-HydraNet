"""PLAN §4.6's retention table and the sweep that enforces it, held to each other.

A policy nothing executes is the failure mode this repository has the most notes about,
and a retention policy is the worst candidate for it: the thing it governs is a customer's
face, and the failure is silent by construction -- nothing is deleted, and nothing says so.

So two halves. The first holds `scripts/retention_sweep.py`'s tiers to the table in
`docs/PLAN.md` §4.6, so a number cannot be changed in one place. The second proves the
sweep **deletes**, which the tree cannot demonstrate on its own: every file under the swept
roots is younger than the shortest tier, so a real run reports "deleted 0" today and will
until the corpus is a month old. A sweep that reports nothing deleted because nothing is
due is indistinguishable, from its output, from one whose glob matches nothing -- which is
why the fixtures below are backdated rather than waited for.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLAN = REPO / "docs" / "PLAN.md"


def _load_sweep():
    """Load the script by path, not as `scripts.retention_sweep`.

    Two reasons, and the first cost a collection error before it was noticed.
    `python -m pytest` puts the working directory on `sys.path` and `uv run pytest` --
    which is what CI runs -- does not, so the import form passes locally and fails in CI.
    `ci.yml`'s own header records that exact divergence about a different import.

    The second is the rule: `tests/test_scripts_are_not_libraries.py` exists because code
    two callers share belongs in the wheel, outside which nothing checks it. Importing a
    script as a module from a test is the first step toward it becoming one.
    """
    name = "_retention_sweep"
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "retention_sweep.py")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec, and the sweep would not load without it: it declares
    # `@dataclass` under `from __future__ import annotations`, so every annotation is a
    # string, and `dataclasses` resolves those through `sys.modules[cls.__module__]`.
    # Left unregistered that lookup returns None and the class body raises
    # `AttributeError: 'NoneType' object has no attribute '__dict__'` -- at collection,
    # far from the cause.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rs = _load_sweep()
IMAGERY, TIERS, main, sweep = rs.IMAGERY, rs.TIERS, rs.main, rs.sweep


def _plan_section() -> str:
    text = PLAN.read_text(encoding="utf-8")
    m = re.search(r"^### 4\.6 Retention.*?^(?=## 5\.)", text, re.M | re.S)
    assert m, "PLAN.md has no §4.6 Retention section; the sweep enforces a policy that is gone"
    return m.group(0)


# ---------------------------------------------------------------------------
# the policy and the code that runs it


@pytest.mark.parametrize("tier", TIERS, ids=lambda t: t.root)
def test_every_tier_the_sweep_enforces_is_written_in_the_plan(tier):
    """The number in the script and the number in the table are the same number.

    Not a style check. Someone shortening retention in the script and not the document
    leaves a paragraph promising 30 days beside code keeping 7, and the document is what
    a store or a lawyer would be shown.
    """
    section = _plan_section()
    assert tier.root in section, f"{tier.root} is swept but §4.6 does not mention it"
    assert re.search(rf"\*\*{tier.days} days\*\*", section), (
        f"§4.6 does not state {tier.days} days for {tier.root}"
    )


def test_the_plan_names_what_is_deliberately_kept():
    """The table has to say what is *not* swept, or an absence reads as an oversight."""
    section = _plan_section()
    for kept in ("datasets/studioa_static", "disposition log", "assets/"):
        assert kept in section, f"§4.6 does not say why {kept} is kept"


def test_measurements_are_not_in_the_imagery_suffixes():
    """`runs/` holds both; the tiers separate them by suffix, so the split must be real.

    Deleting `runs/**/*.json` would delete the apparatus behind every number in PLAN --
    the residual maps, the IDF1 verdicts, the coverage of a sweep. This is the assertion
    that stops a future edit widening the imagery list over them.
    """
    for suffix in (".json", ".jsonl", ".npz", ".log", ".yaml", ".pt"):
        assert suffix not in IMAGERY, f"{suffix} is a measurement and would be deleted"


# ---------------------------------------------------------------------------
# it actually deletes, which the real tree cannot show


def _tree(tmp_path: Path, monkeypatch, age_days: float, names: tuple[str, ...]) -> Path:
    """A fake repo whose files are backdated, since the real corpus is weeks old."""
    monkeypatch.setattr(rs, "REPO", tmp_path)
    old = time.time() - age_days * 86400
    for name in names:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 16)
        import os

        os.utime(p, (old, old))
    return tmp_path


def test_a_frame_past_its_tier_is_deleted(tmp_path, monkeypatch):
    """The half the live tree cannot demonstrate until the corpus is a month old."""
    tier = next(t for t in TIERS if t.root == "datasets/studioa_clips")
    _tree(tmp_path, monkeypatch, 40, ("datasets/studioa_clips/cam01/old.mp4",))

    result = sweep(tier, time.time(), tracked=set(), apply=True)

    assert result.examined == 1
    assert result.over_age == 1
    assert result.deleted == 1
    assert not (tmp_path / "datasets/studioa_clips/cam01/old.mp4").exists()


def test_a_frame_inside_its_tier_is_left_alone(tmp_path, monkeypatch):
    tier = next(t for t in TIERS if t.root == "datasets/studioa_clips")
    _tree(tmp_path, monkeypatch, 5, ("datasets/studioa_clips/cam01/new.mp4",))

    result = sweep(tier, time.time(), tracked=set(), apply=True)

    assert result.examined == 1 and result.over_age == 0 and result.deleted == 0
    assert (tmp_path / "datasets/studioa_clips/cam01/new.mp4").exists()


def test_measurements_survive_a_sweep_that_deletes_the_imagery_beside_them(
    tmp_path, monkeypatch
):
    """One directory, two fates -- the suffix split doing its job on a real layout."""
    tier = next(t for t in TIERS if t.root == "runs")
    _tree(
        tmp_path,
        monkeypatch,
        200,
        ("runs/exp01/crop.jpg", "runs/exp01/result.json", "runs/exp01/train.log"),
    )

    result = sweep(tier, time.time(), tracked=set(), apply=True)

    assert result.deleted == 1, "only the image should go"
    assert not (tmp_path / "runs/exp01/crop.jpg").exists()
    assert (tmp_path / "runs/exp01/result.json").exists()
    assert (tmp_path / "runs/exp01/train.log").exists()


def test_a_tracked_file_is_refused_even_when_over_age(tmp_path, monkeypatch):
    """`assets/` is in the history; deleting the working copy achieves nothing but a diff."""
    tier = next(t for t in TIERS if t.root == "runs")
    _tree(tmp_path, monkeypatch, 200, ("runs/exp01/published.png",))
    tracked = {tmp_path / "runs/exp01/published.png"}

    result = sweep(tier, time.time(), tracked=tracked, apply=True)

    assert result.over_age == 1
    assert result.deleted == 0
    assert result.refused_tracked == 1
    assert (tmp_path / "runs/exp01/published.png").exists()


def test_without_apply_nothing_is_deleted(tmp_path, monkeypatch):
    """The first thing anyone runs a deletion tool with is no arguments at all."""
    tier = next(t for t in TIERS if t.root == "datasets/studioa_clips")
    _tree(tmp_path, monkeypatch, 40, ("datasets/studioa_clips/cam01/old.mp4",))

    result = sweep(tier, time.time(), tracked=set(), apply=False)

    assert result.over_age == 1, "it must still report what it would remove"
    assert result.deleted == 0
    assert (tmp_path / "datasets/studioa_clips/cam01/old.mp4").exists()


def test_a_tier_that_examines_nothing_is_an_error_not_a_clean_run(
    tmp_path, monkeypatch, capsys
):
    """The failure this sweep is most likely to have and least likely to show.

    A moved root, a glob that stopped matching, a mount that did not come back: each
    reports "deleted 0", which is also what a correct run reports on a tree where nothing
    is due. `examined` is what separates them, and a zero there exits non-zero.
    """
    monkeypatch.setattr(rs, "REPO", tmp_path)
    (tmp_path / "datasets/studioa_clips").mkdir(parents=True)
    (tmp_path / "runs").mkdir()

    assert main([]) == 1
    assert "examined 0 files" in capsys.readouterr().out
