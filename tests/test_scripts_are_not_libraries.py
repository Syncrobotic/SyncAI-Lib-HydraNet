"""A script that another script imports is a module in the wrong place.

`analytics/clip_tracks.py` was written to end one instance of this and says why in its own
docstring: `retail_flow.py` had become the de-facto library for three other scripts, which
reached it with a `sys.path` insert. The cost is not the import. It is that the shared code
sits **outside the wheel, outside the type ratchet and outside the coverage floor** —
`[tool.coverage.run] source` is `src/syncai_hydranet` and nothing else — so the thing every
caller depends on is the thing nothing checks. That is also how four copies of one loop
came to disagree about whether to correct the lens, which changed which observations a
tracker linked, under the mining run that concluded "none of the 48 spans is a posture".

`scripts/` is 13,619 lines against `src/`'s 21,585 (2026-08-20) and sits at 7%
coverage. A coverage floor was the obvious guard and is the wrong one: most of these are one-shot research tools that
legitimately have no tests, and a gate that demands them gets deleted. This measures the
thing that has actually gone wrong twice instead.

**Tracked files only.** This measures what the gate that blocks a merge measures; an
untracked script is not in the repository yet and its author is still writing it.

**A ratchet, upward only.** The existing pairs are not a backlog this test is demanding be
cleared; some are deliberate — `live_view_orin` imports `bench_camera_orin` because both are
copied to the board standalone, which `tests/test_orin_standalone_copies.py` exists to keep
honest. The downward half is omitted on purpose, unlike `test_config_depth_ratchet.py`: that
one measures configs, which are stable, and this set is in flux while `scripts/` is being
worked on, so a spurious "you may lower the baseline" failure would be the more likely one.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _tracked_scripts() -> list[Path]:
    """The scripts git knows about, which is what the gate that blocks a merge sees.

    Scanning the filesystem instead was the first version and it was wrong in the way this
    repository keeps writing about: an untracked file mid-edit tripped the ratchet, so the
    local suite went red for work that CI -- which checks out a commit -- cannot see. A
    local gate stricter than the remote one teaches people the suite is flaky, which is the
    mirror of the ruff-pin drift that let an unformatted Markdown block reach `dev`.

    An untracked script becomes visible here the moment it is `git add`ed, which is the
    first point its author can act on it and before it can reach anyone else.
    """
    out = subprocess.run(
        ["git", "ls-files", "scripts/*.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        pytest.skip("not a git checkout (sdist or vendored copy)")
    return [REPO / line for line in out.stdout.split() if line.endswith(".py")]


# Measured 2026-08-19. Lower it when a pair is moved into the package; never raise it.
# 10 -> 9 same day: the plate-calibration pipeline moved to
# `geometry/plate_calibration.py`, which removed `calibrate_from_plate ->
# fit_camera_from_people` (and kept `onboard_camera` from becoming the 11th pair).
# 9 -> 8: the day/night pixel test moved to `syncai_bev3d/teachers/photometry.py`, which removed
# `sam3_product_coverage -> sam3_person_boxes`. It was two copies of one formula, one
# gating and one reporting, and the reporting copy's docstring named the other instead of
# importing it.
# 6 -> 5: `propose_zones -> calibrate_from_plate` was pure indirection -- the script it
# imported `undistort_image` from had itself imported it from
# `geometry/plate_calibration.py`, and carried a comment saying other scripts re-imported
# it. One hop removed, no code moved.
# 8 -> 6: the SAM 3 teacher moved to `syncai_bev3d/teachers/sam3.py`, which removed both
# `-> sam3_prelabel` pairs. `column_camera_sweep._prelabel` went with them: it loaded the
# same script *by path* to stay out of this count, and said in its own docstring that a
# third caller wanting `segment` was the signal to move it into the package. That caller
# arrived.
BASELINE_PAIRS = 5


def _script_to_script_imports() -> list[tuple[str, str]]:
    """Every `scripts/a.py` that imports `scripts/b.py` by bare module name."""
    tracked = _tracked_scripts()
    modules = {f.stem for f in tracked}
    found: list[tuple[str, str]] = []
    for f in sorted(tracked):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:  # a script mid-edit is not this test's business
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [(f.name, a.name) for a in node.names if a.name in modules]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module in modules:
                found.append((f.name, node.module))
    return found


def test_no_new_script_becomes_a_library_for_another():
    pairs = _script_to_script_imports()
    assert len(pairs) <= BASELINE_PAIRS, (
        f"{len(pairs)} script-to-script imports, up from {BASELINE_PAIRS}:\n"
        + "\n".join(f"  {a} -> {b}" for a, b in pairs)
        + "\n\nWhatever is being shared belongs in `src/syncai_hydranet`, where the wheel, "
        "the type ratchet and the coverage floor reach it. "
        "`analytics/clip_tracks.py` is the worked example: four scripts kept their own copy "
        "of one loop, the copies disagreed about lens correction, and the disagreement "
        "changed which observations the tracker linked."
    )


def test_the_measurement_still_finds_the_pairs_it_was_written_against():
    """A guard whose search returns nothing passes forever. Pin two known pairs.

    The two it was written against -- both `-> sam3_prelabel` -- are gone, into
    `syncai_bev3d/teachers/sam3.py`, which is the outcome this test exists to produce. Re-pinned
    on the two remaining kinds rather than deleted, because the *forms* are what the scan
    can lose: `bev_demo -> bev_page` is `from x import Y` and `stable_infer ->
    flicker_baseline` is a plain `import x`, and an AST walk that stops seeing one of
    those goes quiet rather than red.

    `live_view_orin -> bench_camera_orin` is deliberately not pinned here even though it
    is the most durable pair in the list: it is exempt for a reason
    (`tests/test_orin_standalone_copies.py`), and a pin would read as approval.
    """
    pairs = _script_to_script_imports()
    assert pairs, "the scan found no imports at all — has scripts/ moved?"
    assert ("bev_demo.py", "bev_page") in pairs
    assert ("stable_infer.py", "flicker_baseline") in pairs
