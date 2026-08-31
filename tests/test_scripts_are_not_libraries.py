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

**`tools/` is measured too, and for the same reason word for word.** The argument is
about *where the code sits*, not about the directory's name: `tools/` is outside the wheel
(`[tool.hatch.build.targets.wheel]` names `src/` only) and outside the coverage floor,
exactly as `scripts/` is. `service_zones.py` and `footprints_from_masks.py` reach into
`zones_confirm.py` for `_font`, `_plate` and `_to_px` -- *private* names, by a bare module
import that only resolves because Python puts the entry script's own directory on
`sys.path`.

Grouped **per directory**, because `tools/` has subdirectories and a bare `import boxes`
in `tools/site30k/` can only ever find its own neighbour. `tools/site30k/recipe.py`
reaching `scripts/` is a different (and worse) thing that this test does not measure; it
does it with an absolute `/home/paul/...` path insert, which is its own problem.

**Tracked files only.** This measures what the gate that blocks a merge measures; an
untracked script is not in the repository yet and its author is still writing it.

**A ratchet, upward only.** The existing pairs are not a backlog this test is demanding be
cleared; one was deliberate — `live_view_orin` imported `bench_camera_orin` because both
were copied to the Jetson standalone, and `tests/test_orin_standalone_copies.py` existed to
keep the copies honest. All four went on 2026-08-28 with the board as a target. The downward half is omitted on purpose, unlike `test_config_depth_ratchet.py`: that
one measures configs, which are stable, and this set is in flux while `scripts/` is being
worked on, so a spurious "you may lower the baseline" failure would be the more likely one.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _tracked(pathspec: str) -> list[Path]:
    """The files git knows about, which is what the gate that blocks a merge sees.

    Scanning the filesystem instead was the first version and it was wrong in the way this
    repository keeps writing about: an untracked file mid-edit tripped the ratchet, so the
    local suite went red for work that CI -- which checks out a commit -- cannot see. A
    local gate stricter than the remote one teaches people the suite is flaky, which is the
    mirror of the ruff-pin drift that let an unformatted Markdown block reach `dev`.

    An untracked script becomes visible here the moment it is `git add`ed, which is the
    first point its author can act on it and before it can reach anyone else.
    """
    out = subprocess.run(
        ["git", "ls-files", pathspec],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        pytest.skip("not a git checkout (sdist or vendored copy)")
    return [REPO / line for line in out.stdout.split() if line.endswith(".py")]


# Lower it when a pair is moved into the package; never raise it. The history of this
# number is in `git log -p`.
BASELINE_PAIRS = 3


def _sibling_imports(pathspec: str) -> list[tuple[str, str]]:
    """Every tracked `a.py` that imports a **sibling** `b.py` by bare module name.

    Sibling and not "anywhere under the pathspec": a bare `import boxes` resolves against
    the entry script's own directory, so `tools/site30k/boxes.py` and a hypothetical
    `tools/pose/boxes.py` are two unrelated names and pairing them would be a false find.
    """
    tracked = _tracked(pathspec)
    by_dir: dict[Path, set[str]] = {}
    for f in tracked:
        by_dir.setdefault(f.parent, set()).add(f.stem)
    found: list[tuple[str, str]] = []
    for f in sorted(tracked):
        modules = by_dir[f.parent]
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


def _script_to_script_imports() -> list[tuple[str, str]]:
    """Every `scripts/a.py` that imports `scripts/b.py` by bare module name."""
    return _sibling_imports("scripts/*.py")


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
    can lose: `eval_attributes -> train_attributes` is `from x import Y` and
    `stable_infer -> flicker_baseline` is a plain `import x`, and an AST walk that stops
    seeing one of those goes quiet rather than red. (The from-form witness was
    `bev_demo -> bev_page` until the robot dashboard scripts left in the 2026-08-25
    cleanup.)

    """
    pairs = _script_to_script_imports()
    assert pairs, "the scan found no imports at all — has scripts/ moved?"
    assert ("eval_attributes.py", "train_attributes") in pairs
    assert ("stable_infer.py", "flicker_baseline") in pairs


# ---------------------------------------------------------------- the same rule, `tools/`

# Lower it when a pair is moved into `src/`; never raise it. The three that remain:
#
#   footprints_from_masks -> zones_confirm    `_font`, `_plate`, `_to_px`, `PALETTE`
#   service_zones         -> zones_confirm    `_font`
#   heads_video           -> demo_video
#
# The first two are the ones with a name on them. `zones_confirm.py` is a confirm-sheet
# renderer and three underscore-prefixed names in it are now the drawing convention every
# commissioning sheet shares -- which makes them an interface, and an interface spelled
# with a leading underscore is one nobody can change without breaking a caller they were
# never told about.
BASELINE_TOOL_PAIRS = 3


def _tool_to_tool_imports() -> list[tuple[str, str]]:
    return _sibling_imports("tools/*.py")


def test_no_new_tool_becomes_a_library_for_another():
    pairs = _tool_to_tool_imports()
    assert len(pairs) <= BASELINE_TOOL_PAIRS, (
        f"{len(pairs)} tool-to-tool imports, up from {BASELINE_TOOL_PAIRS}:\n"
        + "\n".join(f"  {a} -> {b}" for a, b in pairs)
        + "\n\n`tools/` is outside the wheel, the type ratchet and the coverage floor for "
        "exactly the same reasons `scripts/` is, so whatever is being shared belongs in "
        "`src/`. It resolves at all only because Python puts the entry script's directory "
        "on `sys.path`, which means it works when the tool is run and not when anything "
        "else imports it."
    )


def test_the_tools_measurement_still_finds_the_pairs_it_was_written_against():
    """Same guard as above: a scan that goes quiet passes forever.

    The from-form is what `tools/` has left, and it is the one worth holding here:
    `from zones_confirm import _font, _plate, _to_px` reaches for another module's
    *private* names, and it is the one an AST walk that stopped reading `ImportFrom`
    would silently lose. Two of them are pinned, with different callers.

    **The bare-import witness in `tools/` is gone, and deliberately not replaced.** It
    was `scene_overlay -> scene_mesh`, and it left when the scene build moved into
    `syncai_bev3d`; pinning a pair only to keep a form represented would be pinning the
    debt this test exists to remove. The `scripts/` sibling above holds `import x` on a
    real pair, and both directories run through the same `_sibling_imports`, so the form
    is still witnessed.
    """
    pairs = _tool_to_tool_imports()
    assert pairs, "the scan found no imports at all -- has tools/ moved?"
    assert ("footprints_from_masks.py", "zones_confirm") in pairs
    assert ("heads_video.py", "demo_video") in pairs
