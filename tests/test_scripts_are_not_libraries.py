"""A script that another script imports is a module in the wrong place.

`analytics/clip_tracks.py` was written to end one instance of this and says why in its own
docstring: `retail_flow.py` had become the de-facto library for three other scripts, which
reached it with a `sys.path` insert. The cost is not the import. It is that the shared code
sits **outside the wheel, outside the type ratchet and outside the coverage floor** —
`[tool.coverage.run] source` is `src/syncai_hydranet` and nothing else — so the thing every
caller depends on is the thing nothing checks. That is also how four copies of one loop
came to disagree about whether to correct the lens, which changed which observations a
tracker linked, under the mining run that concluded "none of the 48 spans is a posture".

`scripts/` is 10,748 lines against `src/`'s 18,364 and sits at 7% coverage. A coverage floor
was the obvious guard and is the wrong one: most of these are one-shot research tools that
legitimately have no tests, and a gate that demands them gets deleted. This measures the
thing that has actually gone wrong twice instead.

**A ratchet, upward only.** The existing pairs are not a backlog this test is demanding be
cleared; some are deliberate — `live_view_orin` imports `bench_camera_orin` because both are
copied to the board standalone, which `tests/test_orin_standalone_copies.py` exists to keep
honest. The downward half is omitted on purpose, unlike `test_config_depth_ratchet.py`: that
one measures configs, which are stable, and this set is in flux while `scripts/` is being
worked on, so a spurious "you may lower the baseline" failure would be the more likely one.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

# Measured 2026-08-19. Lower it when a pair is moved into the package; never raise it.
BASELINE_PAIRS = 10


def _script_to_script_imports() -> list[tuple[str, str]]:
    """Every `scripts/a.py` that imports `scripts/b.py` by bare module name."""
    modules = {f.stem for f in SCRIPTS.glob("*.py")}
    found: list[tuple[str, str]] = []
    for f in sorted(SCRIPTS.glob("*.py")):
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

    `sam3_person_boxes -> sam3_prelabel` is the SAM 3 loader being shared out of a script,
    and it is the clearest candidate for the package of anything in the list.
    """
    pairs = _script_to_script_imports()
    assert pairs, "the scan found no imports at all — has scripts/ moved?"
    assert ("sam3_person_boxes.py", "sam3_prelabel") in pairs
    assert ("sam3_product_coverage.py", "sam3_prelabel") in pairs
