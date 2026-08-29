"""One version, three files that state it, and a check that they still agree.

`src/syncai_hydranet/__init__.py` is the single source — `pyproject.toml` reads it through
hatch's `dynamic = ["version"]`, so those two cannot disagree. `CITATION.cff` states it
again by hand, and nothing was keeping it in step: release-please's `extra-files` listed
only `__init__.py`, so the first release would have bumped one and left the other at 0.1.0
in the file whose whole job is to be quoted by someone else.

The fix is the `extra-files` entry; this test is here because of what that fix is. A
release-please generic updater matches on a `x-release-please-version` comment, and an
updater that matches nothing does not fail — it updates nothing and reports success. That
is the same shape as `unsourced_classes()` shipping with tests and no caller, and as the
`--split test` fallback that produced an official-looking circular number: a mechanism
that exists, runs, and quietly does nothing. So the result is checked, not the mechanism.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

import syncai_hydranet

REPO = Path(__file__).resolve().parent.parent
CITATION = REPO / "CITATION.cff"


def test_citation_states_the_package_version():
    stated = str(yaml.safe_load(CITATION.read_text(encoding="utf-8"))["version"])
    assert stated == syncai_hydranet.__version__, (
        f"CITATION.cff says {stated}, the package says {syncai_hydranet.__version__}. "
        "release-please should be updating both; check the `x-release-please-version` "
        "marker on CITATION.cff's version line still matches what its updater looks for."
    )


def test_the_release_please_marker_is_still_on_the_version_line():
    """The marker is what makes the updater match. Without it the file is simply skipped."""
    line = next(
        ln
        for ln in CITATION.read_text(encoding="utf-8").splitlines()
        if ln.startswith("version:")
    )
    assert "x-release-please-version" in line, (
        "CITATION.cff's version line lost its `# x-release-please-version` marker, so "
        "release-please will skip the file and the version will drift at the next release."
    )


def test_citation_is_listed_for_release_please():
    config = (REPO / "release-please-config.json").read_text(encoding="utf-8")
    assert '"CITATION.cff"' in config, (
        "CITATION.cff is not in release-please's extra-files, so nothing updates it."
    )


def test_the_citation_describes_the_product_this_repository_ships():
    """The quadruped line was removed on 2026-08-19 and this file was the last thing still
    describing the project as a quadruped one, months after pyproject and the README had
    moved. Packaging metadata is read by people who never open the code."""
    text = CITATION.read_text(encoding="utf-8").lower()
    for gone in ("quadruped", "lite3", "rk3588", "rknn"):
        assert gone not in text, f"CITATION.cff still describes the removed line: {gone!r}"
    assert re.search(r"cctv|retail", text), "CITATION.cff should name the line that ships"
