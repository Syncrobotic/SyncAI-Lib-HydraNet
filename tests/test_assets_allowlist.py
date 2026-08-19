"""`assets/` is an allowlist, and this is what stops it drifting back into a denylist.

The finding this exists for: `.gitignore` ignored `assets/*.mp4`, `*.mov` and `*.pdf`,
and its own comment said the point was that "`git add -A` in a hurry should not be able
to publish a client's floor plan". Meanwhile `assets/retail_static_plate_cam01.png` --
a 1920x380 strip cut from a customer store's camera 01 -- sat untracked in the working
tree, matching none of the three patterns.

A denylist of formats cannot express "customer footage stays out", because the format is
whichever one the renderer happened to write. An allowlist can, so the test below asserts
the shape rather than the contents: every tracked figure is still addable, and a *new*
file is not, whatever it is called.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)


pytestmark = pytest.mark.skipif(
    _git("rev-parse", "--git-dir").returncode != 0,
    reason="not a git checkout (sdist or vendored copy)",
)


def _ignored(relpath: str) -> bool:
    """`git check-ignore` exits 0 when the path *is* ignored, 1 when it is not."""
    return _git("check-ignore", "-q", relpath).returncode == 0


def test_every_tracked_figure_is_still_addable():
    """The allowlist has to name each figure, or `git add` on it silently does nothing."""
    tracked = _git("ls-files", "assets/").stdout.split()
    assert tracked, "no figures tracked under assets/ -- has the directory moved?"
    still_ignored = [p for p in tracked if _ignored(p)]
    assert not still_ignored, (
        "these are committed but now ignored, so an edit to them cannot be staged "
        f"without -f: {still_ignored}. Add a `!` line for each in .gitignore."
    )


@pytest.mark.parametrize(
    "name",
    [
        # The one that got past the old denylist, and the formats a frame grab lands in.
        "retail_static_plate_cam01.png",
        "Taichung-cam01_frame_004212.jpg",
        "shopper_dwell.jpeg",
        "aisle_walkthrough.gif",
        "fixture_plan.pdf",
        "cam09_night.mp4",
        "cam09_night.webp",
    ],
)
def test_a_new_asset_is_ignored_by_default(name: str):
    """Whatever it is called and whatever format it is in, it needs `git add -f`."""
    assert _ignored(f"assets/{name}"), (
        f"assets/{name} would be picked up by `git add -A`. assets/ must stay an "
        "allowlist -- ignore `assets/*` and name the figures back in individually."
    )


def test_the_allowlist_names_exactly_what_is_tracked():
    """Drift in either direction is a defect.

    A `!` line for a figure that has since been deleted leaves a hole with that exact
    filename's shape in it -- `assets/eprep_sonar_vs_depth.png` was removed with the
    quadruped-line cleanup and its line outlived it by two commits. A tracked figure with
    no line cannot be staged without `-f`, which is the other half of the same mistake.
    """
    listed = {
        line[len("!assets/") :]
        for line in (REPO / ".gitignore").read_text().splitlines()
        if line.startswith("!assets/")
    }
    tracked = {p.split("/", 1)[1] for p in _git("ls-files", "assets/").stdout.split()}
    assert listed == tracked, (
        f"only in .gitignore: {sorted(listed - tracked)}; "
        f"only tracked: {sorted(tracked - listed)}"
    )
