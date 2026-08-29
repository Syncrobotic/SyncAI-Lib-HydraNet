"""Every published figure of a store carries the verdict that licensed it.

`tools/commissioning/demo_gif.py` re-runs the detector on the source frames at 0.03 and
requires every person it finds to fall inside a region the render blurred. That check
found a real defect the first time it ran -- **132 of 954 person boxes over 120 frames of
Tao-Hsin-cam04 had a head the render had left readable**, one of them two shoppers at a
shelf with a face plainly recognisable -- so it is not a formality, and a figure that
reached `assets/` without it is a figure nobody checked.

The verdict is a tracked file beside the figure rather than a directory under `runs/`,
and that was a correction: `runs/` is gitignored, so a guard asserting something there
would pass on the box that rendered the figure and could only skip in CI, where the
directory has never existed. It would be green exactly where nobody needs it.

**The verdict is written whether the audit passed or failed.** A file that only appears
on success makes `failing == 0` true by construction, and this test would then be green
about the act of writing rather than about the frames.

**Scope is by producer, not by exemption.** `demo_gif.py` names its own output
`assets/demo_<camera>.gif`, so that pattern is exactly the set of figures cut from store
footage. `cctv_v1.gif` is a segmentation figure with no person in it and no
render behind it; it is out of scope because nothing produced it here, not because it was
excused.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from syncai_hydranet.utils.face_blur import BLUR_THR

REPO = Path(__file__).resolve().parent.parent


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)


pytestmark = pytest.mark.skipif(
    _git("rev-parse", "--git-dir").returncode != 0,
    reason="not a git checkout (sdist or vendored copy)",
)


def _figures() -> list[str]:
    return [
        p
        for p in _git("ls-files", "assets/").stdout.split()
        if p.startswith("assets/demo_") and p.endswith(".gif")
    ]


def test_there_is_at_least_one_published_figure():
    """The empty-set shape this test is most likely to fail into: no figures tracked, no
    verdicts required, everything below green and nothing checked."""
    assert _figures(), (
        "no assets/demo_*.gif is tracked. Either every store figure has been removed, or "
        "the naming changed and this whole file is now checking nothing."
    )


def test_every_figure_has_a_tracked_verdict():
    tracked = set(_git("ls-files", "assets/").stdout.split())
    missing = [p for p in _figures() if p.replace(".gif", ".audit.json") not in tracked]
    assert not missing, (
        f"published with no audit verdict beside them: {missing}. Re-cut each with "
        "`tools/commissioning/demo_gif.py <camera>`, which writes the verdict, and add "
        "a `!` line for it in .gitignore."
    )


def _verdict(figure: str) -> dict:
    """The verdict for one figure, as a clear failure rather than a traceback when absent.

    `test_every_figure_has_a_tracked_verdict` is the test that owns the missing case; the
    two below should say which figure and why, not raise FileNotFoundError from pathlib.
    """
    path = REPO / figure.replace(".gif", ".audit.json")
    assert path.exists(), (
        f"{figure} has no {path.name} beside it, so nothing says whether its faces were "
        "checked. Re-cut it with `tools/commissioning/demo_gif.py`."
    )
    return json.loads(path.read_text())


@pytest.mark.parametrize("figure", _figures())
def test_a_figures_verdict_says_no_face_was_readable(figure: str):
    v = _verdict(figure)
    assert v["person_boxes_checked"] > 0, (
        f"{figure}'s audit checked zero person boxes, so it cannot have cleared "
        "anything. A window of an empty shop is not a passed audit."
    )
    assert v["person_boxes_failing"] == 0, (
        f"{figure} was published with {v['person_boxes_failing']} of "
        f"{v['person_boxes_checked']} person boxes whose head the render left readable: "
        f"{v.get('failing_examples')}. There is no stated exception for this -- the "
        "remedy is always to blur more, and a false positive costs a blurred fixture."
    )


@pytest.mark.parametrize("figure", _figures())
def test_a_figure_was_blurred_at_least_as_hard_as_today(figure: str):
    """A lower threshold blurs more. When the standard tightens, figures published under
    the old one have to be re-rendered rather than inherited -- which is the situation
    that produced this test: the README figure was cut at 0.10, and 0.10 is the threshold
    that left 132 readable heads on another camera of the same fleet."""
    v = _verdict(figure)
    assert v["blur_score_thr"] <= BLUR_THR, (
        f"{figure} was rendered with a blur threshold of {v['blur_score_thr']} and the "
        f"current standard is {BLUR_THR}. Higher means less was blurred. Re-render it."
    )


# The scene code a figure was rendered by. Anything under these paths changes what the
# right-hand panel shows, so a figure whose verdict predates a change to them is showing
# a reconstruction the repository no longer produces.
SCENE_PATHS = (
    "tools/commissioning/demo_video.py",
    "tools/commissioning/scene_mesh.py",
    "src/syncai_bev3d/scene_mesh.py",
    "src/syncai_bev3d/floorplan.py",
    "src/syncai_bev3d/meshes.py",
    "src/syncai_bev3d/shading.py",
    "src/syncai_hydranet/utils/face_blur.py",
)


@pytest.mark.parametrize("figure", _figures())
def test_a_figure_is_not_older_than_the_code_that_drew_it(figure: str):
    """A verdict records the commit it was rendered at, and nothing was reading it.

    The defect this catches, from 2026-08-29: both README figures were cut at `ef60573`,
    and `f63c0d2` -- which stopped drawing a false wall as a misplaced merchandise run --
    landed after. The figures went on showing boxes hovering above the counters, which the
    repository had already fixed, and every other test here stayed green because the faces
    were blurred and the verdict was present. **A figure is a claim about what this code
    produces**; the moment the code moves, the claim is about a version that is gone.

    Only the paths that change what is drawn count. A README edit or a new test does not
    stale a figure, and treating every commit as staling one would make this unignorable
    noise that somebody would then ignore.
    """
    v = _verdict(figure)
    at = v.get("commit")
    if not at or _git("cat-file", "-e", at).returncode != 0:
        pytest.skip(f"{figure}'s verdict names no commit reachable here")
    changed = _git("log", "--oneline", f"{at}..HEAD", "--", *SCENE_PATHS).stdout.strip()
    assert not changed, (
        f"{figure} was rendered at {at[:7]} and the scene code has moved since:\n"
        f"{changed}\nRe-render it with `tools/commissioning/demo_video.py` and re-cut it "
        "with `demo_gif.py`, or the figure is a claim about a version that no longer "
        "exists."
    )
