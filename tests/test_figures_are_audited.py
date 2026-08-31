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
footage. Anything else tracked in `assets/` is out of scope because nothing here produced
it and there is no render behind it to re-derive a blurred region from -- not because it
was excused. `test_renderers_blur.py` is what covers that case, by asking of every
renderer that can write into `assets/` whether it could produce an unblurred figure at
all; `cctv_v1.gif` is the figure that made both necessary.
"""

from __future__ import annotations

import ast
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


# ------------------------------------------------------------------ the recipe, not just the result

# The parser is read out of the source with `ast` rather than imported. `demo_video`
# builds a model and pulls in torch at import, and a guard that heavy is one somebody
# deletes; `argparse`'s `_actions` is also a private attribute and a poor thing to hang a
# gate on. The string literals passed to `add_argument` are the public surface.
DEMO_VIDEO = "tools/commissioning/demo_video.py"


def _argument_dests(path: str) -> set[str]:
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    dests = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        # Positional arguments only, and only the first. Sweeping every string literal
        # in the call would collect `metavar="MODEL_JSON"`, `action="store_true"` and the
        # help text and derive flags from them.
        explicit = [
            k.value.value
            for k in node.keywords
            if k.arg == "dest" and isinstance(k.value, ast.Constant)
        ]
        if explicit:
            dests.add(explicit[0])
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            dests.add(first.value.lstrip("-").replace("-", "_"))
    return dests


def test_the_parser_is_readable_from_the_source():
    """The scan that everything below depends on, pinned so it cannot go quiet.

    An `ast` walk that stops matching -- the parser moved into a function, the calls
    became `parser.add_argument`, the flags moved to a table -- would return an empty set
    and every assertion below would pass about nothing.
    """
    dests = _argument_dests(DEMO_VIDEO)
    assert {"camera", "clip", "checkpoint", "staff_colours", "metre_scale"} <= dests, dests


@pytest.mark.parametrize("figure", _figures())
def test_a_figure_records_every_argument_its_render_was_given(figure: str):
    """A figure whose recipe is unrecorded is a figure nobody can re-cut correctly.

    The defect this was written against, 2026-08-29: a second session re-rendered
    `demo_Taichung-cam10` without `--staff-colours` and produced a figure with per-track
    identity colours and no staff legend -- undoing `65a6b78`, which is the commit that
    made that figure two-colour on purpose. The audit passed, the blur passed, the
    verdict was written, and **nothing in it named the flag**, because the record was a
    hand-maintained list of eight fields against a parser of ten.

    `--metre-scale` was missing on the same principle and is worse: `positions` are
    written in metres and two runs at 1.0 and 0.8824 produce different numbers under
    identical-looking provenance. So the assertion is over the parser rather than over a
    list of flags somebody has to remember to extend.
    """
    v = _verdict(figure)
    args = v.get("render_args")
    assert args, (
        f"{figure}'s verdict has no `render_args`. Re-cut it with `demo_gif.py`, which "
        "copies the render's own arguments out of the track log."
    )
    missing = sorted(_argument_dests(DEMO_VIDEO) - set(args))
    assert not missing, (
        f"{figure} was rendered before these arguments were recorded, or they were "
        f"dropped from the record: {missing}. A flag that is not in the verdict is a "
        "flag the next person to re-cut this figure cannot know to pass."
    )


@pytest.mark.parametrize("figure", _figures())
def test_a_published_figure_was_cut_with_the_staff_colours(figure: str):
    """The narrower half, and it encodes a decision rather than a rule of nature.

    The user removed the third "unknown" colour on 2026-08-28 and both published figures
    are staff-blue / customer-green. Without this, dropping `--staff-colours` from a
    re-cut is a silent revert -- which is exactly what happened.

    **The day a figure is legitimately published without it, change this line rather than
    delete it.** That day exists: `analytics.staff.require_camera` refuses Tao-Hsin-cam04
    at 0.417, so a figure from that camera cannot carry staff colours and its verdict
    would have to say so deliberately.
    """
    args = _verdict(figure).get("render_args") or {}
    assert args.get("staff_colours"), (
        f"{figure} was cut from a render with no staff model. Both published figures are "
        "staff-blue / customer-green by decision; re-cut with `--staff-colours "
        "runs/staff_model01/model_<camera>.json` (Taichung-cam10 also needs "
        "`--staff-min-accuracy 0.85`, its measured 0.874 against the 0.90 floor)."
    )


@pytest.mark.parametrize("figure", _figures())
def test_a_staff_coloured_figure_records_the_accuracy_it_prints(figure: str):
    """The path is not identity, and the number on the picture has to exist somewhere.

    `render_args.staff_colours` is `runs/staff_model01/model_<camera>.json`, and `runs/`
    is gitignored and regenerable -- the same path can be refitted with a different
    accuracy and no verdict would move. Meanwhile the figure's own legend prints
    `0.87 held out here`, so until this landed **the artefact displayed a number that
    nothing recorded**.

    `min_accuracy_required` is in there too, because Taichung-cam10 ships under an
    explicit exception (0.874 measured against a 0.90 floor, `--staff-min-accuracy
    0.85`), and an exception nobody can see in the record is an exception nobody can
    review.
    """
    v = _verdict(figure)
    if not (v.get("render_args") or {}).get("staff_colours"):
        pytest.skip(f"{figure} was not cut with staff colours")
    model = v.get("staff_model")
    assert model, f"{figure} names a staff model in its arguments and records nothing about it"
    for field in ("sha256", "accuracy", "held_out", "held_out_n", "min_accuracy_required"):
        assert model.get(field) is not None, f"{figure}'s staff_model has no {field}"
    assert model["held_out"] == v["camera"], (
        f"{figure}'s staff model was held out on {model['held_out']}, not on "
        f"{v['camera']} -- `require_camera` should have refused this render"
    )
    assert model["accuracy"] >= model["min_accuracy_required"]
