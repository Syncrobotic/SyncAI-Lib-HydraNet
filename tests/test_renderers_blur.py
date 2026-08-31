"""A renderer that can put a shop floor in `assets/` blurs faces, checked rather than assumed.

`assets/cctv_v1.gif` was a figure of Tao-Hsin-cam03, tracked and published, with **no audit
verdict and no blur anywhere in it**. Its commit message says the camera was chosen partly
because it "has a person walking through it", and it had one: a shopper crossing the aisle
away from the lens for the second half of the cut, unblurred. It was deleted on 2026-08-31.

**The defect was never that file.** It was drawn by `cli/scene.py`, which had no blur stage
at all, while its two sibling renderers -- `demo_video.py` and `heads_video.py` -- blur by
construction. Deleting the figure *instead of* fixing that would have left the tool able to
make another one tomorrow, so the stage came first and the figure went afterwards. This
test is what keeps the stage.

It also closes the gap two green tests left open between them. `test_assets_allowlist.py`
holds the allowlist and the tracked set to each other, and `test_figures_are_audited.py`
holds `assets/demo_*.gif` to their verdicts. A figure under any other name satisfies the
first and falls outside the second's enumeration, so both stay green while it sits
unaudited in the seam -- which is where `cctv_v1.gif` sat. This test asks a different
question -- not "is this file audited" but "could this tool have produced it" -- and the
answer has to be no.

**`scene_overlay.py` is deliberately not in the list, and saying so is the point.** It
renders the same cameras and does not blur, but everything it writes lands under
`assets/*`, which `.gitignore` excludes; reaching the history takes a deliberate
`git add -f` plus an allowlist line, which is the moment CONTRIBUTING exists to govern. It
is named here so its absence is a decision on the record rather than an oversight waiting
to be discovered the way `cctv_v1` was.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Renderers whose output is meant for `assets/` -- that is, for publication.
PUBLISHING_RENDERERS = (
    "src/syncai_hydranet/cli/scene.py",
    "tools/commissioning/demo_video.py",
    "tools/commissioning/heads_video.py",
)

# The module that holds the arithmetic. One definition, so an auditor asking "was this
# face inside a blurred region" and the renderer that blurred it cannot drift apart.
BLUR_MODULE = "face_blur"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module)
                found |= {f"{node.module}.{a.name}" for a in node.names}
            else:  # `from . import x` / `from ..utils import face_blur`
                found |= {a.name for a in node.names}
    return found


@pytest.mark.parametrize("rel", PUBLISHING_RENDERERS)
def test_a_publishing_renderer_imports_the_blur(rel):
    """Not "has a function called blur" -- imports the one module that defines the rule.

    A renderer with its own blurring would pass a looser check and still be wrong: the
    audit re-derives the blurred region with `blur_rect`, so a second implementation is a
    figure whose verdict is computed against arithmetic that did not blur it.
    """
    path = REPO / rel
    assert path.is_file(), f"{rel} has moved; this guard now checks nothing"
    assert any(BLUR_MODULE in m for m in _imports(path)), (
        f"{rel} renders store footage and does not import `{BLUR_MODULE}`. "
        "That is how assets/cctv_v1.gif came to be published: a figure of a customer's "
        "shop floor, unblurred, with no audit verdict, from a renderer that had no blur "
        "stage. If this renderer genuinely cannot see a person, say so here and remove it "
        "from the list -- do not add a local blur."
    )


@pytest.mark.parametrize("rel", PUBLISHING_RENDERERS)
def test_a_publishing_renderer_can_be_asked_not_to_blur_but_must_be_asked(rel):
    """Blur is the default; the flag exists to be refused.

    A renderer whose blur is opt-*in* is one whose next figure is unblurred, because the
    person cutting it did not know the flag existed. `demo_video.py` states the rule in
    its own help text -- "only for a private check, never for anything shared" -- and the
    shape that enforces it is `store_true`: absent means blurred.
    """
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    flags = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert "--no-blur" in flags, (
        f"{rel} has no `--no-blur`. Either the blur is unconditional -- which is fine, and "
        "this list should say so -- or it is opt-in, which is how an unblurred figure gets "
        "cut by someone who did not know the flag was there."
    )


def test_the_blur_module_still_offers_what_an_auditor_needs():
    """`blur_rect` separate from `blur_region` is what lets the audit check the blur.

    `test_figures_are_audited.py` rests on a verdict that says every person the detector
    found fell inside a blurred region. Computing that needs the rectangle *without*
    blurring, which is why the two are separate functions and why merging them would
    quietly make the audit re-derive nothing.
    """
    src = (REPO / "src/syncai_hydranet/utils/face_blur.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"blur_rect", "blur_region"} <= names, (
        "the audit asks 'was this face inside a blurred region' by re-deriving the "
        "rectangle; blur_rect must stay callable without blurring anything"
    )


# ---------------------------------------------------------------------------
# and it actually blurs, which importing a module does not prove


class _FakeModel:
    """A detector that always finds one person, so the blur can be checked without a GPU.

    The guards above hold the *shape* -- the import, the flag. This holds the effect. A
    renderer that imports `face_blur`, takes `--no-blur`, and never calls either would
    pass everything above and publish the same figure.
    """

    def __init__(self, box, score=0.9, label=0):
        self.box, self.score, self.label = box, score, label

    def predict(self, x, score_thr):  # noqa: ARG002 -- signature is the contract
        import torch

        return {
            "detection": [
                {
                    "boxes": torch.tensor([self.box], dtype=torch.float32),
                    "labels": torch.tensor([self.label]),
                    "scores": torch.tensor([self.score]),
                }
            ]
        }


def _flat_image(w=200, h=200):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _dummy_batch():
    """A real tensor, because `blur_people` moves it to the device.

    Passing None and loosening the production code to accept it would be shaping the
    renderer around the test -- and the thing being loosened is a face blur.
    """
    import torch

    return torch.zeros(1, 3, 8, 8)


def test_blur_people_changes_the_pixels_over_a_person():
    import numpy as np

    from syncai_hydranet.cli.scene import blur_people

    base = _flat_image()
    before = np.asarray(base).copy()
    n = blur_people(
        base,
        _FakeModel(box=[40.0, 30.0, 100.0, 170.0]),
        "cpu",
        x=_dummy_batch(),
        region=(0, 0, 200, 200),
        det_names=("person", "bag"),
    )
    after = np.asarray(base)
    assert n == 1, "one person, one blurred region"
    assert not np.array_equal(before, after), "the frame came back untouched"
    # The blur covers the head band, not the whole box -- BLUR_TOP_FRACTION -- so the
    # bottom of the person must survive, or the assertion above would also pass on a
    # renderer that simply destroyed the image.
    assert np.array_equal(before[168:170, 40:100], after[168:170, 40:100]), (
        "the blur reached the feet; it is meant to cover the head band"
    )


def test_blur_people_does_nothing_when_the_vocabulary_has_no_person():
    """A config whose detection head cannot emit `person` has no faces to find.

    Returning 0 rather than raising matters: `cli/scene.py` renders configs that predict
    terrain and products only, and a crash there would be a renderer that stopped working
    the day the blur was added.
    """
    import numpy as np

    from syncai_hydranet.cli.scene import blur_people

    base = _flat_image()
    before = np.asarray(base).copy()
    n = blur_people(
        base,
        _FakeModel(box=[40.0, 30.0, 100.0, 170.0]),
        "cpu",
        x=_dummy_batch(),
        region=(0, 0, 200, 200),
        det_names=("device", "boxed_stock"),
    )
    assert n == 0
    assert np.array_equal(before, np.asarray(base))
