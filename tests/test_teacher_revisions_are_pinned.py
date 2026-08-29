"""A teacher loaded from `main` is a different teacher every time upstream pushes.

Every measurement this project rests on came out of a model it downloaded by name.
`from_pretrained("facebook/sam3")` resolves whatever `main` points at *today*, so an
upstream push changes what the teacher produces -- different masks, different boxes,
different metres -- under artefacts that are byte-identical in every field that names
what made them. `bandit` flags the shape as B615; the reason it matters here is not
security but **provenance**, and this repository has already paid for the difference
once, when the 0.847 NYUv2 scale factor spent two days existing nowhere because its
derivation lived in a deleted script.

The pins are the commits the local cache resolved on 2026-08-28 -- the ones every figure
already in PLAN was actually measured with -- rather than whatever is newest. Pinning
forward would have quietly re-based the existing numbers on a model nothing was measured
against.

**A branch or a tag is not a pin**, so this checks the shape as well as the presence. A
tag can be moved and a branch moves by design; only a full commit id names one artefact
for ever. That is the same argument `.github/workflows/*.yml` makes by pinning every
GitHub Action to a SHA with the tag in a trailing comment, and the same one
`.github/dependabot.yml` makes for excluding torch: "an automated bump would change the
training result silently".

What this does *not* check: that the pinned commit is the right one, or that it still
exists upstream. Nothing mechanical can know the first, and the second needs the network.
This checks that a load site cannot go back to floating, which is the half that rots
without anyone touching the file.

pytest tests/test_teacher_revisions_are_pinned.py -v
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# `from_pretrained` covers transformers' model, processor and tokeniser loaders;
# `pipeline` is the other door into the same download, used by
# `syncai_bev3d/plate_calibration.py` for Depth-Anything V2.
LOADERS = {"from_pretrained", "pipeline"}
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _tracked_python() -> list[Path]:
    """Tracked `.py` under the three trees that load models, which is what a merge sees."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "src", "scripts", "tools"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [REPO / rel for rel in out if rel.endswith(".py")]


def _load_calls() -> list[tuple[Path, ast.Call]]:
    found = []
    for path in _tracked_python():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else None
            name = name or getattr(node.func, "id", None)
            if name in LOADERS:
                found.append((path, node))
    return found


def _pinned_constants() -> dict[str, str]:
    """Every `*_REVISION` string constant defined anywhere in the three trees."""
    out: dict[str, str] = {}
    for path in _tracked_python():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("REVISION"):
                    out[f"{path.relative_to(REPO)}::{target.id}"] = str(node.value.value)
    return out


def test_every_model_load_pins_a_revision():
    missing = [
        f"  {path.relative_to(REPO)}:{node.lineno}"
        for path, node in _load_calls()
        if "revision" not in {kw.arg for kw in node.keywords}
    ]
    assert not missing, (
        "model loads that resolve whatever upstream `main` points at today:\n"
        + "\n".join(missing)
        + "\n\nPass `revision=` a full commit id. Take it from the cache that produced the "
        "numbers already recorded -- `huggingface_hub.scan_cache_dir()` reports the commit "
        "for each downloaded snapshot -- rather than from whatever is newest, or the "
        "existing measurements are silently re-based on a model none of them was made "
        "against."
    )


def test_the_pins_are_commit_ids_and_not_branch_names():
    """`revision="main"` satisfies the check above and pins nothing.

    A branch moves by design and a tag can be moved by hand; only a full 40-character
    commit id names one artefact permanently. Checked on the declared constants rather
    than on the call sites, because that is where the value a reader would edit lives.
    """
    constants = _pinned_constants()
    assert constants, "no *_REVISION constants found -- have the teachers moved?"
    loose = {name: value for name, value in constants.items() if not FULL_SHA.match(value)}
    assert not loose, (
        "revision pins that are not full commit ids:\n"
        + "\n".join(f"  {name} = {value!r}" for name, value in sorted(loose.items()))
        + "\n\nA branch moves and a tag can be moved. Use the 40-character commit."
    )


def test_the_scan_sees_the_teachers_it_is_written_against():
    """A guard whose search returns nothing passes forever.

    Pinned on the four teachers by name rather than on a count: the count moves whenever
    a loader is added, and these four are what the commissioning and labelling paths
    actually depend on -- SAM 3 and Grounding DINO produce the masks and boxes,
    Depth-Anything V2 produces the metres, ViTPose produces the pose labels.
    """
    calls = _load_calls()
    assert len(calls) >= 6, f"only {len(calls)} model loads found -- has the API changed?"

    declared = "\n".join(
        p.read_text(encoding="utf-8") for p in _tracked_python() if "REVISION" in p.read_text()
    )
    for model in (
        "facebook/sam3",
        "IDEA-Research/grounding-dino-base",
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "usyd-community/vitpose-base-simple",
    ):
        assert model in declared, f"{model} no longer sits beside a revision pin"
