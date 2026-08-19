"""255 is defined once, and this is what keeps it that way.

`labels.py` exists because the value was defined four times and written as a literal in
ten more — its own docstring counts "fourteen places, no check". It then reached five of
them. The two scripts it names by name, `sam3_prelabel.py` and `annotation_batch.py`, kept
their private `IGNORE = 255`; `preprocessing.PAD_LABEL` was a second central definition of
the same number under a second name; and the sentinel comparisons in the evaluator, the
schema, the visualiser and the label maps stayed literal.

**Why a check and not a convention.** `labels.py` says it precisely: the mask-file contract
and the loss's `ignore_index` are "two different systems agreeing on one number, which is
exactly the kind of agreement that rots quietly — nothing crashes when they disagree, the
loss simply starts treating unlabelled pixels as a trainable class and every metric stays
plausible." A second definition does not have to be *wrong* to be a defect; it has to be
*separate*, because separate is what lets one of them move.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from syncai_hydranet.labels import IGNORE
from syncai_hydranet.preprocessing import PAD_LABEL

REPO = Path(__file__).resolve().parent.parent
SOURCES = sorted(REPO.glob("src/syncai_hydranet/**/*.py")) + sorted(REPO.glob("scripts/*.py"))
DEFINITION = REPO / "src" / "syncai_hydranet" / "labels.py"


def test_the_alias_is_the_same_value():
    """`PAD_LABEL` is a name for `IGNORE`, not a second opinion about 255."""
    assert PAD_LABEL is IGNORE == 255


def test_nothing_else_assigns_255_to_a_sentinel_name():
    """A module-level `IGNORE = 255` or `PAD_LABEL = 255` outside labels.py is a fork."""
    offenders = []
    for f in SOURCES:
        if f == DEFINITION:
            continue
        for node in ast.parse(f.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            if node.value.value != 255:
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(re.search(r"IGNORE|VOID|PAD_LABEL|UNLABEL", n) for n in names):
                offenders.append(f"{f.relative_to(REPO)}:{node.lineno} {names}")
    assert not offenders, (
        "these define the ignore sentinel again instead of importing it from "
        "syncai_hydranet.labels:\n  " + "\n  ".join(offenders)
    )


def test_the_modules_that_mask_with_it_import_it():
    """The places that compare against the sentinel, checked by import rather than by eye.

    Not a ban on the digits 255 — the palettes are full of them and always will be. This
    asserts the other direction: every module that *does* mask or fill with the sentinel
    got there through the one definition.
    """
    expected = {
        "src/syncai_hydranet/engine/evaluator.py",
        "src/syncai_hydranet/utils/visualize.py",
        "src/syncai_hydranet/data/label_maps.py",
        "src/syncai_hydranet/data/label_maps_retail_objects.py",
        "src/syncai_hydranet/config_schema.py",
        "src/syncai_hydranet/models/losses.py",
        "src/syncai_hydranet/models/hydranet.py",
        "src/syncai_hydranet/geometry/bev.py",
        "src/syncai_hydranet/cli/annotation.py",
        "src/syncai_hydranet/preprocessing.py",
        "src/syncai_hydranet/engine/confusion.py",
        "scripts/sam3_prelabel.py",
        "scripts/annotation_batch.py",
    }
    missing = [
        rel
        for rel in sorted(expected)
        if "import IGNORE" not in (REPO / rel).read_text(encoding="utf-8")
    ]
    assert not missing, f"these stopped importing the sentinel they mask with: {missing}"
