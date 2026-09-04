"""Sentences that were true when written and are checked against what is true now.

This project's recurring defect is a claim with nothing watching it: a docstring states
a fact about the repository, the repository moves, and the sentence stays. The audit of
2026-09-04 found four files saying no labelled site clip existed while seven sets sat in
`runs/gt_*` and `idf1` had already scored both tracker arms on one of them.

A test cannot check prose in general. It can check the specific claims that have already
rotted once, which is what this file is: each test names the sentence it guards and fails
when the tree and the sentence disagree again -- in either direction.

`runs/` is gitignored, so these skip on a clean checkout. That is the honest shape: the
claim is about what this box holds, and on a box that holds nothing there is nothing to
contradict.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GT_SETS = sorted(ROOT.glob("runs/gt_*"))

pytestmark = pytest.mark.skipif(
    not GT_SETS, reason="runs/ is gitignored; nothing here to contradict the prose"
)

CLAIMED_NO_LABELLED_CLIP = [
    "src/syncai_hydranet/analytics/reid_metrics.py",
    "src/syncai_hydranet/analytics/bytetrack.py",
    "scripts/track_review.py",
    "scripts/retail_flow.py",
]


@pytest.mark.parametrize("relpath", CLAIMED_NO_LABELLED_CLIP)
def test_no_file_still_says_a_labelled_site_clip_does_not_exist(relpath):
    """Four files said it; seven `runs/gt_*` sets say otherwise.

    Phrases rather than a regex over the whole idea: this is a tripwire on the exact
    wording that rotted, not an attempt to understand English.
    """
    text = (ROOT / relpath).read_text()
    for dead in (
        "A labelled site clip does not exist yet",
        "has never run on a site clip",
        "no site clip is labelled",
        "nobody has measured them against each other on this footage",
        "no hand-labelled site data at all",
    ):
        assert dead not in text, (
            f"{relpath} still says {dead!r}, and {len(GT_SETS)} labelled sets exist "
            f"({', '.join(p.name for p in GT_SETS)})"
        )


def test_the_idf1_measurement_the_docstrings_quote_is_still_on_disk():
    """`reid_metrics` and `bytetrack` now quote 0.7388 / 6 switches against 0.7418 / 3.

    A quoted measurement whose file is gone is the same defect one step later, so the
    numbers are read back rather than trusted.
    """
    import json

    path = ROOT / "runs/gt_cam01/idf1.json"
    if not path.is_file():
        pytest.skip("runs/gt_cam01 is not on this box")
    arms = json.loads(path.read_text())["arms"]
    single = arms["single-stage (shipped)"]
    two = arms["two-stage"]
    assert round(single["idf1"], 4) == 0.7388, "the quoted single-stage IDF1 moved"
    assert single["switches"] == 6
    assert round(two["idf1"], 4) == 0.7418, "the quoted two-stage IDF1 moved"
    assert two["switches"] == 3
