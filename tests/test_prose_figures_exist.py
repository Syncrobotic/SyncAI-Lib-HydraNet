"""A figure named in prose has to be a figure a reader can open.

`tests/test_deleted_docs_are_cited_as_history.py` polices `docs/` paths for exactly this
reason -- "a path is an instruction, and the reader concludes the repository is wrong about
itself" -- and nothing policed `assets/`. Two dangled: `docs/PLAN.md` cited
`assets/gt_cam01_track8_cut.png` and `assets/gt_th03_podium.png` as openable figures
backing the track-fragmentation argument, and both are gitignored and untracked, so on any
fresh clone both are broken links.

**The arguments survived, which is why the fix was to reword rather than to publish.**
Every number those two figures were cited for is inline beside the citation -- 211/227/
242/424, "thirteen fragments", 58 s -- so the attribution stays honest without the file.
And publishing them was not available: `assets/` is an allowlist, the frames are unaudited
customer shop floor, and `tests/test_figures_are_audited.py` requires a passing verdict
before a figure may be committed.

**Prefixes are not paths.** Prose legitimately writes `assets/demo_` and
`assets/commission_mesh_` to name a family of outputs; only something with a file
extension is a claim that one specific file is there to open.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PROSE = ("docs/PLAN.md", "README.md", "CONTRIBUTING.md")

# `CONTRIBUTING.md` teaches the two-step for adding a figure and needs a name that is
# deliberately not there. Anything else claiming to be a file must be one.
EXAMPLES = {"assets/my_new_figure.png"}

_PATH = re.compile(r"assets/[A-Za-z0-9_][A-Za-z0-9_.\-]*\.[A-Za-z0-9]{2,4}\b")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)


pytestmark = pytest.mark.skipif(
    _git("rev-parse", "--git-dir").returncode != 0,
    reason="not a git checkout (sdist or vendored copy)",
)


def _cited() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name in PROSE:
        text = (REPO / name).read_text(encoding="utf-8")
        for m in sorted(set(_PATH.findall(text))):
            if m not in EXAMPLES:
                found.setdefault(m, []).append(name)
    return found


def test_the_scan_finds_the_figures_that_are_there():
    """The empty-set shape: a regex that matches nothing passes every test below."""
    cited = _cited()
    assert cited, "no assets/<file> path found in any prose file -- has the regex rotted?"


def test_every_figure_named_in_prose_is_tracked():
    tracked = set(_git("ls-files", "assets/").stdout.split())
    missing = {p: where for p, where in _cited().items() if p not in tracked}
    assert not missing, (
        "these are written as openable paths and are not in the repository, so they are "
        f"broken links on any fresh clone: {missing}. Either publish the figure (add it "
        "to the .gitignore allowlist, with the audit verdict `test_figures_are_audited` "
        "requires) or reword the citation to name what produced it, keeping the numbers "
        "inline where they already are."
    )


def test_the_contributing_example_is_still_only_an_example():
    """If somebody ever commits it, the exemption above stops being an exemption."""
    tracked = set(_git("ls-files", "assets/").stdout.split())
    assert not (EXAMPLES & tracked), (
        f"{EXAMPLES & tracked} is exempted here as a documentation example and is now a "
        "real tracked file. Remove it from EXAMPLES."
    )
