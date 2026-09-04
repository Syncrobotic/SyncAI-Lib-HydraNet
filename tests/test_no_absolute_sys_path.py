"""No file may write this developer's home directory into a path.

`tools/site30k/` put it on `sys.path`, in three files -- so those tools ran on exactly one
machine and failed on every other one before any of their own error handling could report
why. Twenty-six more held it as an absolute `ROOT`, which is softer (a `FileNotFoundError`
naming the path it wanted) but has a failure mode of its own that is worse than either:
with two checkouts on this box, a tool run from the second one reads the *first* one's
`runs/` and answers, plausibly, about the wrong tree.

Both are gone. Every one of them was two levels below the repo root, so
`Path(__file__).resolve().parents[2]` is not an approximation of what they said -- it is
the same directory, computed. The corpus assumption these tools make, that `runs/` and
`datasets/` are populated where they stand, is untouched and still fails by name.

`shipped.py` shows the pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TREES = ("src", "scripts", "tools", "tests")

# `sys.path.insert(0, "/...")` and the `append` form. A joined Path is fine -- what this
# refuses is a string literal that starts at the filesystem root.
ABSOLUTE = re.compile(r"""sys\.path\.(?:insert|append)\(\s*(?:0\s*,\s*)?["']/""")

# The home directory itself, anywhere in a line. Assembled so that this file, which has to
# name what it forbids, does not trip its own scan.
HOME = "/home/" + "paul"


def _python_files():
    for tree in TREES:
        for path in sorted((ROOT / tree).rglob("*.py")):
            # This file states the rule and therefore has to quote it; scanning itself
            # would make the guard fail on its own documentation.
            if "__pycache__" not in path.parts and path != Path(__file__).resolve():
                yield path


@pytest.mark.parametrize("path", list(_python_files()), ids=lambda p: str(p.name))
def test_no_absolute_path_reaches_sys_path(path):
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        assert not ABSOLUTE.search(line), (
            f"{path.relative_to(ROOT)}:{lineno} puts an absolute path on sys.path, so this "
            f"file imports on one machine only:\n    {line.strip()}\n"
            "Derive it: _REPO = Path(__file__).resolve().parents[N]"
        )


@pytest.mark.parametrize("path", list(_python_files()), ids=lambda p: str(p.name))
def test_no_file_hardcodes_this_developers_home(path):
    """The wider rule, once the 26 absolute `ROOT` constants were derived.

    Docstrings are exempt only where they are quoting a *payload*: `analytics/delivery.py`
    documents the JSON a client sends, which really does carry absolute paths from
    whichever machine wrote it, and `tests/test_delivery.py` builds one. A path in code is
    not exempt.
    """
    text = path.read_text()
    allowed = {"src/syncai_hydranet/analytics/delivery.py", "tests/test_delivery.py"}
    if str(path.relative_to(ROOT)) in allowed:
        return
    for lineno, line in enumerate(text.splitlines(), 1):
        assert HOME not in line, (
            f"{path.relative_to(ROOT)}:{lineno} writes out a home directory:\n"
            f"    {line.strip()}\n"
            "Derive it: ROOT = Path(__file__).resolve().parents[N]. A second checkout on "
            "the same box otherwise reads the first one's runs/ and answers about it."
        )


def test_the_check_would_catch_the_thing_it_was_written_for():
    """The pattern that was actually in the tree, so a rewrite of the regex that stops
    matching it fails here rather than passing an empty check."""
    # Assembled rather than written out, so the scan above does not flag this file for
    # holding an example of what it forbids.
    call = "sys.path" + ".insert(0, " + chr(34) + "/home/paul/x/src" + chr(34) + ")"
    assert ABSOLUTE.search(call)
    assert ABSOLUTE.search("sys.path" + ".append('/opt/whatever')")
    # ... and does not fire on the derived form that replaced it
    assert not ABSOLUTE.search("sys.path" + '.insert(0, str(_REPO / "src"))')
