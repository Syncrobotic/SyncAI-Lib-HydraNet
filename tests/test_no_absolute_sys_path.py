"""No file may put an absolute path on `sys.path`.

`tools/site30k/` did, in three files, naming `/home/paul/SyncAI-Lib-HydraNet` -- so those
tools ran on exactly one machine and failed on every other one before any of their own
error handling could report why.

**This is narrower than "no absolute paths", deliberately.** Twenty-six files still hold
an absolute `ROOT` for reading `runs/` and `datasets/`, and that is a much softer problem:
they fail at a `FileNotFoundError` that names the path it wanted. A bad `sys.path` fails
at import, on a name, and says nothing about the machine it was typed on. Only the fatal
half is a rule here; the rest is on the record in PLAN section 8 as a decision to take.

Derive the root from `__file__`; `shipped.py` shows the pattern and `tools/site30k` now
uses it.
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
