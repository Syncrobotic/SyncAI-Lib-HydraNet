"""Source comments are English, and this is what keeps them so.

Two scripts carried complete Traditional Chinese docstrings, comments and console output,
and one declared it deliberately in a header: "this file's prose, comments and output are
Traditional Chinese; the full-width punctuation is correct typesetting, not a typo", with
`# ruff: noqa: RUF001, RUF002, RUF003` to silence the ambiguous-unicode rules. That is an
honest way to hold a position and it is not the project's position, so both files were
translated and both exemptions removed.

**Not a rule about characters.** It is about who can maintain a file. A comment explains a
decision to whoever changes the code next, and this repository's comments carry most of
what it knows -- the measurements behind a default, the failure a check exists for. A
reader who cannot read them gets the code without any of that, which is the part that took
the longest to learn.

Docs are exempt. `docs/` is prose for people, and where ARCHITECTURE.md quotes the product
ask in the language it was asked in, that quotation is evidence of what was said -- not an
instruction to whoever maintains the code.

Tracked files only, for the reason `test_scripts_are_not_libraries.py` gives: this measures
what the gate blocking a merge measures, and an untracked file is still being written.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# CJK ideographs, plus kana. Latin-1 accents and the en dash are not what this is about.
#
# Written as escapes rather than as the characters themselves, and that is not a style
# preference. Spelled literally this line is CJK source text inside the file that scans
# for CJK source text, so the scan matched itself and
# `test_no_cjk_in_source[test_source_is_english.py]` failed on the guard rather than on
# anything the guard is for. `test_no_file_re_exempts_itself` below had already hit the
# same trap and worked around it by skipping this file by path, noting "Built rather
# than written out: the literal would match this file's own source" -- one function
# later, and only for its own pattern.
#
# A guard that cannot be run against itself has a blind spot exactly its own size, so
# the fix is for this file to be English like the ones it polices rather than to be
# exempted from itself. The ranges are unchanged: U+3040-30FF kana, U+3400-4DBF and
# U+4E00-9FFF ideographs, U+F900-FAFF compatibility ideographs.
CJK = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ROOTS = ("src", "scripts", "tests", "tools", "configs", "deploy")
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".json", ".cff"}


def _sources() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", *ROOTS], cwd=REPO, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        return []
    return [
        REPO / line
        for line in out.stdout.splitlines()
        if Path(line).suffix in SUFFIXES and (REPO / line).is_file()
    ]


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_cjk_in_source(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pytest.skip("not utf-8 text")
    hits = [
        f"{i}: {line.strip()[:70]}"
        for i, line in enumerate(text.splitlines(), 1)
        if CJK.search(line)
    ]
    assert not hits, (
        f"{path.relative_to(REPO)} carries non-English source text:\n  "
        + "\n  ".join(hits[:10])
        + "\n\nComments are how this repository states what it measured and why a default "
        "is that number. A maintainer who cannot read them gets the code without any of it."
    )


def test_the_scan_covers_the_files_it_was_written_for():
    """A guard whose search set is empty passes forever."""
    names = {p.name for p in _sources()}
    assert {"stable_infer.py", "flicker_baseline.py", "hydranet_retail_security.yaml"} <= names
    assert len(_sources()) > 150


def test_no_file_re_exempts_itself_from_the_ambiguous_unicode_rules():
    """`# ruff: noqa: RUF001` was the marker on both translated files. It is how a file
    announces it is about to stop being English, so it is worth failing on directly."""
    # Built rather than written out: the literal would match this file's own source.
    marker = re.compile(r"ruff:\s*noqa:.*RUF" + "00" + "[123]")
    offenders = [
        str(p.relative_to(REPO))
        for p in _sources()
        if p.suffix == ".py"
        and p != Path(__file__).resolve()
        and marker.search(p.read_text("utf-8"))
    ]
    assert not offenders, f"these exempt themselves from the unicode rules: {offenders}"
