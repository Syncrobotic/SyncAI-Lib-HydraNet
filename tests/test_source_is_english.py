"""Keep implementation comments, docstrings and identifiers readable in English.

Localized UI strings, annotation class names and generated user reports are data.
The user requested Traditional Chinese outputs; banning every CJK code point had
incorrectly rejected those outputs as if they were implementation comments.
Python is parsed so localization does not require file exemptions or escaped text.
Non-Python configuration and source files retain the original whole-file check.
Tracked files only: this checks the same source set as CI.
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import tokenize
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


def _implementation_text(text: str, suffix: str):
    if suffix not in {".py", ".pyi"}:
        return list(enumerate(text.splitlines(), 1))
    rows = [
        (token.start[0], token.string)
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type in {tokenize.COMMENT, tokenize.NAME}
    ]
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                rows.append((node.body[0].lineno, doc))
    return rows


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_cjk_in_implementation_prose(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pytest.skip("not utf-8 text")
    hits = [
        f"{i}: {line.strip()[:70]}"
        for i, line in _implementation_text(text, path.suffix)
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


@pytest.mark.parametrize(
    "source",
    [
        "# " + "\u5b57",
        '"""' + "\u5b57" + '"""',
        'def f():\n    """' + "\u5b57" + '"""\n    pass',
        "\u5b57" + " = 1",
    ],
)
def test_implementation_guard_rejects_comments_docs_and_names(source):
    assert any(CJK.search(text) for _, text in _implementation_text(source, ".py"))


def test_localized_output_is_data_but_inline_comments_are_still_checked():
    source = 'label = "' + "\u5b57" + '" # English explanation\n'
    assert not any(CJK.search(text) for _, text in _implementation_text(source, ".py"))
    source += "# " + "\u5b57"
    assert any(CJK.search(text) for _, text in _implementation_text(source, ".py"))
