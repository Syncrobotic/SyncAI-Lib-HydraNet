"""`§7.4` is an address, not a position, and 30-odd places in this tree dial it.

`docs/PLAN.md` is the project's single live document, and the rest of the tree cites its
sections by number -- `PLAN §7.4`, `PLAN section 7.14`, `PLAN 2.1`, all three spellings in
use. Counted 2026-08-27: **45 citations across 9 files outside `docs/`**, plus 20
cross-references inside PLAN itself. `§7.4` alone is dialled seven times, `§7.11` six.

Nothing checked that any of them resolved.

That matters most at the moment somebody reorganises the file, which is live work: §7 is
464 lines of 991 under a heading that is accurate for two of its fourteen entries, so it
will be regrouped sooner or later. Renumbering during that would break every citation
above **silently** -- a section number that no longer exists reads exactly like one that
does, and the reader who follows it finds the wrong content rather than an error. This
repository spent 2026-08-27 removing that class of rot from about seventy sites; a
renumbering would manufacture forty-five more in one commit.

**Duplicates are checked as well as dangles, and they are the worse failure.** A second
`11.` in §7 resolves fine -- `§7.11` still finds *an* entry -- so nothing dangles and the
citation quietly means two things. A dangling reference at least announces itself the
moment someone follows it.

**The failure message names the citing file and line**, because whoever trips this is
mid-edit in PLAN and not reading this file: "§7.<n> cited in analytics/events/pose.py:218
does not exist" says what to do, where "not all citations resolve" does not.

(That example writes `<n>` rather than a digit on purpose. With a real number it is a real
citation, and this file is tracked, so the scan below reads its own docstring and reports
a dangle that is only an illustration. It did exactly that on the first run after the file
was committed -- which is the scanner being right about a text that was wrong.)

What is deliberately *not* checked: that a citation points at the *right* section. Nothing
mechanical can know that. This checks only that the address exists, which is the half that
can rot without anyone touching the citing file.

pytest tests/test_plan_citations_resolve.py -v
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLAN = REPO / "docs" / "PLAN.md"

# The three spellings actually in the tree. `§` and `section` are only honoured on a line
# that also says PLAN, because neither sigil belongs to this document by right -- and that
# is not hypothetical: `cli/export_onnx.py` carries `DEPLOY.md §4`, and
# `analytics/events/pose.py` cites `runs/pose_pilot01/REPORT.md section 4-2`. Reading
# those as PLAN addresses would fail this test for citations that are perfectly correct.
_SIGIL = re.compile(r"§\s*(\d+(?:\.\d+)?)")
_SECTION = re.compile(r"sections?\s+(\d+\.\d+)")
_BARE = re.compile(r"PLAN(?:\.md)?[`\s]+(\d+\.\d+)")

_TOP = re.compile(r"^## (\d+)\.")
_SUB = re.compile(r"^### (\d+\.\d+)")
_ENTRY = re.compile(r"^(\d+)\. ")


def _plan_lines() -> list[str]:
    return PLAN.read_text(encoding="utf-8").splitlines()


def _section_seven_entries() -> list[tuple[int, str]]:
    """`(number, first line)` for every numbered entry inside `## 7.`, in page order."""
    lines = _plan_lines()
    start = next(i for i, ln in enumerate(lines) if _TOP.match(ln) and ln.startswith("## 7."))
    out = []
    for i in range(start + 1, len(lines)):
        if _TOP.match(lines[i]):
            break
        m = _ENTRY.match(lines[i])
        if m:
            out.append((int(m.group(1)), lines[i]))
    return out


def _addresses() -> set[str]:
    """Every section number a citation may legitimately name."""
    lines = _plan_lines()
    addrs = {m.group(1) for ln in lines if (m := _TOP.match(ln))}
    addrs |= {m.group(1) for ln in lines if (m := _SUB.match(ln))}
    addrs |= {f"7.{n}" for n, _ in _section_seven_entries()}
    return addrs


def _citations() -> list[tuple[str, str, int]]:
    """`(address, path, line_number)` for every PLAN citation in a tracked text file."""
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    suffixes = (".py", ".md", ".yaml", ".yml", ".toml", ".cff", ".sh", ".json")
    found: list[tuple[str, str, int]] = []
    for rel in tracked:
        if not rel.endswith(suffixes):
            continue
        path = REPO / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # a binary or unreadable tracked file
            continue
        # Inside PLAN every line is a PLAN context; outside it, the line has to say so.
        always = rel == "docs/PLAN.md"
        for n, line in enumerate(text.splitlines(), 1):
            if not (always or "PLAN" in line):
                continue
            for pat in (_SIGIL, _SECTION, _BARE):
                for m in pat.finditer(line):
                    found.append((m.group(1), rel, n))
    return found


def test_every_plan_citation_resolves_to_a_section_that_exists():
    addrs = _addresses()
    dangling = [(a, f, n) for a, f, n in _citations() if a not in addrs]
    assert not dangling, "citations naming a section docs/PLAN.md does not have:\n" + "\n".join(
        f"  §{a} cited in {f}:{n}" for a, f, n in sorted(dangling)
    )


def test_no_two_entries_in_section_seven_share_a_number():
    """The failure that resolves fine and still means two things.

    A dangling `§7.15` announces itself the moment someone follows it. A duplicated `7.11`
    does not: the citation lands on whichever entry the reader's eye reaches first, and
    both look correct. Renumbering by hand across 464 lines is exactly how a duplicate
    arrives.
    """
    seen: dict[int, str] = {}
    dupes = []
    for num, line in _section_seven_entries():
        if num in seen:
            dupes.append(f"  {num}. appears twice:\n      {seen[num][:78]}\n      {line[:78]}")
        seen[num] = line
    assert not dupes, "duplicate entry numbers in docs/PLAN.md §7:\n" + "\n".join(dupes)


def test_the_scan_finds_the_citations_it_is_written_against():
    """A guard whose search returns nothing passes forever.

    Pinned on the three spellings rather than on a count, because the count moves with
    every commit: `§7.4`, `section 7.14` and the bare `PLAN 7.10` are the forms in use,
    and an extractor that stops seeing one of them goes quiet instead of red.
    """
    cites = _citations()
    assert len(cites) > 30, f"only {len(cites)} PLAN citations found -- has the format changed?"

    outside = {(a, f) for a, f, _ in cites if not f.startswith("docs/")}
    assert any(f.endswith("README.md") for _, f in outside), "no citation found in a README"
    assert any(f.startswith("src/") for _, f in outside), "no citation found in src/"

    for spelling in ("§7.4", "section 7.14", "PLAN 7.10"):
        got = subprocess.run(
            ["git", "-C", str(REPO), "grep", "-qF", spelling], capture_output=True
        )
        assert got.returncode == 0, f"the {spelling!r} spelling has left the tree"


def test_section_seven_is_the_one_that_gets_reorganised():
    """Why this test exists at all, pinned so the reason cannot quietly stop being true.

    §7 holds the numbers everything else dials -- more citations than the rest of the
    document put together -- and it is the section under pressure to be restructured,
    because its heading says "Open questions" over a majority of closed ones. If that ever
    stops being where the citations point, this test's framing is out of date even if it
    still passes.
    """
    sevens = [a for a, _, _ in _citations() if a.startswith("7.")]
    others = [a for a, _, _ in _citations() if not a.startswith("7.")]
    assert len(sevens) > len(others) // 2, (
        f"{len(sevens)} citations into §7 against {len(others)} elsewhere -- "
        "if §7 is no longer the hot section, revisit what this file argues."
    )
