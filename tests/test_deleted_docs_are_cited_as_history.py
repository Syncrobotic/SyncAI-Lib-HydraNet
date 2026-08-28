"""A path to a document that does not exist reads exactly like a path to one that does.

`26dc519` deleted every document but `docs/PLAN.md` on 2026-08-25, and the source that
cited them was not updated. Counted a day later: **113 references across 60 files**, of
which 26 were written as *paths* -- `docs/RETAIL.md`, `[docs/METHODOLOGY.md](...)`, "see
docs/DEPLOY.md". Two were `--help` strings, so the tool was telling its operator to read
a file that had not existed for three days.

The distinction this test enforces is the one that decides whether a citation is honest:

* A **path** is an instruction. It says the file is there and you may open it, and when
  it is not there the reader concludes the repository is wrong about itself.
* A **`git show <commit>:<path>`** is a location. It says the file is gone, says exactly
  where it went, and is the form `README.md` and `tools/README.md` already use.
* A **bare name** -- "RETAIL.md measured 1,683 `book` at score 0.05" -- is an
  attribution. It credits a source for a number that is written out beside it, which is
  the rule this repository arrived at the hard way when the 0.847 NYUv2 scale factor
  spent two days existing nowhere because its site delegated it to a deleted script.

So paths are banned, `git show` is required to carry them, and the 69 bare attributions
are deliberately left alone: they are accurate about a historical source and they name
numbers that are inline where they are used.

**The anchors differ and that is why they are checked rather than assumed.** Most of the
set is at `b7457c2`, the commit before the documentation reset. `RESEARCH_OCCUPANCY.md`
went earlier, with the quadruped line (`cc80fc3`), and `RETAIL_SECURITY.md` earlier still
(`8b37959`). A citation pointing at the wrong anchor fails in the way this whole class
fails -- silently, at the moment somebody follows it.

pytest tests/test_deleted_docs_are_cited_as_history.py -v
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Every document that lived under `docs/` and does not any more. `PLAN.md` is absent on
# purpose: it is the one that survives, and a path to it is correct.
DELETED_DOCS = (
    "METHODOLOGY",
    "RETAIL",
    "RETAIL_DATA",
    "RETAIL_SCOPE",
    "RETAIL_SECURITY",
    "ARCHITECTURE",
    "DEPLOY",
    "DEPLOY_JETSON",
    "TRAINING_GUIDE",
    "TRAIN_MACOS",
    "RESEARCH_OCCUPANCY",
    "RELEASE",
    "EPREP",
    "BASELINES",
)

_PATH = re.compile(r"docs/(?:journal/[\w.-]*|(?:" + "|".join(DELETED_DOCS) + r")\.md)")

SUFFIXES = (".py", ".sh", ".yaml", ".yml", ".toml", ".md", ".cff", ".json")


def _tracked_text() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    return [REPO / rel for rel in out if rel.endswith(SUFFIXES)]


def _offending_lines() -> list[str]:
    """Every line naming a deleted document as a path rather than as history."""
    found = []
    for path in _tracked_text():
        rel = path.relative_to(REPO)
        if rel == Path(__file__).relative_to(REPO):  # this file names them to ban them
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if "git show" in line:
                continue
            if _PATH.search(line):
                found.append(f"  {rel}:{n}  {line.strip()[:90]}")
    return found


def test_no_deleted_document_is_cited_as_a_path():
    offenders = _offending_lines()
    assert not offenders, (
        f"{len(offenders)} citation(s) name a deleted document as a path a reader could "
        "open:\n" + "\n".join(offenders) + "\n\nUse `git show <commit>:docs/<file>` -- "
        "b7457c2 for the 2026-08-25 documentation set, cc80fc3 for RESEARCH_OCCUPANCY.md, "
        "8b37959 for RETAIL_SECURITY.md -- or re-aim at whatever now carries the content, "
        "which for most rules is a section of `docs/PLAN.md` and for several is live code. "
        "A bare name with the number written out beside it is fine and is not what this "
        "catches."
    )


def test_every_git_show_anchor_actually_holds_the_file_it_names():
    """A pointer to the wrong commit fails the same way the path did, one step later.

    Checked against the object database rather than trusted: `git cat-file -e` on each
    `<commit>:<path>` a citation names. Three anchors are in use because the documents
    died in three different commits, and getting that wrong is invisible until somebody
    runs the command.
    """
    cited = set()
    pat = re.compile(r"git show ([0-9a-f]{7,40})\^?:(\S+?)`|git show ([0-9a-f]{7,40})\^?:(\S+)")
    for path in _tracked_text():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in pat.finditer(text):
            commit = m.group(1) or m.group(3)
            target = (m.group(2) or m.group(4)).rstrip("`.,;)\"'")
            caret = "^" if f"{commit}^:" in m.group(0) else ""
            if "<" in target:  # README's `git show b7457c2:docs/<file>` template
                continue
            cited.add((f"{commit}{caret}", target))

    assert cited, "no `git show` citations found -- has the idiom changed?"

    missing = []
    for commit, target in sorted(cited):
        if target.endswith("/"):  # a directory, e.g. docs/journal/
            probe = subprocess.run(
                ["git", "-C", str(REPO), "cat-file", "-e", f"{commit}:{target.rstrip('/')}"],
                capture_output=True,
            )
        else:
            probe = subprocess.run(
                ["git", "-C", str(REPO), "cat-file", "-e", f"{commit}:{target}"],
                capture_output=True,
            )
        if probe.returncode != 0:
            missing.append(f"  git show {commit}:{target}")

    assert not missing, (
        "citations pointing at an object that is not there:\n"
        + "\n".join(missing)
        + "\n\nThe commit is usually one off: name the commit that still HAS the file, or "
        "the deleting commit with a `^`."
    )
