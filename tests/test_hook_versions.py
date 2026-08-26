"""The local gate and the remote gate must agree about what they enforce.

`.pre-commit-config.yaml` pins ruff by tag; `uv.lock` pins the ruff CI runs. They drifted --
v0.13.0 against 0.16.2 -- and the gap was not cosmetic. Somewhere between the two, ruff
learnt to format Python code blocks inside Markdown, so a commit passed the hook and failed
`ruff format --check .` in CI, on a `docs/*.md` file the author had no reason to think ruff
read at all.

**A local gate more permissive than the remote one is worse than no local gate**, because
it teaches people that CI is flaky rather than that their commit is. That is the same shape
as every other finding in this repository's own notes: a mechanism that runs, reports
success, and is not checking what its reader believes it is.

The root cause was that nothing bumped the hook revs -- `.github/dependabot.yml` covered
`github-actions` and `uv` and not `pre-commit`, so the pins could not float and nobody was
watching them. That is fixed there; this is the check that says so.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _hook_rev(repo_substring: str) -> str:
    r"""The `rev:` pinned for the first hook repo whose URL contains `repo_substring`.

    Read as YAML, not with a regex, and the difference is not tidiness. The regex this
    replaces was `- repo: (\S+)\n(?:\s+.*\n)*?\s+rev: (\S+)` -- a lazy repetition of a
    greedy `.*`, which is the classic catastrophic-backtracking shape. Every block in this
    file happened to be followed by a `rev:`, so it always terminated and nobody saw it.

    Add one `- repo: local` block at the end of the config -- which has no `rev:`, because
    a local hook has no version to pin -- and there is no match to find, so the engine
    explores every way of splitting the remaining lines and never returns. Measured
    2026-08-26 while wiring the type ratchet in: this function span at 100% CPU for 35
    minutes before it was killed, and it takes the whole suite with it.

    **A hang is strictly worse than a failure here.** A red test names itself in one line;
    this burns CI's 20-minute job timeout and reports only that the step was cancelled, on
    a config change the author has no reason to connect to a test about version pins.

    `pyyaml` is a hard dependency of this project, the file is already `check-yaml`ed by
    the hook set it describes, and `local`/`meta` pseudo-repos are skipped by name because
    they legitimately carry no rev.
    """
    config = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    for repo in config["repos"]:
        url = str(repo.get("repo", ""))
        if url in {"local", "meta"}:
            continue
        if repo_substring in url:
            return str(repo["rev"]).lstrip("v")
    raise AssertionError(f"no hook repo matching {repo_substring!r} in .pre-commit-config.yaml")


def _locked_version(package: str) -> str:
    """The version `uv.lock` resolves `package` to -- what CI actually runs."""
    text = (REPO / "uv.lock").read_text(encoding="utf-8")
    m = re.search(rf'name = "{re.escape(package)}"\nversion = "([^"]+)"', text)
    assert m, f"{package} is not in uv.lock"
    return m.group(1)


def test_the_ruff_hook_runs_the_ruff_ci_runs():
    hook, locked = _hook_rev("ruff-pre-commit"), _locked_version("ruff")
    assert hook == locked, (
        f".pre-commit-config.yaml pins ruff {hook}; uv.lock gives CI {locked}. "
        "Bump the hook rev to match. Until they agree, a commit can pass locally and fail "
        "in CI on a rule the local ruff does not have -- which is how a Markdown code "
        "block reached `dev` unformatted."
    )


def test_pre_commit_hooks_are_under_dependabot():
    """A pin that nothing bumps cannot pick up its own fixes. That is why they drifted."""
    config = (REPO / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "package-ecosystem: pre-commit" in config, (
        "dependabot does not watch .pre-commit-config.yaml, so its pinned hook revs will "
        "drift from uv.lock again and nothing will say so."
    )
