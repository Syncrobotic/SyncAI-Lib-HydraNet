"""The local gate and the remote gate must run one ruff -- and exactly one thing may pin it.

`.pre-commit-config.yaml` used to pin ruff by tag while `uv.lock` pinned the ruff CI runs.
They drifted -- v0.13.0 against 0.16.2 -- and the gap was not cosmetic. Somewhere between
the two, ruff learnt to format Python code blocks inside Markdown, so a commit passed the
hook and failed `ruff format --check .` in CI, on a `docs/*.md` file the author had no
reason to think ruff read at all.

**A local gate more permissive than the remote one is worse than no local gate**, because
it teaches people that CI is flaky rather than that their commit is. That is the same shape
as every other finding in this repository's own notes: a mechanism that runs, reports
success, and is not checking what its reader believes it is.

The first fix held the two pins equal and made this file check it. **That deadlocked, and
the deadlock is why the mechanism below is different.** Dependabot watches ruff through two
ecosystems -- `uv` (via `uv.lock`) and `pre-commit` (via the hook rev) -- and groups cannot
span ecosystems, so every ruff release opened two PRs, each moving one pin, each red against
an equality test on its own, and neither mergeable without the other. Measured 2026-08-28 by
running the equality test against a tree shaped like each PR: both fail, the unchanged tree
passes.

So ruff is pinned in `uv.lock` and nowhere else, and the hook runs `uv run --frozen ruff`.
There is no second version to keep level, and one ecosystem bumps it. What these tests hold
is that the second pin does not come back -- for ruff, and for anything else `uv.lock` and
a hook rev could both claim.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

# `.pre-commit-config.yaml` pseudo-repos, which legitimately carry no `rev:` because a hook
# defined in this tree has no upstream version to pin.
PSEUDO_REPOS = {"local", "meta"}


def _config() -> dict:
    """`.pre-commit-config.yaml`, read as YAML rather than with a regex.

    The difference is not tidiness. The regex this replaces was
    `- repo: (\\S+)\\n(?:\\s+.*\\n)*?\\s+rev: (\\S+)` -- a lazy repetition of a greedy
    `.*`, which is the classic catastrophic-backtracking shape. Every block in the file
    happened to be followed by a `rev:`, so it always terminated and nobody saw it.

    Add one `- repo: local` block with no `rev:` -- which is exactly what the ruff fix
    below did -- and there is no match to find, so the engine explores every way of
    splitting the remaining lines and never returns. Measured 2026-08-26 while wiring the
    type ratchet in: that function spun at 100% CPU for 35 minutes before it was killed,
    and it takes the whole suite with it.

    **A hang is strictly worse than a failure here.** A red test names itself in one line;
    this burns CI's 20-minute job timeout and reports only that the step was cancelled, on
    a config change the author has no reason to connect to a test about version pins.
    """
    return yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8"))


def _hooks_named(hook_id: str) -> list[dict]:
    """Every hook with this id, from any repo block."""
    return [
        h
        for repo in _config()["repos"]
        for h in repo.get("hooks", [])
        if h.get("id") == hook_id
    ]


def _locked_version(package: str) -> str:
    """The version `uv.lock` resolves `package` to -- what CI, and now the hook, runs."""
    text = (REPO / "uv.lock").read_text(encoding="utf-8")
    m = re.search(rf'name = "{re.escape(package)}"\nversion = "([^"]+)"', text)
    assert m, f"{package} is not in uv.lock"
    return m.group(1)


def _pinned_tool_names(repo_url: str, hook_ids: list[str]) -> set[str]:
    """The tool names a pinned hook repo could plausibly be a second pin of.

    Derived from the repo's last path segment with the `-pre-commit` / `pre-commit-`
    affixes stripped, plus the hook ids it defines. Deliberately not a substring test:
    `pre-commit/pre-commit-hooks` contains the string `pre-commit`, which the `uv` group
    watches as a PyPI package, and it is not a second pin of it.
    """
    last = repo_url.rstrip("/").rsplit("/", 1)[-1]
    names = {last, *hook_ids}
    if last.endswith("-pre-commit"):
        names.add(last[: -len("-pre-commit")])
    if last.startswith("pre-commit-"):
        names.add(last[len("pre-commit-") :])
    return names


def _uv_group_patterns() -> set[str]:
    """The package names Dependabot's `uv` ecosystem bumps, from its group patterns."""
    config = yaml.safe_load((REPO / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    for block in config["updates"]:
        if block.get("package-ecosystem") == "uv":
            return {p for g in block.get("groups", {}).values() for p in g.get("patterns", [])}
    raise AssertionError("no `uv` package-ecosystem block in .github/dependabot.yml")


def _double_pinned(patterns: set[str]) -> list[str]:
    """Hook repos that pin a `rev:` for a tool the `uv` ecosystem also bumps."""
    offenders = []
    for repo in _config()["repos"]:
        url = str(repo.get("repo", ""))
        if url in PSEUDO_REPOS or "rev" not in repo:
            continue
        ids = [str(h.get("id", "")) for h in repo.get("hooks", [])]
        clash = _pinned_tool_names(url, ids) & patterns
        if clash:
            offenders.append(f"{url} (rev {repo['rev']}) pins {', '.join(sorted(clash))}")
    return offenders


def test_the_ruff_hook_runs_the_ruff_ci_runs():
    """Same binary, not two versions someone has to keep level."""
    hooks = _hooks_named("ruff") + _hooks_named("ruff-format")
    assert len(hooks) == 2, (
        f"{len(hooks)} ruff hook(s) in .pre-commit-config.yaml, expected `ruff` and "
        "`ruff-format`. If ruff no longer runs at commit time, say so here rather than "
        "leaving a test that passes because it found nothing to check."
    )
    for hook in hooks:
        entry = str(hook.get("entry", ""))
        assert entry.startswith("uv run"), (
            f"the {hook['id']!r} hook runs {entry!r}. It has to go through `uv run` so it "
            "resolves the ruff in uv.lock -- the one CI runs. A hook that brings its own "
            "ruff is the second pin this file exists to keep out."
        )
        assert " ruff" in entry, f"the {hook['id']!r} hook does not run ruff at all: {entry!r}"

    # The single remaining pin has to exist, or `uv run --frozen ruff` resolves nothing.
    assert _locked_version("ruff")


def test_ci_runs_ruff_the_same_way():
    """The claim above is about CI, so read CI rather than assume it."""
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for command in ("uv run ruff format --check .", "uv run ruff check"):
        assert command in ci, (
            f"ci.yml no longer runs {command!r}. The pre-commit hooks are written to match "
            "CI's ruff invocation; if CI's has moved, they are no longer the same gate."
        )


def test_no_hook_repo_pins_a_tool_dependabot_bumps_through_uv():
    """Two ecosystems bidding on one tool is a deadlock, not a redundancy.

    Groups cannot span ecosystems, so the two PRs cannot be merged into one, and a test
    that holds the two pins equal makes each of them red on its own.
    """
    patterns = _uv_group_patterns()
    assert patterns, "the `uv` ecosystem block declares no group patterns"
    assert "*" not in patterns, (
        "the `uv` group now matches every package, so which hook revs are second pins "
        "cannot be decided by name any more. Re-examine the pinned repos by hand."
    )
    offenders = _double_pinned(patterns)
    assert not offenders, (
        "hook rev(s) pinning a tool Dependabot also bumps through the `uv` ecosystem:\n"
        + "\n".join(offenders)
        + "\n\nEach release of that tool will open one PR per ecosystem, and neither can "
        "carry the other's change. Run it through `uv run` instead, as the ruff hooks do."
    )


def test_the_scan_still_recognises_a_second_pin(monkeypatch):
    """Positive control: the check above passes today, so show it can still fail.

    Without this, deleting the ruff hooks -- or renaming the config -- would leave a test
    that reports success because it found nothing to look at.
    """
    reinstated = {
        "repos": [
            {
                "repo": "https://github.com/astral-sh/ruff-pre-commit",
                "rev": "v0.16.2",
                "hooks": [{"id": "ruff"}, {"id": "ruff-format"}],
            }
        ]
    }
    monkeypatch.setitem(globals(), "_config", lambda: reinstated)
    # An explicit pattern set, not the live one: this controls the scan, not the config.
    offenders = _double_pinned({"ruff"})
    assert len(offenders) == 1 and "ruff" in offenders[0], offenders


def test_pre_commit_hooks_are_under_dependabot():
    """A pin that nothing bumps cannot pick up its own fixes. That is why they drifted."""
    config = (REPO / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "package-ecosystem: pre-commit" in config, (
        "dependabot does not watch .pre-commit-config.yaml, so its pinned hook revs will "
        "drift from uv.lock again and nothing will say so."
    )
