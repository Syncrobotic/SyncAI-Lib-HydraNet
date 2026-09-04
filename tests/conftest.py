"""Session-level hooks. Currently one: a ceiling on how much of the suite skips.

**Why a suite that asserts everything about itself needs this.** Several guards here
skip on an artefact the repository deliberately does not ship -- `runs/`, `datasets/`,
`weights/` are all gitignored -- and each of those skips is reasoned and reported by
`-ra`. That is honest, and it is also the shape a defect hides in: a path typo turns a
running test into a permanent skipper, and nothing distinguishes the two. It has already
happened once. `test_indoor25_baseline.py` guarded on a *relative* `datasets/coco/...`
path, which resolves against the invocation directory, so from anywhere but the repo
root it skipped and never ran -- found by reading, on 2026-09-04, not by a failure.

So the count is held rather than the reasons. A ceiling cannot say which skip is
legitimate, but it can say that the number stopped being small, which is the signal a
reader would otherwise have to notice for themselves. Same discipline as
`scripts/ty_ratchet.sh`: a bound that may fall and must not rise quietly.

The number is deliberately loose enough to survive a normal environment difference (no
ffmpeg, no GPU, no COCO on disk) and tight enough that a whole file falling out is
visible. Raise it only with a reason, and prefer fixing the skip.
"""

from __future__ import annotations

import pytest

# What skips on a clean checkout with no gitignored artefacts: the `runs/`- and
# `datasets/`-guarded checks, plus the depth model. Measured on this box on 2026-09-04:
# 0 skipped with everything present, 8 with ffmpeg removed from PATH. CI installs ffmpeg
# as of the same day, so its own figure is the artefact-guarded set alone.
MAX_SKIPPED = 25


def pytest_terminal_summary(terminalreporter, config):
    """Fail the session if more of it skipped than the ceiling allows.

    Runs after the report so the skip reasons `-ra` prints are on screen above this
    message -- the number alone would send a reader looking for what it counted.
    """
    if config.getoption("collectonly", default=False):
        return
    skipped = len(terminalreporter.stats.get("skipped", []))
    if skipped > MAX_SKIPPED:
        terminalreporter.write_line("")
        terminalreporter.write_line(
            f"ERROR: {skipped} tests skipped, ceiling is {MAX_SKIPPED}. "
            "A skip that is not deliberate is a test that stopped running without "
            "failing; the reasons are listed above. Fix the guard, or raise "
            "MAX_SKIPPED in tests/conftest.py with a reason.",
            red=True,
            bold=True,
        )
        pytest.exit(f"too many skipped tests: {skipped} > {MAX_SKIPPED}", returncode=1)
