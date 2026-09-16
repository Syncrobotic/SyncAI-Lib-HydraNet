"""Where the tree is: one answer for every tool that reads `runs/` and `datasets/`.

Every commissioning tool derived its root from its own file, so the chain could only ever
run against the checkout it lived in. Commissioning a second site (FTI, 2026-09-10) meant
copying `src/ tools/ scripts/` into a checkout-shaped directory and symlinking the
datasets the hardcoded paths named. `scene_mesh.py` already read `SYNCAI_ROOT`; now the
tools do too, through this one function, so a second site is an environment variable
rather than a second copy of the code.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV = "SYNCAI_ROOT"


def repo_root(derived: Path) -> Path:
    """`$SYNCAI_ROOT` when set, else the root the caller derived from its own location."""
    override = os.environ.get(ENV)
    return Path(override).resolve() if override else derived
