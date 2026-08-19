"""No checkpoint is unpickled outside `load_checkpoint`.

`utils/checkpoint.py` opens with the policy: "Checkpoints get shared between machines and
downloaded from release pages, so every load in this package goes through
`load_checkpoint` instead." It was true of `src/` and not of `scripts/`, where
`offline_tracks.py` passed `weights_only=False` explicitly and two more scripts called
`torch.load` bare.

Why a source grep rather than a runtime check. The default for `weights_only` moved to
True in torch 2.6, and this project floors at torch>=2.1 -- so a bare `torch.load` means
"arbitrary code execution, depending on which torch happens to be installed", and a test
that ran against the pinned torch would pass while the packaged floor stayed exposed.
What has to be pinned is the *source*, which is also what a reviewer reads.

`scripts/robot/` is exempt: it is the quadruped line, standalone by design and on its way
out of this repository.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ROOTS = (REPO / "src", REPO / "scripts")
EXEMPT = (REPO / "scripts" / "robot",)

# The definition itself, and the docstring that explains what not to do.
ALLOWED = {REPO / "src" / "syncai_hydranet" / "utils" / "checkpoint.py"}

TORCH_LOAD = re.compile(r"^(?!\s*#).*\btorch\.load\s*\(", re.MULTILINE)


def _sources() -> list[Path]:
    out = []
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            if any(exempt in path.parents for exempt in EXEMPT) or path in ALLOWED:
                continue
            out.append(path)
    return out


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_module_unpickles_a_checkpoint_itself(path: Path):
    hits = TORCH_LOAD.findall(path.read_text(encoding="utf-8"))
    assert not hits, (
        f"{path.relative_to(REPO)} calls torch.load directly. Use "
        f"`syncai_hydranet.utils.checkpoint.load_checkpoint`, which pins "
        f"weights_only=True and explains the refusal when a file needs more than that."
    )


def test_the_policy_covers_something():
    """A guard whose search set is empty passes forever and protects nothing."""
    assert len(_sources()) > 50
