"""Every tool in the commissioning chain takes its tree root from `repo_root()`.

A tool whose root is only `Path(__file__)...` runs against the checkout it lives in and
nowhere else; commissioning a second site then means a second copy of the code
(FTI, 2026-09-10). `SYNCAI_ROOT` is the one override, read in one place.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FILES = sorted(
    list((REPO / "tools" / "commissioning").glob("*.py"))
    + list((REPO / "tools" / "site30k").glob("*.py"))
    + [REPO / "scripts" / "propose_zones.py", REPO / "scripts" / "campaign_site30k.py"]
)
DERIVED = re.compile(r"^(?:ROOT|REPO|_REPO)\s*=\s*Path\(__file__\)", re.M)  # no override


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_the_root_is_read_through_repo_root(path):
    text = path.read_text()
    assert not DERIVED.search(text), (
        f"{path.relative_to(REPO)} derives its root from its own file with no override; "
        'write ROOT = Path(os.environ.get("SYNCAI_ROOT", Path(__file__).resolve().parents[2]))'
    )


def test_the_override_is_the_environment(monkeypatch, tmp_path):
    from syncai_hydranet.paths import repo_root

    monkeypatch.delenv("SYNCAI_ROOT", raising=False)
    assert repo_root(tmp_path) == tmp_path
    monkeypatch.setenv("SYNCAI_ROOT", str(tmp_path / "site2"))
    assert repo_root(tmp_path) == (tmp_path / "site2").resolve()
