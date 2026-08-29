"""The serving path never imports `syncai_bev3d`, checked rather than remembered.

The two-package boundary (docs/PLAN.md §2): `syncai_bev3d` runs once per camera at
commissioning and may import `syncai_hydranet`'s core -- the labels contract, the runtime
geometry it writes parameters for, the one `iou`. The other direction is banned on the
serving path: at runtime, `camera.json` is the only thing that crosses. Offline modules
(`cli`, `data`, scripts, tools) may import `syncai_bev3d` freely; the modules a serving
process loads may not, because a serving image that quietly pulls in the commissioning
stack has stopped being the thin thing 96 streams were budgeted for.

An `ast` walk rather than an import of the modules themselves, so the check does not
depend on torch being importable in the test environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The modules a serving process loads. `cli` and `data` are deliberately absent: they are
# offline, and cli/scene.py imports the BEV renderer on purpose.
SERVING_PATH = ("serving", "analytics", "models", "engine", "geometry")


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_serving_path_does_not_import_bev3d():
    offenders = []
    for part in SERVING_PATH:
        for f in sorted(REPO.glob(f"src/syncai_hydranet/{part}/**/*.py")):
            hits = {m for m in _imports_of(f) if m.split(".")[0] == "syncai_bev3d"}
            if hits:
                offenders.append(f"{f.relative_to(REPO)} -> {sorted(hits)}")
    assert not offenders, (
        "serving-path modules import syncai_bev3d; the runtime crossing is camera.json, "
        "not code:\n  " + "\n  ".join(offenders)
    )


def test_bev3d_does_not_import_the_serving_stack():
    """`syncai_bev3d` may use hydranet's core, not its model/serving machinery.

    The allowed edges are named: the labels contract, the runtime geometry, the tracker's
    `iou`, and the prompt tables. Importing `models`, `engine` or `serving` from the
    commissioning package would mean the teacher pipeline grew a dependency on the
    student's training stack, which is the coupling the split exists to prevent.
    """
    allowed_prefixes = (
        "syncai_hydranet.labels",
        "syncai_hydranet.geometry",
        "syncai_hydranet.analytics.tracker",
        "syncai_hydranet.data.sam3_prompts",
    )
    offenders = []
    for f in sorted(REPO.glob("src/syncai_bev3d/**/*.py")):
        for m in _imports_of(f):
            if m.split(".")[0] != "syncai_hydranet":
                continue
            if not m.startswith(allowed_prefixes):
                offenders.append(f"{f.relative_to(REPO)} -> {m}")
    assert not offenders, (
        "syncai_bev3d imports syncai_hydranet beyond the named core edges:\n  "
        + "\n  ".join(offenders)
    )
