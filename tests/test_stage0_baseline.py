"""The Stage 0 freeze carries every file `capture_inputs` identifies, code included.

`freeze()` copied `src/`, `scene_mesh.py`, `pyproject.toml` and `uv.lock` by name while
`capture_inputs` also listed `rebuild_geometry.py`; the identity check at the end of the
freeze then failed on every camera (FTI-SMT-cam912, 2026-09-16), and the runner never
reached a scene. The copy now walks the identity record, so the two cannot drift apart.
"""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
from PIL import Image

from syncai_bev3d.scene_audit import capture_inputs
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, GroundPlane

TOOL = Path(__file__).resolve().parents[1] / "tools/commissioning/stage0_baseline.py"


def _site(root: Path) -> None:
    folder = root / "runs/commission01/sample/masks"
    folder.mkdir(parents=True)
    cache = root / "runs/site30k_qa/geometry_cache/sample.npz"
    cache.parent.mkdir(parents=True)
    np.savez(cache, horiz=np.ones((48, 64)))
    CameraFile(
        camera_id="sample",
        image_size_px=(64, 48),
        camera=Camera(fx=42, fy=42, cx=32, cy=24),
        plane=GroundPlane(height=2.8, pitch=math.radians(25)),
        mask_files={"objects": "sample/masks/objects.png"},
    ).save(root / "runs/commission01/sample.camera.json")
    Image.new("L", (64, 48)).save(folder / "objects.png")
    (root / "src/syncai_bev3d").mkdir(parents=True)
    (root / "src/syncai_bev3d/scene.py").write_text("SCENE = 1\n")
    for name in ("scene_mesh.py", "rebuild_geometry.py"):
        (root / "tools/commissioning" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "tools/commissioning" / name).write_text(f"# {name}\n")
    (root / "pyproject.toml").write_text("[project]\nname = 'sample'\n")
    (root / "uv.lock").write_text("version = 1\n")


def test_the_snapshot_identifies_as_the_source_it_was_frozen_from(tmp_path):
    freeze = runpy.run_path(str(TOOL))["freeze"]
    root, out = tmp_path / "site", tmp_path / "out"
    _site(root)
    out.mkdir()
    snapshot = freeze(root, out, ["sample"])
    assert (snapshot / "tools/commissioning/rebuild_geometry.py").is_file()
    assert capture_inputs(snapshot, "sample") == capture_inputs(root, "sample")
    assert (out / "frozen-inputs.json").is_file()
