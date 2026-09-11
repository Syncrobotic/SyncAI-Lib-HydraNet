"""Content identities for reproducible figures, including uncommitted scene changes."""

from __future__ import annotations

import hashlib
from pathlib import Path

from syncai_hydranet.geometry.camera_json import CameraFile


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scene_code(root: Path) -> dict[str, str]:
    paths = set((root / "src/syncai_bev3d").rglob("*.py"))
    paths.update(
        root / p
        for p in (
            "tools/commissioning/demo_video.py",
            "src/syncai_hydranet/utils/face_blur.py",
        )
    )
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths) if p.is_file()}


def capture(root: Path, camera: str, clip: Path, run: Path, checkpoint: str) -> dict:
    camera_path = root / f"runs/commission01/{camera}.camera.json"
    cf = CameraFile.load(camera_path)
    paths = {
        camera_path,
        clip.resolve(),
        run / "config.yaml",
        run / checkpoint,
        root / f"runs/site30k_qa/geometry_cache/{camera}.npz",
    }
    paths.update(root / "runs/commission01" / p for p in cf.mask_files.values())
    paths.update((root / f"runs/commission01/{camera}/masks").glob("*"))
    if cf.plate_file:
        paths.add(root / cf.plate_file)
    inputs = {
        str(p.relative_to(root)) if p.is_relative_to(root) else str(p): sha256(p)
        for p in sorted(paths)
        if p.is_file()
    }
    return {
        "schema": 1,
        "scene_code": scene_code(root),
        "inputs": inputs,
        "detector_config": str((run / "config.yaml").relative_to(root)),
        "detector_checkpoint": str((run / checkpoint).relative_to(root)),
        "detector_weights": "ema",
    }


def changed_inputs(root: Path, provenance: dict) -> list[str]:
    return [
        name
        for name, digest in provenance["inputs"].items()
        if not (root / name).is_file() or sha256(root / name) != digest
    ]
