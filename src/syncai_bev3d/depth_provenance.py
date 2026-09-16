"""Bind reusable unscaled depth to the image and preprocessing that produced it."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from syncai_bev3d.render_provenance import sha256


def plate_contract(cf, plate_path: Path) -> dict:
    with Image.open(plate_path) as image:
        size = list(image.size)
    return {
        "plate_sha256": sha256(plate_path),
        "plate_size_px": size,
        "calibrated_image_size_px": list(cf.image_size_px),
        "preprocessing": {
            "kind": "centred-division-undistort-uncropped-v1",
            "lens": asdict(cf.lens) if cf.lens else None,
            "code_sha256": sha256(Path(__file__).with_name("plate_calibration.py")),
        },
    }


def inference_record(cf, plate_path, depth_path, *, model, revision) -> dict:
    """Record a depth inference performed by this process, not a guessed origin."""
    if not model or not revision:
        raise ValueError("inference provenance needs model and revision")
    record = {
        "schema": 1,
        "status": "bound",
        "source": "inference",
        **plate_contract(cf, plate_path),
        "depth_sha256": sha256(depth_path),
        "model": model,
        "revision": revision,
        "scope": "content provenance, not depth or metric accuracy",
    }
    # Use the serialized form so tuple-valued lens fields compare across JSON loads.
    return json.loads(json.dumps(record))


def validate_plate_contract(record, cf, plate_path):
    expected = json.loads(json.dumps(plate_contract(cf, plate_path)))
    if record.get("schema") != 1 or record.get("status") != "bound":
        raise ValueError("unsupported or unbound depth manifest")
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"stale depth manifest: {key} differs")
    if not record.get("model") or not record.get("revision"):
        raise ValueError("bound depth manifest needs recorded model and revision")


def load_depth_record(manifest_path, cf, plate_path, depth_path) -> dict:
    record = json.loads(manifest_path.read_text())
    validate_plate_contract(record, cf, plate_path)
    if record.get("depth_sha256") != sha256(depth_path):
        raise ValueError("stale depth manifest: depth bytes differ")
    depth = np.load(depth_path, allow_pickle=False)
    if depth.shape != tuple(record["plate_size_px"][::-1]):
        raise ValueError("depth manifest dimensions do not match depth")
    return record
