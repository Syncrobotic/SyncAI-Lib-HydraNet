"""Exercise source-resolution scoring through the real standalone CLI."""

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from syncai_bev3d.render_provenance import sha256
from syncai_hydranet.data.trans10k import GLAZING_CLASSES
from syncai_hydranet.models.glazing import GlazingHead


def test_source_resolution_cli_excludes_duplicates_and_checks_source(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    buffer = io.BytesIO()
    Image.new("RGB", (13, 7), (120, 120, 120)).save(buffer, format="PNG")
    data = buffer.getvalue()
    rows = [{"image": {"path": "door.png", "bytes": data}, "mask": {"bytes": data}}] * 2
    source = tmp_path / "validation-000.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    features = cache / "validation.features.npy"
    np.save(features, np.zeros((2, 64, 4, 8), dtype=np.float16))
    records = [
        {
            "image": "door.png",
            "sha256": hashlib.sha256(data).hexdigest(),
            "region": [0, 8, 64, 34],
            "source_size": [13, 7],
            "excluded_duplicate_of": None,
        },
        {
            "image": "door.png",
            "sha256": hashlib.sha256(data).hexdigest(),
            "excluded_duplicate_of": {"image": "door.png", "split": "validation"},
        },
    ]
    (cache / "validation.images.json").write_text(json.dumps(records))
    spec = {
        "classes": list(GLAZING_CLASSES),
        "input_size": [64, 64],
        "sources": {str(source): sha256(source)},
        "splits": {"validation": {"features_sha256": sha256(features)}},
    }
    manifest = cache / "manifest.json"
    manifest.write_text(json.dumps(spec))
    head = GlazingHead()
    for parameter in head.parameters():
        parameter.data.zero_()
    head.layers[-1].bias.data[1] = 10
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "spec": {"encoder": spec, "cache_manifest_sha256": sha256(manifest)},
            "epoch": 1,
            "state_dict": head.state_dict(),
        },
        checkpoint,
    )
    out = tmp_path / "evaluation.json"
    command = [
        sys.executable,
        "tools/commissioning/evaluate_glazing.py",
        "--checkpoint",
        str(checkpoint),
        "--cache",
        str(cache),
        "--validation",
        str(source),
        "--out",
        str(out),
        "--threads",
        "1",
    ]
    root = Path(__file__).resolve().parents[1]
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    result = json.loads(out.read_text())
    assert result["unique_validation_images"] == 1
    assert result["trained"]["iou"][1] == 1
    assert result["trained"]["confusion"][1][1] == 13 * 7
    assert result["images"][0]["door_predicted_pixels"] == 13 * 7
    out.unlink()
    source.write_bytes(b"changed parquet")
    failed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "validation source differs" in failed.stderr
    assert not out.exists()
