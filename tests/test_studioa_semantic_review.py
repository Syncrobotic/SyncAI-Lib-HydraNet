"""AI semantic corrections must not rewrite evaluation or become instances."""

import copy
import json

import numpy as np
import pytest
from PIL import Image

from _studioa_fixture import source_package
from syncai_hydranet.data.studioa_instances import export_instances
from syncai_hydranet.data.studioa_review import digest, write_json
from syncai_hydranet.data.studioa_semantic_review import (
    REVIEW_SCHEMA,
    apply_correction,
    export_review,
)
from syncai_hydranet.data.studioa_supervision import (
    CLASSES,
    IGNORE,
    StudioAPartialDataset,
    check_supervision,
    export_supervision,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)


def correction():
    return {
        "reason": "AI inspected an opaque panel and a fixed window interior",
        "quarantine_classes": ["glass_panel"],
        "positive_interiors": [
            {
                "entity": "glass_panel",
                "scope": "semantic_interior",
                "reason": "fixed pane core, avoiding frame and door",
                "polygon_xy": [[30, 25], [38, 25], [38, 30], [30, 30]],
            }
        ],
    }


def package(tmp_path):
    source_package(tmp_path / "annotations")
    root = tmp_path / "source"
    export_supervision(tmp_path / "annotations", root)
    m = check_supervision(root)
    frame = m["frames"][0]
    item = {
        **correction(),
        "frame_id": frame["id"],
        "image_sha256": digest(root / frame["image"]),
        "mask_sha256": digest(root / frame["mask"]),
    }
    review = {
        "schema": REVIEW_SCHEMA,
        "reviewer_kind": "ai",
        "held_out": "Tao-Hsin",
        "source_manifest_sha256": digest(root / "manifest.json"),
        "frames": [item],
    }
    path = tmp_path / "review.json"
    write_json(path, review)
    return root, path, review


def test_quarantine_and_interiors_preserve_other_positives_and_input():
    target = np.full((32, 40), IGNORE, dtype=np.uint8)
    target[:5, :5] = CLASSES["glass_panel"]
    target[26:29, 32:35] = CLASSES["person"]
    before = target.copy()
    result, stats = apply_correction(target, correction())
    np.testing.assert_array_equal(target, before)
    assert (result[:5, :5] == IGNORE).all()
    assert (result[26:29, 32:35] == CLASSES["person"]).all()
    assert result[25, 30] == CLASSES["glass_panel"]
    assert stats["quarantined_pixels"] == 25
    assert stats["preserved_other_positive_pixels"] == 9
    assert stats["added_positive_pixels"] == 45


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [40, 0], [30, 30]],
        [[0.5, 0], [30, 0], [30, 30]],
        [[0, 0], [1, 1], [2, 2]],
        [[0, 0], [1, 1]],
    ],
)
def test_bad_polygon_rejected(points):
    item = correction()
    item["positive_interiors"][0]["polygon_xy"] = points
    with pytest.raises(ValueError, match="polygon"):
        apply_correction(np.full((32, 40), IGNORE, dtype=np.uint8), item)


def test_conflicting_positive_interiors_rejected():
    item = correction()
    other = copy.deepcopy(item["positive_interiors"][0])
    other["entity"] = "door"
    item["positive_interiors"].append(other)
    with pytest.raises(ValueError, match="contradictory"):
        apply_correction(np.full((32, 40), IGNORE, dtype=np.uint8), item)


def test_derivative_replays_preserves_evaluation_and_restricts_consumers(tmp_path):
    root, review, _ = package(tmp_path)
    before = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    out = tmp_path / "corrected"
    report = export_review(root, review, out)
    m = check_supervision(out)
    assert report["corrected_train_frames"] == 1
    assert list(m["folds"]) == ["Tao-Hsin"]
    for name, sha in before.items():
        assert digest(root / name) == sha
    for frame in m["frames"]:
        if m["folds"]["Tao-Hsin"]["assignments"][frame["id"]] != "train":
            assert digest(out / frame["mask"]) == digest(root / frame["mask"])
    StudioAPartialDataset(out, "Tao-Hsin", "train")
    with pytest.raises(ValueError, match="audited fold"):
        StudioAPartialDataset(out, "Kaohsiung", "train")
    with pytest.raises(ValueError, match="instance annotations"):
        export_instances(out, review, tmp_path / "instances")
    # Even updating the output hash cannot make an arbitrary mask match review replay.
    changed = m["frames"][0]["mask"]
    Image.new("L", (40, 32), 255).save(out / changed)
    report["outputs"][changed] = digest(out / changed)
    write_json(out / "report.json", report)
    with pytest.raises(ValueError, match="replay"):
        check_supervision(out)


@pytest.mark.parametrize("kind", ["source", "image", "mask", "val", "duplicate"])
def test_bad_bindings_or_nontrain_edits_fail_before_output(tmp_path, kind):
    root, path, review = package(tmp_path)
    item = review["frames"][0]
    if kind == "source":
        review["source_manifest_sha256"] = "foreign"
    elif kind in ("image", "mask"):
        item[f"{kind}_sha256"] = "foreign"
    elif kind == "val":
        m = json.loads((root / "manifest.json").read_text())
        f = m["frames"][1]
        item.update(
            frame_id=f["id"],
            image_sha256=f["image_sha256"],
            mask_sha256=digest(root / f["mask"]),
        )
    else:
        review["frames"].append(copy.deepcopy(item))
    write_json(path, review)
    out = tmp_path / "bad"
    with pytest.raises(ValueError):
        export_review(root, path, out)
    assert not out.exists()
