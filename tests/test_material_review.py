"""Material review resolves contradictory classes without manufacturing ground truth."""

import numpy as np
import pytest
from PIL import Image

from syncai_bev3d.material_review import material_overlap, review_materials
from syncai_bev3d.opening_controls import source_identity
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, GroundPlane


def example(root):
    cf = CameraFile(
        camera_id="sample",
        image_size_px=(64, 48),
        camera=Camera(42, 42, 32, 24),
        plane=GroundPlane(2.8, 0.5),
        plate_file="plate.png",
    )
    folder = root / "runs/commission01/sample/masks"
    folder.mkdir(parents=True)
    cf.save(root / "runs/commission01/sample.camera.json")
    Image.new("RGB", cf.image_size_px).save(root / "plate.png")
    masks = {name: np.zeros((48, 64), bool) for name in ["glass", "glass_door", "door"]}
    masks["glass"][0:30, 10:55] = True
    masks["glass_door"][0:30, 25:38] = True
    masks["door"][0:30, 20:45] = True
    masks["door"][35:45, 2:8] = True
    for name, mask in masks.items():
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"{name}.png")
    review = {
        "schema": 1,
        "camera": "sample",
        "image_size_px": [64, 48],
        "source_identity": source_identity(root, "sample"),
        "reviewer": "test",
        "evidence": "synthetic facade",
        "facade_region_px": [[10, 0], [55, 0], [55, 30], [10, 30]],
        "door_regions_px": [[[28, 0], [40, 0], [40, 30], [28, 30]]],
        "occluded_regions_px": [[[29, 10], [32, 10], [32, 20], [29, 20]]],
    }
    return masks, review


def test_candidate_is_exclusive_and_keeps_unrelated_opaque_door(tmp_path):
    masks, review = example(tmp_path)
    candidate, report = review_materials(tmp_path, "sample", review)
    assert report["before"]["overlap_pixels"] > 0
    assert report["after"]["overlap_pixels"] == 0
    assert np.array_equal(candidate["door"][35:45, 2:8], masks["door"][35:45, 2:8])
    assert not (candidate["door"] & candidate["glass_door"]).any()
    assert candidate["material_ignore"].any()
    assert not (
        candidate["material_ignore"] & (candidate["glass"] | candidate["glass_door"])
    ).any()
    original = masks["glass"] | masks["glass_door"] | masks["door"]
    assert not ((candidate["glass"] | candidate["glass_door"]) & ~original).any()
    assert "not independent" in report["scope"]
    assert material_overlap(masks)["overlap_pixels"] > 0  # inputs unchanged


def test_source_change_cannot_reuse_polygon_review(tmp_path):
    _, review = example(tmp_path)
    Image.new("RGB", (64, 48), "white").save(tmp_path / "plate.png")
    with pytest.raises(ValueError, match="stale material"):
        review_materials(tmp_path, "sample", review)


def test_degenerate_polygon_cannot_silently_erase_a_facade(tmp_path):
    _, review = example(tmp_path)
    review["facade_region_px"] = [[1, 1], [2, 2], [3, 3]]
    with pytest.raises(ValueError, match="degenerate"):
        review_materials(tmp_path, "sample", review)
