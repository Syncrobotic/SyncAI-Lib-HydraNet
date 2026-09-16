"""Visibility is image evidence, not a post-hoc selection of small depth errors."""

import numpy as np
import pytest
from PIL import Image

from syncai_bev3d.floor_review import floor_height_consistency, visible_floor_mask
from syncai_bev3d.opening_controls import source_identity
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, GroundPlane


def test_occupied_walkable_area_and_visible_floor_keep_separate_denominators():
    height = np.zeros((20, 30))
    height[:, 15:] = 0.8  # chairs/people over physically walkable floor
    walk = np.ones_like(height, bool)
    visible = walk.copy()
    visible[:, 15:] = False
    arrays = {"height": height, "geom_ok": walk}
    report = floor_height_consistency(arrays, walk, visible=visible)
    assert report["walkable_area"]["p95_abs_height_m"] == 0.8
    assert report["visible_floor"]["p95_abs_height_m"] == 0
    assert report["visible_floor"]["raw_selected_pixels"] == 300
    assert report["walkable_area"]["raw_selected_pixels"] == 600
    assert not report["independent_metric_accuracy"]
    height[:, :15] = 0.5
    changed = floor_height_consistency(arrays, walk, visible=visible)
    assert changed["visible_floor"]["p95_abs_height_m"] == 0.5
    assert changed["visible_floor"]["raw_selected_pixels"] == 300


def test_missing_or_invalid_visible_evidence_does_not_become_zero_error():
    arrays = {"height": np.full((10, 10), np.nan), "geom_ok": np.ones((10, 10), bool)}
    walk = np.ones((10, 10), bool)
    assert floor_height_consistency(arrays, walk)["visible_floor"] is None
    assert (
        floor_height_consistency(arrays, walk, visible=walk)["visible_floor"][
            "p95_abs_height_m"
        ]
        is None
    )
    with pytest.raises(ValueError, match="subset"):
        floor_height_consistency(arrays, ~walk, visible=walk)


def test_visible_floor_regions_are_bound_to_image_and_intersect_walkable(tmp_path):
    folder = tmp_path / "runs/commission01/sample/masks"
    folder.mkdir(parents=True)
    cf = CameraFile(
        camera_id="sample",
        image_size_px=(64, 48),
        camera=Camera(45, 45, 32, 24),
        plane=GroundPlane(2.8, 0.7),
        plate_file="plate.png",
        mask_files={"walkable": "sample/masks/walkable.png"},
    )
    cf.save(tmp_path / "runs/commission01/sample.camera.json")
    Image.new("RGB", (64, 48)).save(tmp_path / "plate.png")
    walk = np.zeros((48, 64), np.uint8)
    walk[24:] = 255
    Image.fromarray(walk).save(folder / "walkable.png")
    review = {
        "schema": 1,
        "camera": "sample",
        "image_size_px": [64, 48],
        "source_identity": source_identity(tmp_path, "sample"),
        "reviewer": "test",
        "evidence": "synthetic visible floor",
        "regions_px": [[[10, 10], [50, 10], [50, 40], [10, 40]]],
    }
    selected = visible_floor_mask(tmp_path, cf, review)
    assert selected.any() and not selected[:24].any()
    Image.new("RGB", (64, 48), "white").save(tmp_path / "plate.png")
    with pytest.raises(ValueError, match="stale"):
        visible_floor_mask(tmp_path, cf, review)
