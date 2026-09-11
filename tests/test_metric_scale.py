"""Measured distances set scale; held-out distances only validate it."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from syncai_bev3d.ground_control import floor_pixels
from syncai_bev3d.metric_scale import refine_metric_scale
from syncai_hydranet.geometry.camera_json import CameraFile, Lens, Zone
from syncai_hydranet.geometry.ground import Camera, GroundPlane


def case():
    truth = CameraFile(
        camera_id="scale",
        image_size_px=(960, 540),
        camera=Camera.from_vfov(540, 960, 70.4),
        plane=GroundPlane(2.7, np.radians(52), np.radians(3)),
        lens=Lens(-0.225, (480, 270), 550),
        zones=(Zone("floor", "walkable", ((-1, 2), (1, 2), (0, 4))),),
    )
    initial = replace(truth, plane=replace(truth.plane, height=3.24))
    segments = [
        ([[-1, 2], [1, 2]], False),
        ([[0, 2], [0, 4]], False),
        ([[-1, 3], [0.5, 3]], True),
    ]
    controls = {
        "camera_id": "scale",
        "image_size_px": [960, 540],
        "pixel_space": "raw",
        "surface": "floor",
        "measurement_source": "synthetic physical distances",
        "segments": [
            {
                "points_px": floor_pixels(truth, np.array(p)).tolist(),
                "length_m": float(np.linalg.norm(np.diff(p, axis=0))),
                "validation": v,
            }
            for p, v in segments
        ],
    }
    return truth, initial, controls


def test_recovers_scale_without_changing_lens_angles_or_zone_image_positions():
    truth, initial, controls = case()
    candidate, report = refine_metric_scale(initial, controls)
    assert report["accepted"]
    assert candidate.plane.height == pytest.approx(truth.plane.height)
    assert candidate.camera == initial.camera and candidate.lens == initial.lens
    assert candidate.plane.pitch == initial.plane.pitch
    assert candidate.plane.roll == initial.plane.roll
    assert np.allclose(
        floor_pixels(candidate, candidate.zones[0].points_m),
        floor_pixels(initial, initial.zones[0].points_m),
    )
    assert max(map(abs, report["error_m"])) < 1e-10


def test_bad_holdout_cannot_change_fitted_scale():
    _truth, initial, controls = case()
    _, clean = refine_metric_scale(initial, controls)
    controls["segments"][-1]["length_m"] *= 1.3
    _, wrong = refine_metric_scale(initial, controls)
    assert not wrong["accepted"] and wrong["factor"] == clean["factor"]
    assert "held-out" in wrong["reasons"][0]


def test_conflicting_training_lengths_fail_instead_of_silently_rescaling():
    _truth, initial, controls = case()
    controls["segments"][0]["length_m"] *= 1.5
    _, report = refine_metric_scale(initial, controls)
    assert not report["accepted"]


def test_reversed_duplicate_is_not_independent_validation():
    _truth, initial, controls = case()
    controls["segments"][-1] = deepcopy(controls["segments"][0])
    controls["segments"][-1]["validation"] = True
    controls["segments"][-1]["points_px"].reverse()
    with pytest.raises(ValueError, match="duplicate"):
        refine_metric_scale(initial, controls)


@pytest.mark.parametrize(
    "key,value",
    [
        ("surface", "table"),
        ("pixel_space", "ideal"),
        ("measurement_source", ""),
        ("camera_id", "wrong"),
    ],
)
def test_invalid_measurement_contract_is_rejected(key, value):
    _, initial, controls = case()
    controls[key] = value
    with pytest.raises(ValueError):
        refine_metric_scale(initial, controls)
