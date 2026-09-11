"""Specialist labels, ignored pixels and calibrated glass-door geometry."""

import math

import numpy as np
import pytest
import torch

from syncai_bev3d.object_assets import glass_door_meshes
from syncai_bev3d.surfaces import scene_surfaces
from syncai_hydranet.data.trans10k import RGB_TO_GLAZING, glazing_labels
from syncai_hydranet.models.glazing import GlazingHead, confusion_scores, glazing_loss
from test_surfaces import evidence


def test_palette_uses_colours_not_an_unrelated_numeric_class_order():
    colours = np.array(list(RGB_TO_GLAZING), np.uint8)[None]
    assert glazing_labels(colours).tolist() == [list(RGB_TO_GLAZING.values())]
    # The door is grey 120, not the mirror's numeric label 7 or another export's 5.
    assert glazing_labels(
        np.array([[[120, 120, 120], [140, 140, 140]]], np.uint8)
    ).tolist() == [[1, 0]]
    assert glazing_labels(np.array([[[121, 120, 120], [0, 0, 0]]], np.uint8)).tolist() == [
        [255, 0]
    ]
    assert glazing_labels(np.array([[[120, 120, 120, 0]]], np.uint8)).item() == 255
    with pytest.raises(ValueError, match="RGB"):
        glazing_labels(np.zeros((4, 4), np.uint8))


def test_ignored_pixels_contribute_no_training_gradient():
    torch.manual_seed(42)
    logits = torch.randn(1, 4, 5, 6, requires_grad=True)
    labels = torch.ones(1, 5, 6, dtype=torch.long)
    labels[:, :2] = 255
    loss = glazing_loss(logits, labels, torch.ones(4))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[:, :, :2]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 2:]) > 0
    ignored = glazing_loss(logits, torch.full_like(labels, 255), torch.ones(4))
    assert ignored == 0


def test_specialist_head_can_learn_a_separable_signal():
    torch.manual_seed(1)
    features = torch.randn(2, 64, 6, 8)
    labels = (features[:, 0] > 0).long()
    head = GlazingHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=0.02)
    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        loss = glazing_loss(head(features), labels, torch.ones(4))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.6


def test_absent_classes_do_not_inflate_validation_scores():
    metrics = confusion_scores([[10, 2, 0, 0], [3, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert metrics["iou"][1] == 0.5
    assert metrics["iou"][2:] == [None, None]
    assert metrics["foreground_miou"] == 0.5
    assert metrics["door_precision"] == 5 / 7
    assert metrics["door_recall"] == 5 / 8


@pytest.mark.parametrize("angle", [0, 0.7, math.pi / 2, -2.0])
def test_frame_and_leaf_share_calibrated_plane_and_do_not_enlarge_opening(angle):
    centre = np.array([1.2, 4.0])
    along = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([-along[1], along[0]])
    points = centre + np.array([[-0.6], [0.6]]) * along
    pane, frame = glass_door_meshes(points, 2.1, 0.1)
    for vertices, faces in [pane, frame]:
        planar = vertices[:, [0, 2]] - centre
        assert np.abs(planar @ along).max() <= 0.6 + 1e-8
        assert np.abs(planar @ normal).max() <= 0.02 + 1e-8
        assert vertices[:, 1].min() >= 0.1 - 1e-8
        assert vertices[:, 1].max() <= 2.2 + 1e-8
        assert faces.min() >= 0 and faces.max() < len(vertices)
    assert np.isclose(np.ptp((frame[0][:, [0, 2]] - centre) @ along), 1.2)
    assert np.isclose(np.ptp(frame[0][:, 1]), 2.1)
    assert pane[0][:, 1].min() > frame[0][:, 1].min()


def test_invalid_opening_is_rejected():
    with pytest.raises(ValueError):
        glass_door_meshes([[1, 2], [1, 2]], 2)
    with pytest.raises(ValueError):
        glass_door_meshes([[1, 2], [2, 2]], float("nan"))


def test_prediction_preview_uses_grounded_geometry_without_writing_masks(tmp_path):
    mask, ev = evidence()
    mask = mask.copy()
    mask[:, :265] = False
    mask[:, 375:] = False
    reports = []
    surfaces = scene_surfaces(
        "surface", ev, tmp_path, mask_overrides={"glass_door": mask}, report=reports
    )
    assert len(surfaces) == 1
    assert surfaces[0].kind == "glass_door"
    assert np.allclose(surfaces[0].points[:, 1], 5, atol=0.12)
    assert reports[0]["status"] == "supported"
    assert reports[0]["mask_source"] == "candidate override"
    assert not list(tmp_path.rglob("*.png"))
    assert not scene_surfaces("surface", ev, tmp_path)


def test_oversized_candidate_door_is_rejected_even_with_floor_contact(tmp_path):
    mask, ev = evidence()
    reports = []
    assert not scene_surfaces(
        "surface", ev, tmp_path, mask_overrides={"glass_door": mask}, report=reports
    )
    assert "outside span" in reports[0]["reason"]


def test_unanchored_glass_prediction_is_reported_and_not_extruded(tmp_path):
    mask, ev = evidence(bottom=0.8)
    reports = []
    assert not scene_surfaces(
        "surface", ev, tmp_path, mask_overrides={"glass_door": mask}, report=reports
    )
    assert reports[0]["status"] == "rejected"
    assert reports[0]["reason"] == "no supported plane passes reprojection"


def test_training_cli_writes_a_usable_checkpoint_from_verified_features(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from syncai_bev3d.render_provenance import sha256
    from syncai_hydranet.data.trans10k import GLAZING_CLASSES

    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(42)
    spec = {"classes": list(GLAZING_CLASSES), "splits": {}}
    for split in ["train", "validation"]:
        features = rng.normal(size=(4, 64, 4, 8)).astype(np.float16)
        labels = np.tile(np.arange(4, dtype=np.uint8)[:, None], (4, 1, 8))
        np.save(cache / f"{split}.features.npy", features)
        np.save(cache / f"{split}.labels.npy", labels)
        spec["splits"][split] = {
            "feature_shape": list(features.shape),
            "class_pixels": [32] * 4,
            "features_sha256": sha256(cache / f"{split}.features.npy"),
            "labels_sha256": sha256(cache / f"{split}.labels.npy"),
        }
    (cache / "manifest.json").write_text(json.dumps(spec))
    out = tmp_path / "model"
    subprocess.run(
        [
            sys.executable,
            "tools/commissioning/train_glazing.py",
            "--threads",
            "1",
            "train",
            "--cache",
            str(cache),
            "--out",
            str(out),
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    saved = torch.load(out / "best.pt", weights_only=True)
    head = GlazingHead()
    head.load_state_dict(saved["state_dict"])
    assert saved["epoch"] == 1
    assert torch.isfinite(head(torch.from_numpy(features.astype(np.float32)))).all()
