"""Surface hints must preserve raw provenance and cannot invent weak boundaries."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from syncai_bev3d.structural_observations import (
    ObservationConfig,
    extract_edges,
    observe_directory,
)
from syncai_bev3d.structural_surfaces import PROMPTS, enrich_surfaces


def test_weak_pass_uses_only_native_pixels_inside_support():
    image = Image.new("RGB", (480, 320), (100, 100, 100))
    ImageDraw.Draw(image).line([(60, 160), (420, 160)], fill=(108, 108, 108), width=2)
    rgb = np.asarray(image)
    ordinary, _ = extract_edges(rgb)
    assert not ordinary
    mask = np.zeros((320, 480), dtype=bool)
    mask[130:190, 50:430] = True
    weak, _ = extract_edges(
        rgb, ObservationConfig(min_gradient=0.003, edge_quantile=0.8), support_mask=mask
    )
    assert weak
    for row in weak:
        x, y = np.asarray(row["points_px"]).T
        assert mask[y, x].all()
        assert np.abs(y - 160).max() <= 3
    empty, _ = extract_edges(rgb, support_mask=np.zeros_like(mask))
    assert not empty


class FakeTeacher:
    def __init__(self):
        self.identity = {"model": "synthetic-test-only", "revision": "1"}

    def __call__(self, rgb):
        masks = {key: np.zeros(rgb.shape[:2], dtype=bool) for key in PROMPTS}
        masks["floor"][:] = True
        masks["person"][:, 40:120] = True
        return masks, {key: [0.9] for key in PROMPTS}


def test_person_exclusion_and_mask_provenance(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        image = Image.new("RGB", (320, 240), (30 + i, 30, 30))
        draw = ImageDraw.Draw(image)
        draw.line([(80, 20), (80, 220)], fill="white", width=3)
        draw.line([(220, 20), (220, 220)], fill="white", width=3)
        image.save(frames / f"{i}.png")
    raw = observe_directory(frames, "unseen")
    result = enrich_surfaces(raw, FakeTeacher(), tmp_path / "evidence")
    assert result["rejected_observations"]
    assert all(row["blocked_fraction"] <= 0.1 for row in result["observations"])
    assert all(
        np.mean(np.asarray(row["points_px"])[:, 0]) > 120 for row in result["observations"]
    )
    assert result["surface_evidence"]["status"] == "predictions_only"
    assert len(result["surface_evidence"]["frames"]) == 3
    assert not result["calibration_ready"]
    for item in result["surface_evidence"]["frames"]:
        source = next(f for f in raw["frames"] if f["id"] == item["frame_id"])
        assert source["sha256"] == item["source_sha256"]
    with pytest.raises(FileExistsError):
        enrich_surfaces(raw, FakeTeacher(), tmp_path / "evidence")


def test_changed_raw_frame_is_rejected_before_teacher_runs(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    image = frames / "000.png"
    Image.new("RGB", (80, 80), "gray").save(image)
    raw = observe_directory(frames, "unseen")
    Image.new("RGB", (80, 80), "white").save(image)
    with pytest.raises(ValueError, match="source frame changed"):
        enrich_surfaces(raw, FakeTeacher(), tmp_path / "evidence")
    assert '"status": "failed"' in (tmp_path / "evidence/progress.json").read_text()
