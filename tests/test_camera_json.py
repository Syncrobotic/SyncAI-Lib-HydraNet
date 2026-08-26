"""`camera.json` round-trips exactly, and refuses the silent failures by name.

The file is the only thing that crosses the package boundary at runtime, so the tests
here are contract tests: what survives a save/load unchanged, and which broken states
are caught at load rather than discovered as NaN metres or a zone that never fires.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from syncai_hydranet.geometry.camera_json import (
    SCHEMA_VERSION,
    ZONE_KINDS,
    CameraFile,
    Lens,
    Zone,
)
from syncai_hydranet.geometry.ground import Camera, GroundPlane
from syncai_hydranet.labels import IGNORE


def a_camera_file() -> CameraFile:
    return CameraFile(
        camera_id="Tao-Hsin-cam03",
        image_size_px=(1920, 1080),
        camera=Camera(fx=1490.0, fy=1490.0, cx=960.0, cy=540.0),
        plane=GroundPlane(height=2.38, pitch=math.radians(50.2)),
        lens=Lens(k1=-0.18, centre_px=(960.0, 540.0), radius_px=960.0),
        zones=(
            Zone("entrance", "entrance_line", ((-1.0, 2.0), (1.0, 2.0))),
            Zone("till", "till", ((0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0))),
        ),
        shelf_rois_px=((100.0, 200.0, 600.0, 500.0),),
        false_positive_polygons_px=(((10.0, 10.0), (50.0, 10.0), (50.0, 80.0)),),
        mask_files={"floor": "masks/floor.png", "glass": "masks/glass.png"},
        plate_file="plate.png",
        plate_sha256="0" * 64,
        commissioned_at="2026-08-25T03:00:00+08:00",
    )


def test_round_trip_is_exact(tmp_path):
    path = tmp_path / "camera.json"
    original = a_camera_file()
    original.save(path)
    assert CameraFile.load(path) == original


def test_a_future_schema_version_is_refused_not_guessed(tmp_path):
    path = tmp_path / "camera.json"
    a_camera_file().save(path)
    path.write_text(
        path.read_text().replace(f'"schema_version": {SCHEMA_VERSION}', '"schema_version": 999')
    )
    with pytest.raises(ValueError, match="schema_version"):
        CameraFile.load(path)


def test_degrees_in_the_pitch_field_are_caught():
    """The likeliest unit mistake: 50.2 degrees written as 50.2 'radians'."""
    broken = dataclasses.replace(a_camera_file(), plane=GroundPlane(height=2.38, pitch=50.2))
    with pytest.raises(ValueError, match="degrees in a radian field"):
        broken.validate()


def test_an_unknown_zone_kind_is_refused_because_it_would_never_fire():
    broken = dataclasses.replace(
        a_camera_file(), zones=(Zone("till", "counter", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))),)
    )
    with pytest.raises(ValueError, match="never fires and never errors"):
        broken.validate()


def test_an_roi_outside_the_frame_is_refused():
    broken = dataclasses.replace(
        a_camera_file(), shelf_rois_px=((100.0, 200.0, 2600.0, 500.0),)
    )
    with pytest.raises(ValueError, match="quietly blind"):
        broken.validate()


def test_a_foreign_ignore_sentinel_is_refused_at_load():
    broken = dataclasses.replace(a_camera_file(), mask_ignore=IGNORE - 1)
    with pytest.raises(ValueError, match="different sentinel"):
        broken.validate()


def test_a_zero_height_is_refused_before_it_becomes_nan_metres():
    broken = dataclasses.replace(a_camera_file(), plane=GroundPlane(height=0.0, pitch=0.5))
    with pytest.raises(ValueError, match="NaN"):
        broken.validate()


def test_lens_is_optional_and_survives_absence(tmp_path):
    path = tmp_path / "camera.json"
    no_lens = dataclasses.replace(a_camera_file(), lens=None)
    no_lens.save(path)
    assert CameraFile.load(path).lens is None


# ------------------------------------------------------- the v1 -> v2 widening


def test_display_is_a_kind_because_a_shop_is_mostly_neither_till_nor_premium():
    """Added 2026-08-26 when the fixture footprints reached their accept/reject pass."""
    assert "display" in ZONE_KINDS
    cf = dataclasses.replace(
        a_camera_file(),
        zones=(Zone("left_tables", "display", ((0.0, 1.0), (1.0, 1.0), (1.0, 2.0))),),
    )
    cf.validate()  # would raise on an unknown kind


def test_a_v1_file_is_read_without_re_commissioning(tmp_path):
    """v2 only widened the kind set, so nothing a v1 file can contain changed meaning."""
    path = tmp_path / "cam.json"
    a_camera_file().save(path)
    raw = json.loads(path.read_text())
    raw["schema_version"] = 1
    path.write_text(json.dumps(raw))
    assert CameraFile.load(path).camera_id == "Tao-Hsin-cam03"


def test_a_version_this_reader_does_not_know_is_still_refused(tmp_path):
    path = tmp_path / "cam.json"
    a_camera_file().save(path)
    raw = json.loads(path.read_text())
    raw["schema_version"] = 99
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Re-commission"):
        CameraFile.load(path)
