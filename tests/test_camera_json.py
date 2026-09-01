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


# ------------------------------------- which teacher produced these metres


def test_a_v2_file_reads_as_teachers_not_recorded_rather_than_none_used(tmp_path):
    """A file written before v3 must stay readable, and stay honest.

    `READABLE_VERSIONS` is a claim that nothing a file of that version holds has changed
    meaning, so a v2 file has to keep loading. What it must not do is come back saying it
    used no teacher: it used one and did not write it down, and the two are different
    facts. `None` is "not recorded"; `{}` would be "recorded as none".

    **The v2 payload is written here rather than borrowed from `runs/commission01`.** It
    used to read the first shipped camera.json and assert its version was 2, which held
    only while nothing legitimately re-wrote those files -- correcting their drifted
    geometry on 2026-09-01 stamped them v3 and this test failed for a reason that had
    nothing to do with v2 files. A schema fixture that a valid re-commissioning can break
    was pinning the wrong thing.
    """
    payload = {
        "schema_version": 2,
        "camera_id": "Test-cam01",
        "image_size_px": [960, 540],
        "camera": {"fx": 700.0, "fy": 700.0, "cx": 480.0, "cy": 270.0},
        "plane": {"height": 2.49, "pitch": 0.8645, "roll": 0.0147},
        "lens": None,
        "zones": [],
        "shelf_rois_px": [],
        "false_positive_polygons_px": [],
        "mask_files": {},
        "mask_ignore": 255,
        "plate_file": None,
        "plate_sha256": None,
        "commissioned_at": "2026-08-19",
    }
    assert "teachers" not in payload, "a v2 file is one written before the field existed"
    path = tmp_path / "v2.camera.json"
    path.write_text(json.dumps(payload))

    assert CameraFile.load(path).teachers is None


def test_teachers_survives_a_save_and_load_round_trip(tmp_path):
    """The field is only worth adding if it reaches the file, so check the file."""
    cam = a_camera_file()
    pinned = {"depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf": "d2fc6a93"}
    stamped = dataclasses.replace(cam, teachers=pinned)

    path = tmp_path / "cam.camera.json"
    stamped.save(path)

    raw = json.loads(path.read_text())
    assert raw["schema_version"] == 3
    assert raw["teachers"] == pinned
    assert CameraFile.load(path).teachers == pinned


def test_the_calib_scan_carries_its_depth_model_into_the_camera_file():
    """The one producer wired today, checked end to end rather than by inspection.

    `onboard_camera.py` writes `provenance.depth_model` and now its revision beside it;
    `from_onboard_calib` is what turns that scan into the contract the event layer reads.
    A scan with no revision yields None -- not a half-recorded pair.
    """
    from syncai_bev3d.commissioning import _teachers_of

    assert _teachers_of({"provenance": {"depth_model": "m", "depth_model_revision": "r"}}) == {
        "m": "r"
    }
    assert _teachers_of({"provenance": {"depth_model": "m"}}) is None
    assert _teachers_of({}) is None
