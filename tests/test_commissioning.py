"""The calib -> camera.json converter refuses cameras without metres, and keeps units.

`from_onboard_calib` is the bridge between the 2026-08-19 fleet scan and the
`camera.json` contract. The dangerous path is not the conversion -- it is writing a
plausible file for a camera whose scale was never measured, which would put confident
wrong metres under every downstream event. That refusal is the first thing tested.
"""

from __future__ import annotations

import json
import math

import pytest

from syncai_bev3d.commissioning import from_onboard_calib

A_MEASURED_CALIB = {
    "camera": "Test-cam01",
    "generated": "2026-08-19",
    "vfov_assumed_deg": 70.4,
    "k1_division_model": -0.225,
    "frame_hw_px": [540, 960],
    "pitch_deg": 49.53,
    "roll_deg": 0.84,
    "height_m": 2.49,
    "scale_source": "person_height_median_vs_1.7m_prior_n37",
    "plate_used": "datasets/studioa_static/Test-cam01/plate.png",
}


def write(tmp_path, **overrides):
    payload = {**A_MEASURED_CALIB, **overrides}
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(payload))
    return p


def test_a_measured_camera_converts_with_its_units_intact(tmp_path):
    cf = from_onboard_calib(write(tmp_path))
    assert cf.camera_id == "Test-cam01"
    assert cf.image_size_px == (960, 540)  # calib stores (h, w); the contract stores (w, h)
    assert cf.plane.height == pytest.approx(2.49)
    assert math.degrees(cf.plane.pitch) == pytest.approx(49.53)  # degrees in, radians stored
    assert cf.lens is not None and cf.lens.k1 == pytest.approx(-0.225)
    assert cf.lens.radius_px == pytest.approx(math.hypot(540, 960) / 2)
    cf.validate()


def test_an_unmeasured_scale_is_refused_not_written(tmp_path):
    """DA-V2's plane is a shape; without the person-height scale there are no metres."""
    path = write(tmp_path, scale_source="unmeasured", height_m=3.87)
    with pytest.raises(ValueError, match="waits for its visual reference"):
        from_onboard_calib(path)


def test_a_camera_with_no_fitted_floor_is_refused(tmp_path):
    path = write(tmp_path, height_m=None, pitch_deg=None)
    with pytest.raises(ValueError, match="no fitted floor"):
        from_onboard_calib(path)
