"""A measured vfov reaches the calibration: the pin is data, the primary row follows it.

`onboard_camera.py` pinned exactly one camera by name and swept the 70.4 constant as
the primary for every other; `floor_calibrate.py`'s measured vfov had no consumer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    name = "_onboard_camera"
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "onboard_camera.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ob = _load()


def test_the_tile_pinned_camera_is_data_not_a_name_test():
    pin = ob.pin_for("Taichung-cam01", None)
    assert pin["vfov_deg"] == 70.4 and pin["source"] == "tile_grid_pinned"
    assert ob.pin_for("Taichung-cam04", None) is None


def test_a_pins_file_adds_measured_cameras_and_a_later_pin_wins(tmp_path):
    path = tmp_path / "pins.json"
    path.write_text(
        json.dumps(
            {
                "FTI-MDF-cam821": {"vfov_deg": 62.75, "source": "floor_orthogonality"},
                "Taichung-cam01": {"vfov_deg": 69.0, "source": "floor_orthogonality"},
            }
        )
    )
    pins = ob.load_pins(path)
    assert pins["FTI-MDF-cam821"]["vfov_deg"] == 62.75
    assert pins["Taichung-cam01"]["vfov_deg"] == 69.0  # the measurement over the built-in
    assert ob.pin_for("Taichung-cam04", pins) is None


def test_a_pin_without_a_vfov_is_refused(tmp_path):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"X-cam01": {"source": "typo"}}))
    with pytest.raises(ValueError, match="vfov_deg"):
        ob.load_pins(path)


def test_the_primary_row_is_the_cameras_own_vfov_and_a_failed_row_is_none():
    rows = [
        {"vfov_deg": 55.0, "pitch_deg": 34.9},
        {"vfov_deg": 62.75, "pitch_deg": 39.1},
        {"vfov_deg": 70.4, "failed": "no plausible floor plane"},
    ]
    assert ob.primary_row(rows, 62.75)["pitch_deg"] == 39.1
    assert ob.primary_row(rows, 70.4) is None
    assert ob.primary_row(rows, 85.0) is None
