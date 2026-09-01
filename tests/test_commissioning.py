"""The calib -> camera.json converter refuses cameras without metres, and keeps units.

`from_onboard_calib` is the bridge between the 2026-08-19 fleet scan and the
`camera.json` contract. The dangerous path is not the conversion -- it is writing a
plausible file for a camera whose scale was never measured, which would put confident
wrong metres under every downstream event. That refusal is the first thing tested.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from syncai_bev3d.commissioning import from_onboard_calib, regeometry_from_calib
from syncai_hydranet.geometry.camera_json import Zone

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


# --- the return leg: a corrected calibration reaching an already-commissioned file -----


def _commissioned(tmp_path, height_m=2.49):
    """A camera.json that has been through the later passes, as a real one has."""
    cf = from_onboard_calib(write(tmp_path, height_m=height_m))
    from dataclasses import replace

    cf = replace(
        cf,
        zones=(
            Zone(
                name="walkable_floor",
                kind="walkable",
                points_m=((1.0, 2.0), (3.0, 4.0), (5.0, 1.0)),
            ),
        ),
        shelf_rois_px=((10.0, 20.0, 30.0, 40.0),),
        false_positive_polygons_px=(((1.0, 1.0), (2.0, 2.0), (3.0, 1.0)),),
        mask_files={"floor": "Test-cam01/masks/floor.png"},
    )
    out = tmp_path / "cam.json"
    cf.save(out)
    return out


def test_a_corrected_calibration_keeps_what_the_later_passes_added(tmp_path):
    """The reason this function exists rather than re-running `from_onboard_calib`.

    A fresh conversion carries only what the scan measured, so on Taichung-cam10 it would
    have dropped 13 mask files, 14 zones, 5 shelf ROIs and 3 false-positive polygons --
    which is why the geometry of a commissioned camera had no way to be corrected at all,
    and why seven of eight had silently drifted from their own calib.json by 2026-09-01.
    """
    cam = _commissioned(tmp_path)
    calib = write(tmp_path, height_m=2.57)
    out = regeometry_from_calib(cam, calib)
    assert out.plane.height == pytest.approx(2.57), "the geometry is the point"
    assert out.mask_files == {"floor": "Test-cam01/masks/floor.png"}
    assert len(out.shelf_rois_px) == 1
    assert len(out.false_positive_polygons_px) == 1
    assert len(out.zones) == 1


def test_metre_zones_move_with_the_plane_they_were_measured_on(tmp_path):
    """Zones are metres on the floor, so a height change relocates them.

    Exact rather than approximate while pitch and roll hold: `pixel_to_ground` puts a floor
    pixel at a distance proportional to the plane height, so every zone point scales by the
    same ratio. Preserving them unscaled would leave a zone naming a different piece of
    floor than the one it was drawn on -- and nothing downstream reads metres sceptically
    enough to notice.
    """
    cam = _commissioned(tmp_path, height_m=2.87)
    out = regeometry_from_calib(cam, write(tmp_path, height_m=2.57))
    r = 2.57 / 2.87
    assert out.zones[0].points_m[0] == pytest.approx((1.0 * r, 2.0 * r))
    assert out.zones[0].points_m[1] == pytest.approx((3.0 * r, 4.0 * r))
    assert out.zones[0].points_m[2] == pytest.approx((5.0 * r, 1.0 * r))
    assert out.zones[0].name == "walkable_floor", "only the coordinates move"


def test_a_pitch_change_is_refused_rather_than_scaled(tmp_path):
    """The rescale is exact only while the plane's orientation holds.

    Under a changed pitch the zones have to be re-derived from the masks. Scaling them
    anyway would produce a file that validates, loads, and is wrong.
    """
    cam = _commissioned(tmp_path)
    with pytest.raises(ValueError, match="orientation holds"):
        regeometry_from_calib(cam, write(tmp_path, pitch_deg=55.0))


# --- the drift itself, on the shipped files -------------------------------------------

REPO = Path(__file__).resolve().parent.parent


def _shipped_pairs():
    """(camera.json, calib.json) for every commissioned camera that has both."""
    out = []
    for f in sorted((REPO / "runs/commission01").glob("*.camera.json")):
        calib = REPO / f"runs/onboard01/{f.stem.replace('.camera', '')}.calib.json"
        if calib.exists():
            out.append((f, calib))
    return out


def test_a_commissioned_camera_still_agrees_with_its_own_calibration():
    """The drift that had nothing watching it.

    `runs/onboard01` is re-run whenever the calibration improves; `runs/commission01` is
    written once and appended to. Nothing held them together, so on 2026-09-01 seven of
    the eight shipped cameras disagreed with their own source -- Taichung-cam10 by 0.30 m,
    both files stamped `commissioned_at: 2026-08-19`, the figure it feeds still printing
    the old metres. A camera.json is what every downstream metre is drawn in, so a stale
    one is not a stale file, it is a wrong measurement in a picture nobody re-checks.

    Skipped rather than failed on a fresh checkout: both directories are gitignored, and a
    guard that fails where its input cannot exist is the shape `test_map_anything_eval`
    already had to be rescued from.
    """
    pairs = _shipped_pairs()
    if not pairs:
        pytest.skip("no commissioned cameras in this checkout")
    drift = []
    for cam_path, calib_path in pairs:
        cam = json.loads(cam_path.read_text())
        cal = json.loads(calib_path.read_text())
        dh = abs(cam["plane"]["height"] - cal["height_m"])
        dp = abs(math.degrees(cam["plane"]["pitch"]) - cal["pitch_deg"])
        if dh > 0.005 or dp > 0.05:
            drift.append(
                f"  {cal['camera']}: camera.json {cam['plane']['height']:.3f} m / "
                f"{math.degrees(cam['plane']['pitch']):.2f} deg vs calib.json "
                f"{cal['height_m']:.3f} m / {cal['pitch_deg']:.2f} deg"
            )
    assert not drift, (
        "commissioned geometry has drifted from the calibration it came from:\n"
        + "\n".join(drift)
        + "\n\nRun `commissioning.regeometry_from_calib` -- it carries the later passes "
        "across and rescales the metre zones, which re-running `from_onboard_calib` "
        "would discard."
    )
