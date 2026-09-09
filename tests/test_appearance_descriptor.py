"""The descriptor the gate is calibrated on and applied with.

`camera.json`'s `appearance_thr` is a distance in whatever space this function defines, so
a second copy of the arithmetic anywhere would make the number a measurement of something
else while both sides stayed internally consistent.
"""

from __future__ import annotations

import numpy as np

from syncai_hydranet.analytics.appearance import HIST_MIN_H_PX, torso_histogram
from syncai_hydranet.analytics.tracker import appearance_distance


def _flat(colour, w=40, h=120):
    return np.tile(np.asarray(colour, np.uint8), (h, w, 1))


def test_two_colours_are_far_apart_and_one_colour_is_zero_from_itself():
    red, blue = _flat((220, 30, 30)), _flat((30, 30, 220))
    hr = torso_histogram(red, (0, 0, 40, 120))
    hb = torso_histogram(blue, (0, 0, 40, 120))
    assert appearance_distance(hr, hr) == 0.0
    assert appearance_distance(hr, hb) > 0.8


def test_it_normalises_so_box_size_does_not_change_the_descriptor():
    """Two crops of the same shirt at two distances must compare as the same person."""
    big = torso_histogram(_flat((90, 160, 60), 80, 240), (0, 0, 80, 240))
    small = torso_histogram(_flat((90, 160, 60), 20, 60), (0, 0, 20, 60))
    assert appearance_distance(big, small) < 1e-9
    assert np.isclose(big.sum(), 1.0)


def test_it_reads_torso_band_and_not_the_head_or_the_legs():
    """The same band `torso_stats` uses, and for its reasons: the head is hair and skin,
    the legs are trousers that vary per person, and the crop's edges are background."""
    from syncai_hydranet.analytics.appearance import TORSO_BAND

    r0, r1, _, _ = TORSO_BAND
    frame = np.zeros((100, 40, 3), np.uint8)
    frame[:] = (30, 30, 220)  # head, legs and edges: must not reach the histogram
    frame[int(r0 * 100) : int(r1 * 100)] = (220, 30, 30)  # the torso band alone
    got = torso_histogram(frame, (0, 0, 40, 100))
    red_only = torso_histogram(_flat((220, 30, 30)), (0, 0, 40, 120))
    assert appearance_distance(got, red_only) < 1e-9


def test_a_box_too_small_returns_none_rather_than_a_zero_vector():
    """A zero vector is far from every real histogram, so it would read as "a different
    person" at every comparison and turn small boxes into false identity changes."""
    assert torso_histogram(_flat((10, 10, 10)), (0, 0, 40, HIST_MIN_H_PX - 1)) is None
    assert torso_histogram(_flat((10, 10, 10)), (0, 0, 4, 120)) is None


def test_a_box_outside_the_frame_is_clipped_not_wrapped():
    frame = _flat((200, 40, 40), 40, 120)
    assert torso_histogram(frame, (-50, -50, 40, 120)) is not None
    assert torso_histogram(frame, (200, 200, 300, 400)) is None


# -- the threshold lives on the camera, not in a constant -------------------


def _bare_camera(**kw):
    """A CameraFile built here rather than read out of `runs/`.

    `runs/` is gitignored, so a test that borrows a commissioned file runs on this box and
    skips forever on a clean checkout -- and `tests/conftest.py` records that a path typo
    turning a test into a permanent skipper has already happened once. The two files that
    already build their own (`test_camera_json`, `test_figures_geometry`) say the same.
    """
    from syncai_hydranet.geometry.camera_json import CameraFile
    from syncai_hydranet.geometry.ground import Camera, GroundPlane

    return CameraFile(
        camera_id="test",
        image_size_px=(960, 540),
        camera=Camera(fx=380.0, fy=380.0, cx=480.0, cy=270.0),
        plane=GroundPlane(height=2.5, pitch=0.9),
        **kw,
    )


def test_camera_file_carries_and_round_trips_the_threshold(tmp_path):
    """`appearance_thr` is a distance in the descriptor's space and a property of one
    camera's lighting and clothing, so it has to survive save/load or commissioning's
    answer is lost.

    The expected value is written here rather than read back from the source file. The
    first cut compared a loaded file against its own reload, so a `load` that ignored the
    field returned None on both sides and passed -- caught by reverting `load` and finding
    the suite still green.
    """
    import json as _json

    from syncai_hydranet.geometry.camera_json import CameraFile

    out = tmp_path / "c.json"
    _bare_camera(appearance_thr=0.3584).save(out)
    assert _json.loads(out.read_text())["appearance_thr"] == 0.3584, "save must write it"
    assert CameraFile.load(out).appearance_thr == 0.3584, "load must read it back"

    raw = _json.loads(out.read_text())
    raw.pop("appearance_thr")
    out.write_text(_json.dumps(raw))
    assert CameraFile.load(out).appearance_thr is None, "an older file means 'do not gate'"


def test_the_fleet_calibration_is_not_one_number():
    """The evidence for the field existing at all, read off the commissioned files.

    Skipped rather than failed on a fresh checkout: `runs/` is gitignored and a guard that
    fails where its input cannot exist is the shape this suite has had to rescue before.
    """
    from pathlib import Path

    import pytest as _pytest

    from syncai_hydranet.geometry.camera_json import CameraFile

    paths = sorted(Path("runs/commission01").glob("*.camera.json"))
    if len(paths) < 4:
        _pytest.skip("no commissioned cameras in this checkout")
    got = {}
    for path in paths:
        cf = CameraFile.load(path)
        got[cf.camera_id] = cf.appearance_thr
    have = [v for v in got.values() if v is not None]
    if not have:
        _pytest.skip("this checkout's cameras are not calibrated")
    assert max(have) / min(have) > 1.5, f"a constant would do; the scan said otherwise: {got}"
    assert any(v is None for v in got.values()), "None must stay reachable: it is the safe half"


def test_a_non_positive_threshold_on_a_camera_is_refused():
    """Zero would refuse every re-association on that camera, splitting every track that
    passes behind a fixture. 'Do not gate here' is spelled None."""
    import dataclasses

    import pytest as _pytest

    with _pytest.raises(ValueError, match="not a positive distance"):
        dataclasses.replace(_bare_camera(), appearance_thr=0.0).validate()
