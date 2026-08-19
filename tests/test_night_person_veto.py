"""The night `person` veto, and the three claims the recipe was accepted on.

The acceptance criteria are not this file's invention: they are what the fleet re-check of
2026-08-19 established and what a night tranche is allowed to rely on.

  1. **Every eye-verified person box survives.** One killed is a rejected recipe -- deleting
     the person a security model exists to find is the failure `sam3_person_boxes.py`'s
     first static gate was switched off for.
  2. **The veto fires only where it should.** A camera with no static false positive must
     come through untouched, or the gate is buying its precision with recall somewhere
     nobody is looking.
  3. **The two unprotected cameras refuse rather than pass.** They are the ones the veto
     provably cannot separate, and silence there would be worse than the failure.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from syncai_hydranet.data.night_person import (
    DROP_ABOVE,
    MIN_MASK_PIXELS,
    UNPROTECTED,
    Decision,
    NightPersonVeto,
    camera_of,
    slot_of,
    static_share,
)

FRAME = "Taichung-cam01__archive_20260816-155837_20260816-160341__01000.jpg"


def test_slot_and_camera_are_read_from_the_name():
    assert camera_of(FRAME) == "Taichung-cam01"
    assert slot_of(FRAME) == "20260816-155837"


def test_slot_is_the_utc_stamp_not_the_second_one():
    """The name carries two timestamps and only the first is the plate's slot key."""
    assert slot_of(FRAME) != "20260816-160341"


class _Veto(NightPersonVeto):
    """A veto with a mask supplied directly, so the tests need no plate on disk."""

    def __init__(self, mask, **kw):
        super().__init__(static_root="/nonexistent", **kw)
        self._mask = mask

    def mask(self, camera, slot):  # noqa: ARG002 - signature is the base class's
        return self._mask


def _mask(share: float, size: int = 40) -> np.ndarray:
    """A uniform mask whose static share is `share` to within one cell."""
    flat = np.zeros(size * size, dtype=bool)
    flat[: round(share * size * size)] = True
    return flat.reshape(size, size)


def test_a_static_box_is_dropped_and_a_moving_one_is_kept():
    box = [(0, 0, 1920, 1080)]
    assert not _Veto(_mask(1.0)).apply(FRAME, [(box[0], 0.9)], 1920, 1080)[0].keep
    assert _Veto(_mask(0.0)).apply(FRAME, [(box[0], 0.9)], 1920, 1080)[0].keep


def test_the_threshold_is_inclusive_at_its_own_value():
    """`>= DROP_ABOVE` drops. Stated as a test because the swept plateau is one-sided."""
    d = _Veto(_mask(DROP_ABOVE)).apply(FRAME, [((0, 0, 1920, 1080), 0.9)], 1920, 1080)[0]
    assert not d.keep and d.share == pytest.approx(DROP_ABOVE, abs=1e-3)


def test_a_box_below_the_score_threshold_is_refused_for_that_reason():
    """Two different refusals must not be reported as one: a caller tuning the detector
    needs to know the plate was never consulted."""
    d = _Veto(_mask(0.0)).apply(FRAME, [((0, 0, 100, 100), 0.20)], 1920, 1080)[0]
    assert not d.keep and d.reason == "below score threshold"
    assert d.share is None


def test_no_plate_is_a_refusal_rather_than_a_pass():
    """A camera with no midnight plate has never been checked. 13 of 42 fail this test
    and none of them is guessable from the frame, so the absence cannot mean yes."""
    veto = _Veto(None)
    d = veto.apply(FRAME, [((0, 0, 1920, 1080), 0.9)], 1920, 1080)[0]
    assert not d.keep and d.reason == "no static plate"


def test_a_box_smaller_than_the_plate_resolution_is_kept_not_dropped():
    """A share of 1.00 over a handful of plate cells is a resolution artefact. The veto
    only ever removes, so where it cannot see it must decline to act."""
    mask = _mask(1.0, size=8)  # 8x8 plate over a 1920x1080 frame
    d = _Veto(mask).apply(FRAME, [((0, 0, 12, 12), 0.9)], 1920, 1080)[0]
    assert d.keep and d.mask_pixels < MIN_MASK_PIXELS


def test_unprotected_cameras_refuse_every_box_with_the_reason_attached():
    for cam, why in UNPROTECTED.items():
        frame = f"{cam}__archive_20260816-160129_20260816-160632__00125.jpg"
        d = _Veto(_mask(0.0)).apply(frame, [((0, 0, 900, 900), 0.9)], 1920, 1080)[0]
        assert not d.keep, f"{cam} must not label unreviewed"
        assert why in d.reason


def test_status_names_the_three_kinds_of_camera():
    veto = NightPersonVeto("/nonexistent")
    assert veto.status("Kaohsiung-cam02") == "excluded"
    assert veto.status("Taichung-cam01") == "veto_active"
    assert veto.status("Kaohsiung-cam04") == "clean"


def test_static_share_maps_the_box_into_plate_coordinates():
    """The plate is 960x540 and the boxes are in source pixels; a top-left quadrant box
    must read the top-left quadrant of the plate and nothing else."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[:50, :50] = True
    share, npx = static_share(mask, (0, 0, 960, 540), 1920, 1080)
    assert share == pytest.approx(1.0)
    assert npx == 50 * 50
    share, _ = static_share(mask, (960, 540, 960, 540), 1920, 1080)
    assert share == pytest.approx(0.0)


def test_decision_is_frozen_so_a_caller_cannot_edit_a_verdict():
    d = Decision((0, 0, 1, 1), 0.9, False, "static: furniture", 0.99, 500)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.keep = True  # type: ignore[misc]
