"""The day/night gate, which decides which frames a teacher is allowed to label.

It is worth pinning because of what sits behind it: SAM 3 returns 14 `person` instances
at score >= 0.5 on an empty IR frame -- hanging packets, not people -- and this test is
the only thing between those frames and the `person` teacher. It was also written twice
under two names before it moved into the package, so the first test here is that one
call answers both questions.

pytest tests/test_teacher_photometry.py -v
"""

from __future__ import annotations

import numpy as np

from syncai_hydranet.data.teachers.photometry import (
    MIN_CHROMA,
    MIN_LUMA,
    is_daylight,
    luma_chroma,
)


def _ir(luma: int = 120) -> np.ndarray:
    """A monochrome frame: three equal channels, which is what an IR camera writes."""
    grey = np.full((32, 48), luma, dtype=np.uint8)
    return np.stack([grey, grey, grey], axis=2)


def _colour(luma: int = 120, spread: int = 40) -> np.ndarray:
    frame = _ir(luma).astype(np.int16)
    frame[:, :, 0] += spread
    frame[:, :, 2] -= spread
    return frame.clip(0, 255).astype(np.uint8)


def test_ir_frame_has_exactly_zero_chroma():
    """The measured claim the threshold rests on: monochrome collapses to 0.00, not near it.

    21 of 96 sampled fleet frames sit at exactly 0.00 and every colour frame at >= 2.13.
    A default of 1.0 is only defensible if the low population really is a point at zero.
    """
    luma, chroma = luma_chroma(_ir(120))
    assert luma == 120.0
    assert chroma == 0.0


def test_a_bright_ir_frame_is_rejected_on_chroma_alone():
    """Brightness cannot rescue it. A floodlit IR frame is the case that matters."""
    assert not is_daylight(_ir(200))


def test_a_daylight_frame_passes():
    assert is_daylight(_colour(120, 40))


def test_a_dark_colour_frame_is_rejected_on_luma():
    assert not is_daylight(_colour(10, 40))


def test_chroma_is_per_pixel_not_over_the_frame():
    """Half red and half blue averages to neutral and is plainly colour.

    A frame-wide mean would call this monochrome and throw away a real daylight frame,
    so the axis this reduces over is a correctness property rather than a detail.
    """
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:, :24] = (200, 100, 0)  # warm half
    frame[:, 24:] = (0, 100, 200)  # cool half, chosen so the channel means match
    assert luma_chroma(frame)[1] > MIN_CHROMA
    assert frame.mean(axis=(0, 1)).std() < 1.0  # neutral frame-wide, by construction


def test_the_defaults_sit_in_the_measured_gap():
    """0.00 below, 2.13 above, and nothing in between -- the cut belongs in the gap."""
    assert 0.0 < MIN_CHROMA < 2.13
    assert MIN_LUMA == 40.0


def test_the_gate_and_the_report_are_one_call():
    """`is_daylight` is `luma_chroma` compared, not a second implementation of it.

    Two copies of this formula existed -- one gating, one reporting -- and the reporting
    one's docstring said "same pixel test as sam3_person_boxes.is_daylight". A comment is
    not an import: nothing failed when they drifted, because nothing compared them.
    """
    for frame in (_ir(200), _colour(120, 40), _colour(10, 40), _colour(45, 1)):
        luma, chroma = luma_chroma(frame)
        assert is_daylight(frame) == (luma >= MIN_LUMA and chroma >= MIN_CHROMA)
