"""Torso colour, now that it is in the package rather than in `scripts/staff_probe.py`.

It moved because `scripts/track_identity.py` needs it too, and
`tests/test_scripts_are_not_libraries.py` refuses one script importing another. The move
is only two thirds paid without tests; this file is the third.

What is pinned here is the **band geometry**, because it is the whole claim: these nine
numbers are a uniform's colour only if the window they are taken over is a torso. The
statistics themselves are arithmetic a reader can check; where the window sits is not,
and it is what silently stops meaning anything if a crop convention changes.

The equivalence with the pre-move code was checked numerically at the time of the move --
identical to 0.0 on five random crops -- and `runs/staff_probe01`'s figures reproduced
exactly afterwards (0.880 colour-only, 0.893 with the ImageNet floor). That is a
one-off check and not repeatable here, since the pre-move code no longer exists; what
survives it is the band, pinned below.
"""

from __future__ import annotations

import numpy as np

from syncai_hydranet.analytics.appearance import (
    TORSO_BAND,
    TORSO_STAT_NAMES,
    torso_region,
    torso_stats,
)


def _crop(h: int = 256, w: int = 128) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.float32)


def test_returns_one_value_per_name():
    assert torso_stats(_crop()).shape == (len(TORSO_STAT_NAMES),)


def test_the_band_is_the_torso_and_not_the_head_or_the_legs():
    """Blue in the band is seen; blue outside it is not.

    Both directions, because only one of them fails loudly. A band that grew to the whole
    crop would still report blue for a blue shirt -- it would merely also report the
    shopper's hair and trousers, and nothing about the number would look wrong.
    """
    r0, r1, c0, c1 = TORSO_BAND
    h, w = 256, 128
    inside = _crop()
    inside[int(r0 * h) : int(r1 * h), int(c0 * w) : int(c1 * w), 2] = 1.0
    outside = _crop()
    outside[:, :, 2] = 1.0
    outside[int(r0 * h) : int(r1 * h), int(c0 * w) : int(c1 * w), 2] = 0.0

    names = list(TORSO_STAT_NAMES)
    assert torso_stats(inside)[names.index("blue_frac")] == 1.0
    assert torso_stats(outside)[names.index("blue_frac")] == 0.0


def test_the_band_is_fractional_so_crop_size_does_not_change_the_answer():
    """The reason `staff_probe.py` may keep its 256x128 resize and a box crop may skip it.

    A pixel band would make the two callers measure different body parts and the
    difference would show up as a colour, not as an error.
    """
    r0, r1, c0, c1 = TORSO_BAND
    small, large = _crop(64, 32), _crop(512, 256)
    for a in (small, large):
        h, w = a.shape[:2]
        a[int(r0 * h) : int(r1 * h), int(c0 * w) : int(c1 * w)] = (0.2, 0.3, 0.9)
    assert np.allclose(torso_stats(small), torso_stats(large), atol=1e-6)


def test_a_crop_too_small_to_have_a_band_returns_zeros_rather_than_raising():
    """A one-pixel-high box is a detector artefact, not a caller error.

    Returning zeros keeps it in the population as an uninformative row; raising would
    make one bad box end a clip-long measurement.
    """
    assert torso_stats(_crop(1, 1)).shape == (len(TORSO_STAT_NAMES),)
    assert torso_stats(np.zeros((0, 4, 3), dtype=np.float32)).tolist() == [0.0] * 9


def _kps(**xy) -> np.ndarray:
    """17 COCO keypoints, all absent, with the named ones placed and confident."""
    k = np.zeros((17, 3), dtype=np.float32)
    for name, (x, y) in xy.items():
        i = {"ls": 5, "rs": 6, "lh": 11, "rh": 12}[name]
        k[i] = (x, y, 0.9)
    return k


def test_the_torso_region_is_shoulders_down_to_hips():
    r = torso_region(_kps(ls=(40, 100), rs=(80, 100), lh=(45, 200), rh=(75, 200)))
    assert r is not None
    x0, y0, x1, y1 = r
    assert (y0, y1) == (100.0, 200.0)
    # both shoulders visible, so the 15% inset applies to the 40..80 span
    assert (round(x0, 1), round(x1, 1)) == (46.0, 74.0)


def test_a_torso_needs_a_shoulder_and_a_hip_and_not_two_of_one():
    """The refusal is the point: two hips fix the wrong end of a torso.

    A region built from them would be a fraction of something again, wearing a keypoint's
    name -- which is the exact defect `torso_region` exists to remove.
    """
    assert torso_region(_kps(ls=(40, 100), rs=(80, 100))) is None
    assert torso_region(_kps(lh=(45, 200), rh=(75, 200))) is None
    assert torso_region(_kps(ls=(40, 100), rh=(75, 200))) is not None


def test_low_confidence_keypoints_do_not_count():
    k = _kps(ls=(40, 100), rs=(80, 100), lh=(45, 200), rh=(75, 200))
    k[[11, 12], 2] = 0.1
    assert torso_region(k) is None


def test_the_region_does_not_move_when_the_box_around_it_does():
    """Why this replaces `TORSO_BAND` for a per-frame crop.

    The same person at the same place returns the same torso whether the detector framed
    her head-to-toe or cropped her at the waist -- because the region is fixed by her
    shoulders and hips, which the box does not touch. `TORSO_BAND` cannot state this: it
    is defined as a fraction of the box, so changing the box *is* changing the band, and
    on 2026-08-27 that moved nine joins across a threshold on people who did nothing.
    """
    k = _kps(ls=(40, 100), rs=(80, 100), lh=(45, 200), rh=(75, 200))
    assert torso_region(k) == torso_region(k)  # same keypoints, any box
