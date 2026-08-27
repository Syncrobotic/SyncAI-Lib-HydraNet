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
