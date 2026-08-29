"""One answer per shopper, and the confidence to read it with.

The crop encoder is a per-frame model and a shopper is not a per-frame quantity: measured
on Kaohsiung-cam04 the gender decision flips on 16.2% of consecutive frame pairs, so the
same staff member is `F` and `M` in adjacent frames of the demo video. Pooling fixes that
by construction, which is why these tests are mostly about the parts that are *not*
automatic -- the refusals, the agreement number, and the age bands that only become
mutually exclusive once pooled.

pytest tests/test_track_attributes.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from syncai_hydranet.analytics.track_attributes import (
    age_band,
    pool,
    track_attributes,
    usable_crops,
)

NAMES = ("Female", "AgeLess18", "Age18-60", "AgeOver60")


def test_a_confident_minority_is_not_outvoted_by_weak_frames():
    """Log-odds pooling, not a count of decisions -- the reason to pool at all.

    Seven frames lean `not female` by a hair and three are certain of the opposite. A
    majority vote returns the seven; the evidence is with the three.
    """
    logits = np.zeros((10, 4))
    logits[:7, 0] = -0.1
    logits[7:, 0] = 6.0
    attrs = track_attributes(logits, NAMES)
    assert attrs["Female"].value is True
    assert attrs["Female"].agreement == pytest.approx(0.3), "and it says the frames split"


def test_agreement_separates_a_certain_track_from_a_coin_flip():
    certain = np.full((20, 4), -4.0)
    certain[:, 0] = 4.0
    split = np.zeros((20, 4))
    split[::2, 0] = 3.0
    split[1::2, 0] = -2.9
    assert track_attributes(certain, NAMES)["Female"].agreement == 1.0
    assert track_attributes(split, NAMES)["Female"].agreement < 0.6


def test_a_track_with_too_few_usable_crops_answers_nothing():
    """A refusal, not a smoothing constant: two good crops is not a smaller measurement."""
    logits = np.random.default_rng(0).normal(size=(9, 4))
    use = np.zeros(9, dtype=bool)
    use[:2] = True
    assert track_attributes(logits, NAMES, use) == {}


def test_truncated_and_tiny_boxes_are_rejected_for_different_reasons():
    boxes = np.array(
        [
            [100.0, 100.0, 200.0, 400.0],  # whole and large -- kept
            [0.0, 100.0, 120.0, 400.0],  # touches the left edge -- truncated
            [100.0, 100.0, 200.0, 150.0],  # 50 px tall -- too small to read
            [100.0, 100.0, 200.0, 400.0],
        ]
    )
    assert usable_crops(boxes, 1920, 1080).tolist() == [True, False, False, True]


def test_the_age_bands_become_exclusive_only_once_pooled():
    """Three independent sigmoids can say yes twice; a track cannot be two ages."""
    logits = np.zeros((8, 4))
    logits[:, 1] = 1.0  # AgeLess18 says yes
    logits[:, 3] = 2.0  # and so does AgeOver60
    attrs = track_attributes(logits, NAMES)
    assert attrs["AgeLess18"].value and attrs["AgeOver60"].value, "the frames contradict"
    band, p = age_band(attrs)
    assert band == "AgeOver60" and p > 0.5, "pooled, the largest is the answer"


def test_pooling_is_a_mean_so_a_long_track_is_not_automatically_certain():
    short = pool(np.full((4, 4), 1.0))
    long = pool(np.full((400, 4), 1.0))
    assert np.allclose(short, long), "length belongs in `frames`, not in the confidence"


def test_no_usable_frame_gives_nan_rather_than_a_confident_zero():
    assert np.isnan(pool(np.ones((5, 4)), np.zeros(5, dtype=bool))).all()
