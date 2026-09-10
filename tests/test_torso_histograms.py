"""The batch descriptor: one row per box, NaN where a box is too small to carry one.

pytest tests/test_torso_histograms.py -v
"""

from __future__ import annotations

import numpy as np

from syncai_hydranet.analytics.appearance import HIST_BINS, torso_histograms


def test_one_row_per_box_nan_for_the_small_one_and_the_rest_normalised():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 90, 250], [200, 20, 205, 30], [150, 40, 260, 280]], float)
    out = torso_histograms(frame, boxes)
    assert out.shape == (3, int(np.prod(HIST_BINS)))
    assert np.isnan(out[1]).all()
    assert np.isclose(out[0].sum(), 1.0) and np.isclose(out[2].sum(), 1.0)
    assert torso_histograms(frame, np.zeros((0, 4))).shape == (0, int(np.prod(HIST_BINS)))
