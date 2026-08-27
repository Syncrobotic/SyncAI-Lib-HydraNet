"""Torso colour as an identity witness: nine numbers over a person crop.

These are the statistics `scripts/staff_probe.py` measured on 2026-08-27 (PLAN §7.15),
where they read **0.893 balanced accuracy held out by camera for `staff/customer` and
beat every embedding in the tree** -- both attribute fine-tunes and untrained ImageNet
features. They are here rather than in that script because a second caller now needs
them, and `tests/test_scripts_are_not_libraries.py` refuses one script importing another.

**What moved is the arithmetic, not the probe.** `staff_probe.py` standardises its
feature matrix before fitting, and that standardisation stayed there: it is a property of
the *set being fitted*, so a function that returned it would be answering a different
question for every caller. `torso_stats` returns the raw nine, in their own units, and
each caller says what it wants them compared against.

**What these are not.** They are a colour, and a colour is not an identity. Two shoppers
in dark coats are the same nine numbers; §7.15's own two failure cases were a camera whose
staff wear jackets over the polo and a near-top-down camera whose crops are hair and
shoulders. So a *difference* here is evidence and a *match* is not -- which is the
asymmetry any caller using this to tell two people apart has to be built around.
"""

from __future__ import annotations

import numpy as np

# The chest and upper arms of a standing person, as fractions of the crop: excluding the
# head (hair and skin), the legs (trousers vary per person) and the edges (background).
# Fractions rather than pixels so the band is the same body part at any crop size, which
# is what lets `staff_probe.py` keep its 256x128 resize and a per-frame box crop skip it.
TORSO_BAND = (0.18, 0.55, 0.25, 0.75)

# The names, in the order `torso_stats` returns them. Kept beside the code that produces
# them because a bare 9-vector in a JSON file is unreadable a week later.
TORSO_STAT_NAMES = (
    "r_mean", "g_mean", "b_mean", "b_minus_r", "b_minus_g",
    "sat_mean", "max_mean", "blue_frac", "b_minus_r_p75",
)  # fmt: skip


def torso_stats(rgb: np.ndarray) -> np.ndarray:
    """Nine colour statistics over the torso band of one person crop.

    `rgb` is an H x W x 3 array of floats in [0, 1]. Any size: the band is fractional.
    Returns the nine values named by `TORSO_STAT_NAMES`, raw and unstandardised.
    """
    h, w = rgb.shape[:2]
    r0, r1, c0, c1 = TORSO_BAND
    y0, x0 = int(r0 * h), int(c0 * w)
    # `max` so a crop too short to have a band still yields one row rather than none:
    # a one-pixel-high box is a detector artefact, and it should not end a clip-long run.
    y1, x1 = max(int(r1 * h), y0 + 1), max(int(c1 * w), x0 + 1)
    band = rgb[y0:y1, x0:x1]
    px = band.reshape(-1, 3)
    if not len(px):
        return np.zeros(len(TORSO_STAT_NAMES), dtype=np.float32)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mx, mn = px.max(1), px.min(1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    # "blueness" two ways: blue over the other channels, and blue-dominant saturated
    # pixels as a fraction -- a polo fills the band, a blue carrier bag does not.
    blue_frac = float(np.mean((b > r + 0.06) & (b > g + 0.02) & (sat > 0.25)))
    return np.asarray(
        [
            float(r.mean()), float(g.mean()), float(b.mean()),
            float(b.mean() - r.mean()), float(b.mean() - g.mean()),
            float(sat.mean()), float(mx.mean()), blue_frac,
            float(np.percentile(b - r, 75)),
        ],
        dtype=np.float32,
    )  # fmt: skip
