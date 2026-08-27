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


# COCO's shoulders and hips. The torso is what they enclose, which is the definition the
# fractional band above was approximating and getting wrong whenever the box changed shape.
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
TORSO_KEYPOINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)

# `models/heads/pose.py` renders its Gaussians at min_conf 0.3 and decodes the sigmoid
# peak, so this is the same floor the head was trained against rather than a new one.
KP_MIN_CONF = 0.3

# Pulled in from each side when both shoulders are visible, to drop the background either
# side of a person seen front-on. Not applied to a one-sided torso, which is already a
# narrow strip and would inset itself into nothing.
TORSO_INSET = 0.15


def torso_region(kps: np.ndarray, min_conf: float = KP_MIN_CONF):
    """The torso as `(x0, y0, x1, y1)` from one person's keypoints, or `None`.

    **This is the fix for what `TORSO_BAND` gets wrong, and the defect is worth keeping
    written down.** The band is a fraction *of the detected box*, and a box's framing
    changes along a single track -- full standing body at the back of a shop, tight upper
    body walking under the camera, clipped at a frame edge. When it changes, 18-55% of the
    box moves off the shirt and onto hair or trousers, and the nine statistics change
    completely for one person who did nothing. Measured on 2026-08-27: **all nine joins
    that `scripts/track_identity.py` flagged as "further apart than most genuinely
    different people" were one person each**, read by eye off the crops, and box framing
    was the cause in all nine.

    `kps` is `(K, 3)` -- x, y, confidence -- in whatever pixel frame the caller will crop
    in; `models/heads/pose.py:decode_boxes` produces it per box.

    Returns `None` rather than a guess when the keypoints do not support a torso, which
    is **at least one shoulder and at least one hip** above `min_conf`. One shoulder alone
    fixes no vertical extent and two hips alone fix the wrong end of it, so a region built
    from either would be a fraction-of-something again, wearing a keypoint's name. A
    caller that falls back to `TORSO_BAND` here has to record that it did: the whole
    reason this function exists is that the fallback is a different measurement.
    """
    kps = np.asarray(kps, dtype=float)
    ok = kps[:, 2] >= min_conf
    sh = [i for i in (L_SHOULDER, R_SHOULDER) if ok[i]]
    hp = [i for i in (L_HIP, R_HIP) if ok[i]]
    if not sh or not hp:
        return None
    xs = kps[sh + hp, 0]
    y0 = float(np.mean(kps[sh, 1]))
    y1 = float(np.mean(kps[hp, 1]))
    x0, x1 = float(xs.min()), float(xs.max())
    if len(sh) == 2:
        pad = TORSO_INSET * (x1 - x0)
        x0, x1 = x0 + pad, x1 - pad
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1
