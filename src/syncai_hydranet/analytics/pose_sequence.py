"""Keypoint sequences as the temporal model eats them: view-invariant, not pixels.

PLAN 2.3 puts `sit / crouch / fall` on a small temporal model over pose sequences and
names the transfer risk without a number. Here is the number, measured 2026-08-26 on the
two corpora that would train it:

    height / torso (median)      PoseLift 0.9      ours 2.6
    torso length px (median)     PoseLift  70      ours  66
    joint confidence p50         PoseLift 0.71     ours 0.84

The apparent scale matches almost exactly and the body elongation does not: PoseLift's
six cameras look far closer to straight down than our 42-degree mounts. A model fed raw
coordinates would learn one projection of a human and be shown another, which is the
transfer failure PLAN warns about, arriving through geometry rather than through noise.

So the features are chosen to survive that. Per frame:

* **limb angles** -- the direction of each bone, as (cos, sin). A rotation of the camera
  about the person changes these smoothly and a change of elevation changes them less
  than it changes coordinates.
* **limb lengths normalised by the torso** -- ratios, so the person's distance from the
  camera divides out. This is what makes `crouch` legible: the hip-to-ankle ratio
  collapses whatever the pixel scale is.
* **confidence** per joint, carried through rather than thresholded away, because a
  missing ankle and a badly-placed one are different evidence and the model can learn
  the difference only if it can see which happened.

What is deliberately absent: absolute position, absolute scale, and image size. A model
that cannot see where in the frame the person is cannot learn a camera's furniture.

**And it is not enough, which was measured rather than assumed.** A logistic domain
classifier -- PoseLift frames against ours, held out -- separates the two corpora
perfectly on these features. Per block:

    all features                  53 dims   domain AUC 1.000
    limb angles only              24 dims   domain AUC 1.000
    joint confidences only        17 dims   domain AUC 0.980
    limb length ratios only       12 dims   domain AUC 0.722
    normalised coordinates        34 dims   domain AUC 1.000   (the baseline)

An angle *is* the camera's elevation, so removing position and scale leaves the viewpoint
untouched; and two pose estimators have two confidence distributions. Only the length
ratios carry real invariance, and 0.722 is not 0.5 either.

So: use this to feed a model trained and used on the **same** camera family, and treat
the ratio block as the only part with a claim to transfer. Borrowing another store's 2D
pose corpus to train a behaviour model for ours is not supported by this measurement --
which is why PLAN 2.3 routes `fall` and `crouch` through 3D mocap projected with our own
camera parameters instead. That way the projection is ours by construction.
"""

from __future__ import annotations

import numpy as np

# COCO-17, the order both `events/pose.py` and PoseLift use.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_ELB, R_ELB, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

# The bones whose direction and length carry posture. Face joints are left out: they move
# with the head rather than with the body, and at 66 px of torso they are noise.
LIMBS = (
    (L_SH, L_ELB), (L_ELB, L_WR), (R_SH, R_ELB), (R_ELB, R_WR),
    (L_HIP, L_KNEE), (L_KNEE, L_ANK), (R_HIP, R_KNEE), (R_KNEE, R_ANK),
    (L_SH, R_SH), (L_HIP, R_HIP), (L_SH, L_HIP), (R_SH, R_HIP),
)  # fmt: skip
N_FEATURES = len(LIMBS) * 3 + 17  # (cos, sin, length ratio) per limb, plus 17 confidences
MIN_TORSO_PX = 8.0


def torso_length(kps: np.ndarray) -> float:
    """Shoulder-midpoint to hip-midpoint, the one scale every ratio is taken against.

    NaN when either midpoint is unusable, so a caller cannot silently normalise by a
    number that was never measured.
    """
    sh = kps[[L_SH, R_SH], :2].mean(axis=0)
    hip = kps[[L_HIP, R_HIP], :2].mean(axis=0)
    if not (np.isfinite(sh).all() and np.isfinite(hip).all()):
        return float("nan")
    return float(np.hypot(*(hip - sh)))


def frame_features(kps: np.ndarray) -> np.ndarray:
    """One (17, 3) pose -> `N_FEATURES`. Unmeasurable entries are 0, and say so via conf.

    Zero is a safe filler here precisely because the confidences travel alongside: a limb
    whose joints were never seen has zero direction, zero length AND zero confidence, so
    it is distinguishable from a limb that genuinely points nowhere.
    """
    kps = np.asarray(kps, dtype=float).reshape(17, 3).copy()
    # A zero confidence means the joint was not measured, whatever coordinates came with
    # it. AlphaPose -- which produced PoseLift -- writes (0, 0, 0) for a joint it did not
    # find, and taken literally that is a joint at the image origin: every limb reaching
    # it then points at the top-left corner, and the feature moves when the shopper walks.
    # Two of this module's own tests failed exactly that way before this line existed.
    kps[kps[:, 2] <= 0.0, :2] = np.nan
    out = np.zeros(N_FEATURES, dtype=np.float32)
    torso = torso_length(kps)
    scale = torso if np.isfinite(torso) and torso >= MIN_TORSO_PX else float("nan")
    for i, (a, b) in enumerate(LIMBS):
        pa, pb = kps[a, :2], kps[b, :2]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        d = pb - pa
        n = float(np.hypot(*d))
        if n > 1e-6:
            out[i * 3] = d[0] / n
            out[i * 3 + 1] = d[1] / n
        if np.isfinite(scale):
            out[i * 3 + 2] = n / scale
    out[len(LIMBS) * 3 :] = np.clip(kps[:, 2], 0.0, 1.0)
    return out


def sequence_features(seq: np.ndarray) -> np.ndarray:
    """(T, 17, 3) -> (T, N_FEATURES * 2): the frame's own features and their velocity.

    The difference is taken on the invariant features rather than on coordinates, so a
    shopper walking across the frame contributes nothing to it -- only a shopper whose
    *posture* changes does. That is the whole reason the model is temporal.
    """
    seq = np.asarray(seq, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (17, 3):
        raise ValueError(f"expected (T, 17, 3), got {seq.shape}")
    feats = np.stack([frame_features(f) for f in seq])
    delta = np.zeros_like(feats)
    delta[1:] = feats[1:] - feats[:-1]
    return np.concatenate([feats, delta], axis=1).astype(np.float32)
