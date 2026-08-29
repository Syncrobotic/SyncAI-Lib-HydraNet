"""The invariances the temporal model's features claim, and the one they do not."""

import numpy as np
import pytest

from syncai_hydranet.analytics.pose_sequence import (
    L_ANK,
    L_HIP,
    L_KNEE,
    L_SH,
    N_FEATURES,
    R_ANK,
    R_HIP,
    R_KNEE,
    R_SH,
    frame_features,
    sequence_features,
    torso_length,
)


def standing(scale: float = 1.0, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    """A crude upright figure, joints in image coordinates with y down."""
    kp = np.full((17, 3), np.nan)
    kp[:, 2] = 0.0  # unmeasured joints: no coordinates and no confidence
    pts = {
        L_SH: (-20, -60), R_SH: (20, -60), L_HIP: (-14, 0), R_HIP: (14, 0),
        L_KNEE: (-14, 60), R_KNEE: (14, 60), L_ANK: (-14, 120), R_ANK: (14, 120),
    }  # fmt: skip
    for j, (px, py) in pts.items():
        kp[j] = (x + px * scale, y + py * scale, 0.9)
    return kp


def test_features_do_not_move_when_the_person_does():
    a = frame_features(standing())
    b = frame_features(standing(x=500.0, y=-300.0))
    assert np.allclose(a, b, atol=1e-6)


def test_features_do_not_change_when_the_person_walks_towards_the_camera():
    """Twice as large is the same posture; a coordinate model has to learn that, this
    one is told it."""
    a = frame_features(standing(scale=1.0))
    b = frame_features(standing(scale=2.0))
    assert np.allclose(a, b, atol=1e-6)


def test_a_crouch_moves_the_hip_to_ankle_ratio_and_a_walk_does_not():
    up = standing()
    crouched = standing()
    for j in (L_KNEE, R_KNEE, L_ANK, R_ANK):
        crouched[j, 1] = crouched[j, 1] * 0.35  # legs folded, torso unchanged
    i_leg = 5 * 3 + 2  # (L_KNEE, L_ANK) length ratio
    assert frame_features(crouched)[i_leg] < frame_features(up)[i_leg] * 0.6
    walking = standing(x=300.0)
    assert np.isclose(frame_features(walking)[i_leg], frame_features(up)[i_leg], atol=1e-6)


def test_velocity_is_zero_for_a_shopper_who_only_translates():
    seq = np.stack([standing(x=40.0 * t) for t in range(6)])
    f = sequence_features(seq)
    assert f.shape == (6, N_FEATURES * 2)
    assert np.allclose(f[1:, N_FEATURES:], 0.0, atol=1e-6)


def test_an_unmeasurable_limb_is_zero_and_says_so_through_its_confidence():
    kp = standing()
    kp[L_ANK] = (np.nan, np.nan, 0.0)
    f = frame_features(kp)
    i_shin = 5 * 3
    assert np.allclose(f[i_shin : i_shin + 3], 0.0)
    assert f[len(f) - 17 + L_ANK] == 0.0


def test_torso_refuses_rather_than_guessing_when_a_hip_is_missing():
    kp = standing()
    kp[L_HIP, :2] = np.nan
    assert np.isnan(torso_length(kp))


def test_the_features_are_not_invariant_to_a_change_of_camera_elevation():
    """The honest limit, and the reason transfer has to be measured rather than assumed.

    PoseLift's cameras look far closer to straight down than ours -- median height/torso
    0.9 against 2.6 -- which is an anisotropic foreshortening, not a scale change. Limb
    ratios and angles both move under it. The features remove position and scale; they do
    not remove viewpoint, and a test that pretended otherwise would be the reason someone
    later trusts a transfer number that was never earned.
    """
    up = standing()
    squashed = up.copy()
    squashed[:, 1] *= 0.35  # a more overhead view compresses the vertical extent
    assert not np.allclose(frame_features(up), frame_features(squashed), atol=1e-3)


def test_a_malformed_sequence_is_refused():
    with pytest.raises(ValueError, match=r"\(T, 17, 3\)"):
        sequence_features(np.zeros((4, 15, 3)))
