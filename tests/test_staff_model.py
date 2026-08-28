"""The staff/customer artefact, and mostly the refusals around it.

`analytics/staff.py` exists because §7.15's classifier had no consumer -- it was a number
in `probe.json` and nothing could load it. What is pinned here is not the arithmetic
(`fit_logreg` is forty lines a reader can check, and `test_appearance.py` already pins
the feature window) but the **four ways this model can be applied to something it does
not describe and say nothing about it**:

    a camera it was never scored on          -> `require_camera`
    a camera it was scored badly on          -> `require_camera`, MIN_DEPLOY_ACCURACY
    a feature window that has since moved    -> `load` on `stat_names`
    a crop size that has since moved         -> `load` on `crop_size`
    a track too short to be one person       -> `staff_verdict` returns None

All four are silent by default: every one of them still produces a confident colour on
every frame of a render. That is the shape the 2026-08-28 handoffs name as the day's
recurring failure -- a green light wired to nothing -- so each is a red test here.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from syncai_hydranet.analytics.appearance import TORSO_STAT_NAMES
from syncai_hydranet.analytics.staff import (
    CROP_SIZE,
    MIN_DEPLOY_ACCURACY,
    MIN_OBSERVATIONS,
    StaffModel,
    crop_features,
    fit_staff_model,
    require_camera,
    staff_verdict,
    track_staff,
)
from syncai_hydranet.analytics.tracker import Tracker


def _batch(n: int = 60):
    """A separable toy set: `staff` are bluer, spread over three cameras."""
    rng = np.random.default_rng(0)
    y = np.array([i % 2 for i in range(n)])
    stats = rng.normal(0.4, 0.05, size=(n, len(TORSO_STAT_NAMES)))
    stats[y == 1, 2] += 0.35  # b_mean
    stats[y == 1, 3] += 0.35  # b_minus_r
    cams = [f"cam{i % 3:02d}" for i in range(n)]
    return stats, y, cams


def test_fit_reports_the_accuracy_of_the_camera_it_was_held_out_on():
    stats, y, cams = _batch()
    m = fit_staff_model(stats, y, cams, held_out="cam01")
    assert m.held_out == "cam01"
    assert "cam01" not in m.trained_cameras
    assert m.held_out_n == cams.count("cam01")
    assert m.n_crops == len(cams) - m.held_out_n
    assert m.accuracy is not None and m.accuracy > 0.9


def test_holding_out_a_camera_that_is_not_in_the_set_is_refused():
    """Otherwise it fits on everything and reports an accuracy for nothing."""
    stats, y, cams = _batch()
    with pytest.raises(ValueError, match="not in the labelled set"):
        fit_staff_model(stats, y, cams, held_out="Taichung-cam99")


def test_a_model_refuses_the_cameras_it_cannot_speak_for():
    stats, y, cams = _batch()
    m = fit_staff_model(stats, y, cams, held_out="cam01")
    assert require_camera(m, "cam01") is m
    with pytest.raises(ValueError, match="was measured on cam01"):
        require_camera(m, "cam02")


def test_the_right_camera_with_the_wrong_number_is_still_refused():
    """The real one, and the reason the gate has two conditions rather than one.

    `runs/staff_model01/model_Tao-Hsin-cam04.json` names its own camera and scores 0.417
    on it. A gate that only matched the camera would wave that through.
    """
    stats, y, cams = _batch()
    m = fit_staff_model(stats, y, cams, held_out="cam01")
    weak = replace(m, accuracy=0.417, held_out_n=12)
    with pytest.raises(ValueError, match="right camera and the wrong number"):
        require_camera(weak, "cam01")
    # ...and it is a floor a caller can lower on purpose, never one it can miss.
    assert require_camera(weak, "cam01", min_accuracy=0.4) is weak


def test_the_deploy_floor_is_the_derived_one():
    """Pinned because the derivation is what makes it defensible: a three-minute clip
    carries 4-10 distinct people, so 0.90 keeps the expected miscoloured count below
    one. If someone lowers it, that argument has to be revisited, not just the constant."""
    assert MIN_DEPLOY_ACCURACY == 0.90


def test_a_model_fitted_on_everything_speaks_for_no_camera():
    """The dangerous one: it has the most training data and the least evidence."""
    stats, y, cams = _batch()
    m = fit_staff_model(stats, y, cams, held_out=None)
    assert m.accuracy is None
    for cam in ("cam00", "cam01", "cam02", "somewhere-else"):
        with pytest.raises(ValueError, match="no held-out accuracy"):
            require_camera(m, cam)


def test_save_load_round_trip_gives_the_same_probabilities(tmp_path):
    stats, y, cams = _batch()
    m = fit_staff_model(stats, y, cams, held_out="cam01")
    back = StaffModel.load(m.save(tmp_path / "model.json"))
    assert np.allclose(m.probability(stats), back.probability(stats))
    assert (back.held_out, back.accuracy, back.held_out_n) == (
        m.held_out,
        m.accuracy,
        m.held_out_n,
    )


def test_a_model_whose_feature_has_moved_under_it_is_refused(tmp_path):
    """Coefficients are positional. A renamed or reordered statistic keeps loading and
    starts meaning something else, which is exactly what nothing downstream could see."""
    stats, y, cams = _batch()
    p = fit_staff_model(stats, y, cams, held_out="cam01").save(tmp_path / "m.json")
    d = json.loads(p.read_text())
    d["stat_names"] = list(reversed(d["stat_names"]))
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match=r"Refit it|refit it"):
        StaffModel.load(p)


def test_a_model_fitted_at_another_crop_size_is_refused(tmp_path):
    """The torso band is a fraction of the crop, so a different resize is a different
    body part and the nine numbers stay perfectly plausible."""
    stats, y, cams = _batch()
    p = fit_staff_model(stats, y, cams, held_out="cam01").save(tmp_path / "m.json")
    d = json.loads(p.read_text())
    d["crop_size"] = [128, 64]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="crop size"):
        StaffModel.load(p)


def test_probability_refuses_a_feature_vector_of_the_wrong_width():
    stats, y, cams = _batch()
    m = fit_staff_model(stats, y, cams, held_out="cam01")
    with pytest.raises(ValueError, match="torso statistics"):
        m.probability(np.zeros((2, len(TORSO_STAT_NAMES) - 1)))


def test_a_verdict_needs_enough_of_one_person():
    """`None` is not `customer`: a two-frame fragment has no opinion, and a consumer that
    renders it as a shopper has invented one."""
    assert staff_verdict([0.9] * (MIN_OBSERVATIONS - 1)) is None
    assert staff_verdict([0.9] * MIN_OBSERVATIONS) is True
    assert staff_verdict([0.1] * MIN_OBSERVATIONS) is False


def test_a_verdict_is_a_median_so_a_coasting_tail_cannot_flip_it():
    p = [0.9] * MIN_OBSERVATIONS + [0.02] * (MIN_OBSERVATIONS // 2)
    assert staff_verdict(p) is True


def _person(h: int, w: int) -> np.ndarray:
    """A crop shaped like a standing person: skin head, blue shirt, dark trousers.

    Structured rather than random on purpose. White noise is *not* size-independent
    under a resize -- resampling it changes its variance, and an earlier version of this
    test failed on exactly that, measuring the interpolator instead of the band. A
    uniform is a large flat region, which is the thing the claim is about.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: int(0.18 * h)] = (214, 178, 150)  # head
    img[int(0.18 * h) : int(0.60 * h)] = (40, 80, 200)  # shirt
    img[int(0.60 * h) :] = (35, 35, 40)  # trousers
    return img


def test_crop_features_are_size_independent_and_stat_shaped():
    """The resize is inside the feature, so two framings of one person agree."""
    a, b = crop_features(_person(40, 20)), crop_features(_person(320, 160))
    assert a.shape == (len(TORSO_STAT_NAMES),)
    assert np.allclose(a, b, atol=0.02)


def test_crop_features_read_the_shirt_and_not_the_head_or_the_trousers():
    """The one claim the nine numbers rest on. A blue shirt has to come back blue even
    though two thirds of the crop is neither blue nor clothing."""
    f = crop_features(_person(256, 128))
    names = list(TORSO_STAT_NAMES)
    assert f[names.index("b_mean")] > f[names.index("r_mean")] + 0.3
    assert f[names.index("blue_frac")] > 0.9


def test_crop_features_take_uint8_and_float_alike():
    rng = np.random.default_rng(2)
    u8 = rng.integers(0, 255, size=(64, 32, 3), dtype=np.uint8)
    assert np.allclose(crop_features(u8), crop_features(u8.astype(np.float32) / 255.0))


def test_the_crop_size_constant_is_the_attribute_crop_geometry():
    """Pinned because §7.15's numbers were measured through it; if it changes, every
    saved model is refused by `load` rather than quietly re-interpreted."""
    assert CROP_SIZE == (256, 128)


# ------------------------------------------------- the wire from the tracker to a verdict
#
# `Track.staff_scores` is the third index-aligned list on `Track`, and it inherits the two
# failures the first two documented: a list that skips frames still zips against `frames`
# without error, and a tracker given staff scores on some frames and not others produces a
# track whose appearance belongs to different frames than it claims.


def _box(x, y, w=20.0, h=60.0):
    return [x, y, x + w, y + h]


def test_a_track_carries_one_staff_score_per_observed_frame():
    t = Tracker()
    for i, p in enumerate([0.91, 0.88, 0.93]):
        t.update(np.array([_box(10.0 + i, 10.0)]), i, staff_scores=np.array([p]))
    track = t.tracks[0]
    assert track.staff_scores == pytest.approx([0.91, 0.88, 0.93])
    assert len(track.staff_scores) == len(track.frames) == len(track.boxes)


def test_staff_scores_stay_optional_so_every_existing_caller_still_works():
    t = Tracker()
    for i in range(3):
        t.update(np.array([_box(10.0 + i, 10.0)]), i)
    assert t.tracks[0].staff_scores == []
    assert track_staff(t.tracks[0]) is None


def test_a_tracker_that_has_seen_staff_scores_refuses_a_frame_without_them():
    t = Tracker()
    t.update(np.array([_box(10.0, 10.0)]), 0, staff_scores=np.array([0.9]))
    with pytest.raises(ValueError, match="staff scores"):
        t.update(np.array([_box(11.0, 10.0)]), 1)


def test_a_tracker_that_has_not_seen_staff_scores_refuses_a_frame_with_them():
    t = Tracker()
    t.update(np.array([_box(10.0, 10.0)]), 0)
    with pytest.raises(ValueError, match="staff scores"):
        t.update(np.array([_box(11.0, 10.0)]), 1, staff_scores=np.array([0.9]))


def test_a_staff_score_per_box_is_required_not_per_track():
    t = Tracker()
    with pytest.raises(ValueError, match="no safe interpretation"):
        t.update(
            np.array([_box(10.0, 10.0), _box(60.0, 10.0)]), 0, staff_scores=np.array([0.9])
        )


def test_track_staff_reduces_a_whole_track_to_one_person():
    t = Tracker()
    for i in range(MIN_OBSERVATIONS):
        t.update(np.array([_box(10.0 + i, 10.0)]), i, staff_scores=np.array([0.8]))
    assert track_staff(t.tracks[0]) is True
