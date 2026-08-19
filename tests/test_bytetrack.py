"""The ByteTrack forward pass, now that it is in the package rather than in a script.

It moved out of `scripts/offline_tracks.py` because `scripts/stable_infer.py` needed it
too and was reaching it through a `sys.path` insert.
`tests/test_scripts_are_not_libraries.py` states what that costs -- shared code in
`scripts/` sits outside the wheel, the type ratchet *and* the coverage floor -- so a move
with no tests behind it would have paid only two thirds of the point. This file is the
third.

What is pinned here is the association policy, because that is what a caller depends on
and what a reader cannot check by eye: which detections may start a track, which may only
continue one, how long a lost track coasts, and what box it reports while coasting. The
Kalman arithmetic itself is ByteTrack's and is checked only where this project's use of it
differs -- the frame-rate rescaling of the velocity prior.
"""

from __future__ import annotations

import numpy as np
import pytest

from syncai_hydranet.analytics.bytetrack import (
    Fragment,
    Kalman,
    OfflineForward,
    to_cwh,
    to_xyxy,
)


def _fwd(**kw) -> OfflineForward:
    """The defaults `stable_infer.py` runs with, unless a test says otherwise."""
    args = {
        "high_thr": 0.35,
        "low_thr": 0.20,
        "iou_thr": 0.3,
        "iou_thr_low": 0.5,
        "max_age": 5,
        "min_hits": 2,
        "vel_scale": 1.0,
    }
    args.update(kw)
    return OfflineForward(**args)


def _box(cx, cy, w=40.0, h=100.0) -> np.ndarray:
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)


# --------------------------------------------------------------------- box conversion


def test_cwh_and_xyxy_are_inverses():
    box = _box(120.0, 300.0)
    assert np.allclose(to_xyxy(to_cwh(box)), box)


def test_cwh_reports_centre_and_extent_not_corners():
    """The stitch pass in `offline_tracks.py` takes `to_cwh(box)[:2]` as a centre."""
    cx, cy, w, h = to_cwh(_box(120.0, 300.0, 40.0, 100.0))
    assert (cx, cy, w, h) == (120.0, 300.0, 40.0, 100.0)


# ----------------------------------------------------------------------------- Kalman


def test_a_kalman_with_no_observation_holds_still():
    """One `predict` off a single box: constant velocity, and the velocity is zero."""
    k = Kalman(_box(100.0, 200.0), vel_scale=1.0)
    k.predict()
    assert np.allclose(k.box, _box(100.0, 200.0))
    assert np.allclose(k.velocity, [0.0, 0.0])


def test_the_filter_follows_a_moving_box_and_learns_its_direction():
    k = Kalman(_box(100.0, 200.0), vel_scale=1.0)
    for step in range(1, 8):
        k.predict()
        k.update(_box(100.0 + 10.0 * step, 200.0))
    k.predict()
    assert k.box[0] > _box(170.0, 200.0)[0], "the prediction should lead the last observation"
    assert k.velocity[0] > 0, "rightward motion should read as a positive x velocity"
    assert abs(k.velocity[1]) < abs(k.velocity[0]), "no vertical motion was ever observed"


def test_the_velocity_prior_scales_with_the_frame_rate():
    """The one adaptation this project makes to ByteTrack's MOT17 weights.

    At 5 fps a walking shopper moves ~5x further between frames than at MOT17's 25-30, so
    an unscaled prior gates out every real match. A larger `vel_scale` must therefore admit
    more velocity uncertainty, which is the covariance this asserts on -- not the state,
    which one observation cannot move.
    """
    tight = Kalman(_box(100.0, 200.0), vel_scale=1.0)
    loose = Kalman(_box(100.0, 200.0), vel_scale=5.0)
    assert loose.P[4, 4] > tight.P[4, 4]


# ------------------------------------------------------------------------ association


def test_only_a_high_score_detection_may_start_a_track():
    """The low band exists to *continue* a track through an occlusion, not to begin one.

    A low-score box that could give birth is how a fixture that momentarily looks like a
    shopper enters the count, and analytics counts tracks -- so a spurious birth is not
    noise on the number, it is an addition to it.
    """
    fwd = _fwd()
    fwd.update(np.array([_box(100.0, 200.0)]), np.array([0.25]), 0)
    assert fwd.tracks == []

    fwd.update(np.array([_box(100.0, 200.0)]), np.array([0.40]), 1)
    assert len(fwd.tracks) == 1


def test_a_track_is_unconfirmed_until_min_hits():
    fwd = _fwd(min_hits=3)
    for i in range(2):
        fwd.update(np.array([_box(100.0 + i, 200.0)]), np.array([0.9]), i)
    assert [t.confirmed for t in fwd.tracks] == [False]
    assert fwd.finished() == []

    fwd.update(np.array([_box(102.0, 200.0)]), np.array([0.9]), 2)
    assert [t.confirmed for t in fwd.tracks] == [True]
    assert len(fwd.finished()) == 1


def test_min_hits_of_one_confirms_at_birth():
    fwd = _fwd(min_hits=1)
    fwd.update(np.array([_box(100.0, 200.0)]), np.array([0.9]), 0)
    assert fwd.tracks[0].confirmed


def test_a_low_score_detection_continues_an_existing_track():
    """The second stage: the shopper is still there, the detector merely got unsure."""
    fwd = _fwd()
    for i in range(3):
        fwd.update(np.array([_box(100.0 + 2 * i, 200.0)]), np.array([0.9]), i)
    assert len(fwd.tracks) == 1
    born = fwd.tracks[0].frag_id

    fwd.update(np.array([_box(106.0, 200.0)]), np.array([0.25]), 3)
    assert len(fwd.tracks) == 1, "a low-score box near a live track must not start a second"
    assert fwd.tracks[0].frag_id == born
    assert fwd.tracks[0].age == 0, "it was an observation, so the track is not coasting"


def test_the_low_band_is_gated_more_strictly_than_the_high_one():
    """`iou_thr_low` above `iou_thr` on purpose: a low-score box is a noisier box, and
    this stage bridges occlusion rather than growing a box into the fixture behind it."""
    fwd = _fwd(iou_thr=0.1, iou_thr_low=0.9)
    for i in range(3):
        fwd.update(np.array([_box(100.0, 200.0)]), np.array([0.9]), i)
    # Far enough that IoU clears 0.1 but not 0.9.
    fwd.update(np.array([_box(125.0, 200.0)]), np.array([0.25]), 3)
    assert fwd.tracks[0].age == 1, "the low-band box should have been rejected by the gate"


# ------------------------------------------------------------- coasting and retirement


def test_a_missed_detection_coasts_on_the_motion_model():
    fwd = _fwd()
    for i in range(4):
        fwd.update(np.array([_box(100.0 + 10 * i, 200.0)]), np.array([0.9]), i)
    last_seen = fwd.tracks[0].boxes[-1].copy()

    fwd.update(np.zeros((0, 4)), np.zeros((0,)), 4)
    t = fwd.tracks[0]
    assert t.age == 1
    assert t.kalman.box[0] > last_seen[0], "coasting should carry the box along its velocity"
    assert t.frames[-1] == 3, "a coasted frame is not an observation and is not recorded"


def test_a_track_retires_after_max_age_and_a_confirmed_one_is_kept():
    fwd = _fwd(max_age=3)
    for i in range(3):
        fwd.update(np.array([_box(100.0, 200.0)]), np.array([0.9]), i)
    assert fwd.tracks[0].confirmed

    for i in range(3, 3 + 4):
        fwd.update(np.zeros((0, 4)), np.zeros((0,)), i)
    assert fwd.tracks == [], "past max_age it is no longer live"
    assert len(fwd.retired) == 1, "and being confirmed, it is kept"
    assert len(fwd.finished()) == 1


def test_an_unconfirmed_track_that_dies_is_discarded_rather_than_retired():
    """It never met `min_hits`, so it was never evidence of anything."""
    fwd = _fwd(max_age=1, min_hits=3)
    fwd.update(np.array([_box(100.0, 200.0)]), np.array([0.9]), 0)
    for i in range(1, 4):
        fwd.update(np.zeros((0, 4)), np.zeros((0,)), i)
    assert fwd.tracks == []
    assert fwd.retired == []
    assert fwd.finished() == []


def test_finished_returns_retired_and_still_live_confirmed_tracks_once_each():
    fwd = _fwd(max_age=1)
    for i in range(3):  # track A, which will be allowed to die
        fwd.update(np.array([_box(100.0, 200.0)]), np.array([0.9]), i)
    for i in range(3, 6):
        fwd.update(np.zeros((0, 4)), np.zeros((0,)), i)
    for i in range(6, 9):  # track B, still live at the end
        fwd.update(np.array([_box(500.0, 200.0)]), np.array([0.9]), i)

    done = fwd.finished()
    assert len(done) == 2
    assert len({t.frag_id for t in done}) == 2, "no fragment may be reported twice"


# --------------------------------------------------------------------------- two tracks


def test_two_separated_people_keep_separate_ids():
    fwd = _fwd()
    for i in range(4):
        boxes = np.array([_box(100.0 + 5 * i, 200.0), _box(600.0 - 5 * i, 210.0)])
        fwd.update(boxes, np.array([0.9, 0.9]), i)
    assert len(fwd.tracks) == 2
    assert len({t.frag_id for t in fwd.tracks}) == 2


def test_fragment_ids_start_at_one_and_never_repeat():
    fwd = _fwd(max_age=0)
    seen = []
    for i in range(4):
        fwd.update(np.array([_box(100.0 + 400 * i, 200.0)]), np.array([0.9]), i)
        seen += [t.frag_id for t in fwd.tracks]
    assert min(seen) == 1
    assert len(set(seen)) == len(seen)


def test_an_empty_update_on_an_empty_tracker_is_not_an_error():
    fwd = _fwd()
    fwd.update(np.zeros((0, 4)), np.zeros((0,)), 0)
    assert fwd.tracks == []
    assert fwd.finished() == []


# ----------------------------------------------------------------------------- storage


def test_fragment_carries_the_crop_lists_the_tracker_never_reads():
    """`stash_crops` moved to `offline_tracks.py` and fills these; nothing here looks at
    them, which is why they could stay as plain storage rather than following it out."""
    f = Fragment(1, Kalman(_box(100.0, 200.0), 1.0))
    assert f.review_crops == [] and f.embed_crops == []
    fwd = _fwd()
    fwd.update(np.array([_box(100.0, 200.0)]), np.array([0.9]), 0)
    assert fwd.tracks[0].review_crops == []


@pytest.mark.parametrize("n", [1, 3, 9])
def test_update_accepts_the_shapes_a_decoder_hands_it(n):
    """Detection heads return (N, 4) float64 and (N,) float64; that is the only contract."""
    fwd = _fwd()
    boxes = np.stack([_box(50.0 + 120.0 * i, 200.0) for i in range(n)])
    fwd.update(boxes, np.full(n, 0.9), 0)
    assert len(fwd.tracks) == n
