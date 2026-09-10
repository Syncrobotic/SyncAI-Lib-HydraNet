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

from syncai_hydranet.analytics import tracker
from syncai_hydranet.analytics.bytetrack import (
    Fragment,
    Kalman,
    OfflineForward,
    as_track,
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


# --------------------------------------------------------------- the two IoUs agree
#
# `tracker.iou` is (N,4) x (M,4) -> (N,M); `tracker.iou_pair` is one box against one, and
# is written out rather than wrapping the first because a pair through the vectorised form
# costs 14x (17.8 us against 1.2 us) and its callers are loops. Two implementations of one
# formula is the shape this repository keeps finding, so they are held equal here rather
# than by inspection: it is the second implementation that makes the first one's callers
# safe to move, and nothing else would notice them drifting.


def test_the_scalar_iou_agrees_with_the_vectorised_one():
    rng = np.random.default_rng(20260831)
    boxes = rng.uniform(0, 400, size=(60, 2))
    sizes = rng.uniform(1, 200, size=(60, 2))
    xyxy = np.hstack([boxes, boxes + sizes])
    grid = tracker.iou(xyxy, xyxy)
    for i in range(len(xyxy)):
        for j in range(len(xyxy)):
            assert tracker.iou_pair(xyxy[i], xyxy[j]) == pytest.approx(grid[i, j], abs=1e-12), (
                f"boxes {i} and {j} disagree: {tracker.iou_pair(xyxy[i], xyxy[j])} "
                f"against the vectorised {grid[i, j]}"
            )


@pytest.mark.parametrize(
    "a, b",
    [
        ([0, 0, 10, 10], [10, 0, 20, 10]),  # edge-touching: no area, not overlap
        ([0, 0, 10, 10], [20, 20, 30, 30]),  # disjoint
        ([0, 0, 10, 10], [0, 0, 10, 10]),  # identical
        ([0, 0, 10, 10], [2, 2, 4, 4]),  # contained
        ([0, 0, 0, 0], [0, 0, 10, 10]),  # degenerate: zero area, zero union contribution
        ([5, 5, 5, 10], [0, 0, 10, 10]),  # degenerate: zero width
    ],
)
def test_the_two_ious_agree_on_the_degenerate_boxes_too(a, b):
    """Random boxes never produce these, and they are where two formulas usually part."""
    pair = np.array([a], dtype=float), np.array([b], dtype=float)
    assert tracker.iou_pair(a, b) == pytest.approx(tracker.iou(*pair)[0, 0], abs=1e-12)


# -- the same band, without the Kalman ---------------------------------------
#
# 7.11 measured `bytetrack` against the shipped `tracker.Tracker` and could not say which
# half carried the result, because this file brings a high/low band *and* a Kalman filter
# and `tracker.py` refuses the filter on stated grounds. `Tracker(birth_thr=...)` is the
# band alone, so the two can finally be separated.


def test_tracker_birth_thr_admits_a_low_box_to_a_track_and_not_to_a_new_one():
    tr = tracker.Tracker(iou_threshold=0.3, max_age=5, min_hits=1, birth_thr=0.35)
    box = np.array([[10.0, 10.0, 50.0, 90.0]])

    tr.update(box, 0, scores=np.array([0.9]))
    assert len(tr.tracks) == 1

    tr.update(box, 1, scores=np.array([0.22]))
    assert len(tr.tracks) == 1, "a low box on the track must continue it"
    assert tr.tracks[0].frames == [0, 1]
    assert tr.tracks[0].scores == [0.9, 0.22]

    far = np.array([[500.0, 300.0, 540.0, 380.0]])
    tr.update(np.vstack([box, far]), 2, scores=np.array([0.22, 0.22]))
    assert len(tr.tracks) == 1, "a box below the birth edge must not open a track"


def test_tracker_without_birth_thr_is_unchanged():
    """The default is None and every number this project has published was measured under
    it, so the band has to be invisible until a caller asks for it."""
    a = tracker.Tracker(iou_threshold=0.3, max_age=5, min_hits=1)
    b = tracker.Tracker(iou_threshold=0.3, max_age=5, min_hits=1, birth_thr=None)
    rng = np.random.default_rng(20260908)
    for frame in range(12):
        n = int(rng.integers(0, 5))
        boxes = rng.uniform(0, 400, (n, 4))
        boxes[:, 2:] += boxes[:, :2] + 20
        sc = rng.uniform(0.05, 0.99, n)
        a.update(boxes, frame, scores=sc)
        b.update(boxes, frame, scores=sc)
    assert [t.track_id for t in a.tracks] == [t.track_id for t in b.tracks]
    assert [t.frames for t in a.tracks] == [t.frames for t in b.tracks]


def test_tracker_birth_thr_refuses_a_frame_with_no_scores():
    """Defaulting to "let it birth" would disable the band silently, on the frames where
    a caller forgot -- the failure mode this repository keeps paying for."""
    tr = tracker.Tracker(min_hits=1, birth_thr=0.35)
    with pytest.raises(ValueError, match="no scores"):
        tr.update(np.array([[0.0, 0.0, 20.0, 40.0]]), 0)


def test_tracker_birth_thr_refuses_a_value_that_is_not_a_score():
    assert tracker.Tracker(birth_thr=0.0).birth_thr == 0.0
    for bad in (1.5, -0.1):
        with pytest.raises(ValueError, match="birth_thr"):
            tracker.Tracker(birth_thr=bad)


# ------------------------------------------------- what a fragment carries out of here


def test_a_fragment_records_one_score_per_observed_frame():
    """The contract `tracker.Track` already states for its own `scores`, and which this
    tracker did not keep -- it took `scores` on every `update` and stored none of them.

    docs/PLAN.md step 4 is why it has to: the measured conclusion there is that the next
    mechanism is not another box filter, it is that *the event layer has to see the
    detection confidence a track was built from*. `world.WorldObject.score` carries that
    and could not be filled from here.
    """
    fwd = _fwd(min_hits=1)
    fwd.update(np.stack([_box(100.0, 200.0)]), np.array([0.90]), 0)
    fwd.update(np.stack([_box(102.0, 200.0)]), np.array([0.42]), 1)  # high band
    fwd.update(np.stack([_box(104.0, 200.0)]), np.array([0.25]), 2)  # low band
    (t,) = fwd.tracks
    assert t.scores == pytest.approx([0.90, 0.42, 0.25])
    assert len(t.scores) == len(t.frames) == len(t.boxes)


def test_a_coasting_frame_adds_no_score_because_it_is_not_an_observation():
    """`scores` is index-aligned with `frames`, and `frames` records only observations --
    so a coasted frame must not push a value in, or every later index is off by one."""
    fwd = _fwd(min_hits=1)
    fwd.update(np.stack([_box(100.0, 200.0)]), np.array([0.90]), 0)
    fwd.update(np.zeros((0, 4)), np.zeros(0), 1)  # nothing seen
    fwd.update(np.stack([_box(104.0, 200.0)]), np.array([0.55]), 2)
    (t,) = fwd.tracks
    assert t.frames == [0, 2]
    assert t.scores == pytest.approx([0.90, 0.55])


def test_as_track_hands_the_producer_what_it_asks_for():
    """`world.world_frame` takes `tracker.Track`; this tracker makes `Fragment`."""
    fwd = _fwd(min_hits=1)
    fwd.update(np.stack([_box(100.0, 200.0)]), np.array([0.90]), 0)
    fwd.update(np.stack([_box(110.0, 200.0)]), np.array([0.70]), 1)
    (frag,) = fwd.tracks

    t = as_track(frag)
    assert isinstance(t, tracker.Track)
    assert t.track_id == frag.frag_id
    assert t.frames == frag.frames and t.scores == pytest.approx(frag.scores)
    assert t.age == 0
    # Observed: the current box is the observation, and `foot` reads it.
    assert t.box == pytest.approx(frag.boxes[-1])
    assert t.foot == pytest.approx([110.0, 250.0])


def test_a_coasting_fragment_converts_to_its_prediction_not_its_last_sighting():
    """`serving.camera.confirmed_track_boxes`'s rule, so a missed detection does not
    blink the position off -- and `world_frame` reads `age` to report `observed=False`
    for the same frame, so the two agree by construction."""
    fwd = _fwd(min_hits=1)
    fwd.update(np.stack([_box(100.0, 200.0)]), np.array([0.90]), 0)
    fwd.update(np.stack([_box(110.0, 200.0)]), np.array([0.90]), 1)
    fwd.update(np.zeros((0, 4)), np.zeros(0), 2)
    (frag,) = fwd.tracks

    t = as_track(frag)
    assert t.age == 1
    assert t.box == pytest.approx(frag.kalman.box)
    assert t.box != pytest.approx(frag.boxes[-1]), (
        "a prediction that equals the last "
        "sighting would make this test blind to the rule it is checking"
    )


# -- an appearance gate on re-association ------------------------------------
#
# `SIMPLIFICATIONS` has always said this tracker has no appearance model and that two
# shoppers who swap while overlapping swap ids. Measured on Taichung-cam04's 14:31 clip
# (2026-09-08): appearance distance between consecutive observations of one track sits at
# p50 0.021 / p99 0.225, and between two different shoppers in the same frame at p10
# 0.376 -- disjoint, which is what lets a threshold between them mean something. 17 of
# 108 tracks carried at least one change of person, corroborated by an impossible floor
# speed or a doubled box. `appearance_thr` gates the step where 48% of them happen: the
# re-association after a gap.

RED = np.array([0.9, 0.05, 0.05])
BLUE = np.array([0.05, 0.05, 0.9])


def test_appearance_distance_separates_two_histograms_and_refuses_a_shape_mismatch():
    assert tracker.appearance_distance(RED, RED) == pytest.approx(0.0)
    assert tracker.appearance_distance(RED, BLUE) > 0.5
    with pytest.raises(ValueError, match="differ in length"):
        tracker.appearance_distance(RED, np.array([0.5, 0.5]))


def _coast_then_return(thr, second_look):
    """One track, a two-frame gap, then a box back where it was wearing `second_look`."""
    tr = tracker.Tracker(iou_threshold=0.3, max_age=5, min_hits=1, appearance_thr=thr)
    box = np.array([[10.0, 10.0, 50.0, 90.0]])
    tr.update(box, 0, appearance=RED[None, :])
    tr.update(np.zeros((0, 4)), 1, appearance=np.zeros((0, 3)))
    tr.update(np.zeros((0, 4)), 2, appearance=np.zeros((0, 3)))
    tr.update(box, 3, appearance=second_look[None, :])
    return tr


def test_a_returning_box_that_looks_like_someone_else_starts_its_own_track():
    tr = _coast_then_return(0.3, BLUE)
    assert len(tr.tracks) == 2, "the gate must refuse the match, not reassign it"
    assert [t.frames for t in tr.tracks] == [[0], [3]]


def test_a_returning_box_that_still_looks_like_itself_is_re_associated():
    """The gate has to be a gate and not a ban: same clothes, same person, one track."""
    tr = _coast_then_return(0.3, RED)
    assert len(tr.tracks) == 1
    assert tr.tracks[0].frames == [0, 3]


def test_the_gate_does_not_fire_on_consecutive_frames():
    """Only the step after a gap is checked. A shopper who turns round between two frames
    changes colour and must keep their id -- gating that would cost tracks to buy nothing,
    because 48% of the measured identity changes are re-associations and 1% of all steps
    are."""
    tr = tracker.Tracker(iou_threshold=0.3, max_age=5, min_hits=1, appearance_thr=0.3)
    box = np.array([[10.0, 10.0, 50.0, 90.0]])
    tr.update(box, 0, appearance=RED[None, :])
    tr.update(box, 1, appearance=BLUE[None, :])
    assert len(tr.tracks) == 1
    assert tr.tracks[0].frames == [0, 1]


def test_appearance_is_recorded_even_when_the_gate_is_off():
    """Recorded-but-unused, like `scores`: a consumer asking "did this track change
    person" needs the series, and it cannot be recovered afterwards."""
    tr = tracker.Tracker(iou_threshold=0.3, max_age=5, min_hits=1)
    box = np.array([[10.0, 10.0, 50.0, 90.0]])
    tr.update(box, 0, appearance=RED[None, :])
    tr.update(box, 1, appearance=BLUE[None, :])
    assert len(tr.tracks[0].appearance) == 2
    assert tr.tracks[0].appearance[1] == pytest.approx(BLUE)


def test_tracker_without_appearance_thr_is_unchanged():
    a = tracker.Tracker(iou_threshold=0.3, max_age=5, min_hits=1)
    b = tracker.Tracker(iou_threshold=0.3, max_age=5, min_hits=1, appearance_thr=None)
    rng = np.random.default_rng(20260909)
    for frame in range(12):
        n = int(rng.integers(0, 5))
        boxes = rng.uniform(0, 400, (n, 4))
        boxes[:, 2:] += boxes[:, :2] + 20
        a.update(boxes, frame)
        b.update(boxes, frame)
    assert [t.frames for t in a.tracks] == [t.frames for t in b.tracks]


def test_appearance_thr_refuses_a_frame_with_no_appearance():
    """Otherwise the gate silently passes exactly the matches it exists to check."""
    tr = tracker.Tracker(iou_threshold=0.3, max_age=5, min_hits=1, appearance_thr=0.3)
    with pytest.raises(ValueError, match="carried no appearance"):
        tr.update(np.array([[10.0, 10.0, 50.0, 90.0]]), 0)


def test_a_negative_threshold_is_refused():
    with pytest.raises(ValueError, match="distance >= 0"):
        tracker.Tracker(appearance_thr=-0.1)


def test_appearance_is_all_frames_or_none():
    tr = tracker.Tracker(iou_threshold=0.3, max_age=5, min_hits=1)
    box = np.array([[10.0, 10.0, 50.0, 90.0]])
    tr.update(box, 0, appearance=RED[None, :])
    with pytest.raises(ValueError, match="appearance"):
        tr.update(box, 1)


def test_a_nan_descriptor_is_carried_and_never_splits_a_track():
    """`appearance.torso_histograms` writes NaN for a box too small to describe; at the
    gate NaN exceeds no threshold, so the box continues its track across a gap."""
    box = np.array([[10.0, 10.0, 50.0, 100.0]])
    tr = tracker.Tracker(iou_threshold=0.3, max_age=5, min_hits=1, appearance_thr=0.1)
    tr.update(box, 0, appearance=RED[None, :])
    tr.update(np.zeros((0, 4)), 1, appearance=np.zeros((0, 3)))
    tr.update(box, 2, appearance=np.full((1, 3), np.nan))
    assert len(tr.tracks) == 1 and tr.tracks[0].frames == [0, 2]
    assert np.isnan(tracker.appearance_distance(RED, np.full(3, np.nan)))


# -- the appearance gate on the two-stage tracker (2026-09-10) --------------------------
#
# The gate `tracker.Tracker` gained on 2026-09-08 now lives here too, because this is the
# tracker the serving path runs and the one step 6 runs from today. Same rule: only a
# re-association after a gap is gated, a refusal is recorded with its frame, and the
# refused detection falls through to a birth (high band) or nothing (low band).

GAP_BOX = np.array([[10.0, 10.0, 50.0, 100.0]])


def gated(thr=0.3, min_hits=1):
    return OfflineForward(0.35, 0.20, 0.3, 0.4, 5, min_hits, 5.0, appearance_thr=thr)


def test_a_consecutive_frame_is_never_gated_whatever_it_looks_like():
    tr = gated()
    tr.update(GAP_BOX, np.array([0.9]), 0, appearance=RED[None, :])
    tr.update(GAP_BOX, np.array([0.9]), 1, appearance=BLUE[None, :])
    assert len(tr.tracks) == 1 and tr.tracks[0].frames == [0, 1] and tr.refusals == []


def test_after_a_gap_a_different_look_is_refused_recorded_and_starts_its_own_track():
    tr = gated()
    tr.update(GAP_BOX, np.array([0.9]), 0, appearance=RED[None, :])
    tr.update(np.zeros((0, 4)), np.zeros(0), 1, appearance=np.zeros((0, 3)))
    tr.update(GAP_BOX, np.array([0.9]), 2, appearance=BLUE[None, :])
    ids = sorted(t.frag_id for t in tr.tracks)
    assert ids == [1, 2], "the refused detection was a high-band box, so it was born"
    (r,) = tr.refusals
    assert (r.frame, r.frag_id, r.age, r.band) == (2, 1, 2, "high")
    assert r.distance > 0.3 and np.allclose(r.box, GAP_BOX[0])


def test_after_a_gap_the_same_look_is_taken_and_the_descriptor_is_kept_aligned():
    tr = gated()
    tr.update(GAP_BOX, np.array([0.9]), 0, appearance=RED[None, :])
    tr.update(np.zeros((0, 4)), np.zeros(0), 1, appearance=np.zeros((0, 3)))
    tr.update(GAP_BOX, np.array([0.9]), 2, appearance=RED[None, :])
    (t,) = tr.tracks
    assert t.frames == [0, 2] and len(t.appearance) == 2 and tr.refusals == []
    assert len(as_track(t).appearance) == 2


def test_a_low_band_refusal_starts_nothing():
    tr = gated()
    tr.update(GAP_BOX, np.array([0.9]), 0, appearance=RED[None, :])
    tr.update(np.zeros((0, 4)), np.zeros(0), 1, appearance=np.zeros((0, 3)))
    tr.update(GAP_BOX, np.array([0.25]), 2, appearance=BLUE[None, :])
    assert [t.frag_id for t in tr.tracks] == [1] and tr.tracks[0].frames == [0]
    assert tr.refusals[0].band == "low"


def test_a_nan_descriptor_never_splits_on_the_two_stage_tracker_either():
    tr = gated()
    tr.update(GAP_BOX, np.array([0.9]), 0, appearance=RED[None, :])
    tr.update(np.zeros((0, 4)), np.zeros(0), 1, appearance=np.zeros((0, 3)))
    tr.update(GAP_BOX, np.array([0.9]), 2, appearance=np.full((1, 3), np.nan))
    assert len(tr.tracks) == 1 and tr.tracks[0].frames == [0, 2]


def test_the_gate_refuses_a_frame_without_descriptors_and_a_switch_midway():
    tr = gated()
    with pytest.raises(ValueError, match="no appearance vectors"):
        tr.update(GAP_BOX, np.array([0.9]), 0)
    tr2 = OfflineForward(0.35, 0.20, 0.3, 0.4, 5, 1, 5.0)
    tr2.update(GAP_BOX, np.array([0.9]), 0, appearance=RED[None, :])
    with pytest.raises(ValueError, match="index-aligned"):
        tr2.update(GAP_BOX, np.array([0.9]), 1)


def test_without_a_threshold_the_two_stage_tracker_is_unchanged_and_records_looks():
    tr = OfflineForward(0.35, 0.20, 0.3, 0.4, 5, 1, 5.0)
    tr.update(GAP_BOX, np.array([0.9]), 0, appearance=RED[None, :])
    tr.update(np.zeros((0, 4)), np.zeros(0), 1, appearance=np.zeros((0, 3)))
    tr.update(GAP_BOX, np.array([0.9]), 2, appearance=BLUE[None, :])
    assert len(tr.tracks) == 1 and len(tr.tracks[0].appearance) == 2 and tr.refusals == []


def test_shipped_forward_is_the_serving_operating_point_at_the_consumption_rate():
    from syncai_hydranet.analytics.bytetrack import (
        MOT17_FPS,
        SHIPPED_IOU,
        SHIPPED_IOU_LOW,
        SHIPPED_MAX_AGE,
        SHIPPED_MIN_HITS,
        shipped_forward,
    )
    from syncai_hydranet.serving.camera import BIRTH_REF, KEEP_REF

    tr = shipped_forward(5.0, appearance_thr=0.39)
    assert (tr.high_thr, tr.low_thr) == (BIRTH_REF, KEEP_REF)
    assert (tr.iou_thr, tr.iou_thr_low, tr.max_age, tr.min_hits) == (
        SHIPPED_IOU, SHIPPED_IOU_LOW, SHIPPED_MAX_AGE, SHIPPED_MIN_HITS,
    )  # fmt: skip
    assert tr.vel_scale == MOT17_FPS / 5.0 and tr.appearance_thr == 0.39
    with pytest.raises(ValueError):
        shipped_forward(0.0)


def test_the_clip_adapter_speaks_track_clip_and_returns_tracks():
    from syncai_hydranet.analytics.bytetrack import TwoStageForClip, shipped_forward

    tr = TwoStageForClip(shipped_forward(5.0))
    for f in range(3):
        tr.update(GAP_BOX, f, scores=np.array([0.9]))
    (t,) = tr.finished()
    assert type(t).__name__ == "Track" and t.frames == [0, 1, 2] and t.confirmed
    with pytest.raises(ValueError, match="scores"):
        tr.update(GAP_BOX, 3)
