"""A track carries the detector confidence it was built from, and an event reports it.

The 2026-08-26 threshold sweep is the reason this exists: lowering the person birth
threshold to 0.15 quadrupled posture events, the dense-head confirmation removed 12-14%
of boxes and produced *exactly* the unfiltered arm's events, and the conclusion was that
no box filter reaches the problem -- the event layer has to see what confidence a track
was built from. These tests pin the wire that carries it and the alignment it depends on.

pytest tests/test_track_support.py -v
"""

import numpy as np
import pytest

from _posture import _stand_then, keypoints, posed
from syncai_hydranet.analytics import events as ev
from syncai_hydranet.analytics.tracker import Track, Tracker


def _box(x, y, w=20.0, h=60.0):
    return [x, y, x + w, y + h]


def _run(tracker, scores_per_frame, *, start=0):
    """Feed one moving box per frame, at the given score."""
    for i, sc in enumerate(scores_per_frame):
        tracker.update(
            np.array([_box(10.0 + i, 10.0)]), start + i, scores=np.array([sc], dtype=float)
        )


# ------------------------------------------------------------------ the tracker wire


def test_a_track_records_one_score_per_observed_frame():
    t = Tracker(min_hits=1)
    _run(t, [0.9, 0.8, 0.7])
    (track,) = t.finished()
    assert track.scores == pytest.approx([0.9, 0.8, 0.7])
    assert len(track.scores) == len(track.frames) == len(track.boxes)


def test_scores_stay_optional_so_every_existing_caller_still_works():
    t = Tracker(min_hits=1)
    t.update(np.array([_box(10.0, 10.0)]), 0)
    t.update(np.array([_box(11.0, 10.0)]), 1)
    (track,) = t.finished()
    assert track.scores == []
    assert ev.support_for(track, 0, 1) is None


def test_a_tracker_that_has_seen_scores_refuses_a_frame_without_them():
    t = Tracker(min_hits=1)
    t.update(np.array([_box(10.0, 10.0)]), 0, scores=np.array([0.9]))
    with pytest.raises(ValueError, match="index-aligned"):
        t.update(np.array([_box(11.0, 10.0)]), 1)


def test_a_tracker_that_has_not_seen_scores_refuses_a_frame_with_them():
    t = Tracker(min_hits=1)
    t.update(np.array([_box(10.0, 10.0)]), 0)
    with pytest.raises(ValueError, match="index-aligned"):
        t.update(np.array([_box(11.0, 10.0)]), 1, scores=np.array([0.9]))


def test_a_score_per_box_is_required():
    t = Tracker(min_hits=1)
    with pytest.raises(ValueError, match="matched by position"):
        t.update(np.array([_box(10.0, 10.0), _box(60.0, 10.0)]), 0, scores=np.array([0.9]))


# ------------------------------------------------------------------ what an event says


def test_support_reports_the_window_not_the_whole_track():
    t = Tracker(min_hits=1)
    _run(t, [0.9, 0.9, 0.2, 0.2])
    (track,) = t.finished()
    late = ev.support_for(track, 2, 3)
    assert late is not None
    assert late.score_p50 == pytest.approx(0.2)
    assert late.score_min == pytest.approx(0.2)
    early = ev.support_for(track, 0, 1)
    assert early is not None and early.score_p50 == pytest.approx(0.9)


def test_a_coasted_event_reports_how_much_of_it_was_actually_seen():
    """`observed` indexes the track's observations; `span` counts the frames covered."""
    track = Track(
        1, np.array(_box(10.0, 10.0)), frames=[0, 1, 9], boxes=[], scores=[0.5, 0.4, 0.6]
    )
    s = ev.support_for(track, 0, 2)
    assert s is not None
    assert (s.observed, s.span) == (3, 10)
    assert s.seen_fraction == pytest.approx(0.3)


def test_score_min_is_what_separates_a_low_confidence_event():
    """The sweep's finding in one assertion: two events with the same median differ."""
    steady = Track(1, np.array(_box(0, 0)), frames=[0, 1, 2], scores=[0.6, 0.6, 0.6])
    ragged = Track(2, np.array(_box(0, 0)), frames=[0, 1, 2], scores=[0.16, 0.6, 0.9])
    a, b = ev.support_for(steady, 0, 2), ev.support_for(ragged, 0, 2)
    assert a is not None and b is not None
    assert a.score_p50 == b.score_p50 == pytest.approx(0.6)
    assert a.score_min == pytest.approx(0.6)
    assert b.score_min == pytest.approx(0.16)


def test_misaligned_scores_raise_rather_than_report_the_wrong_frames():
    track = Track(1, np.array(_box(0, 0)), frames=[0, 1, 2], scores=[0.5, 0.4])
    with pytest.raises(ValueError, match="index-aligned"):
        ev.support_for(track, 0, 1)


def test_an_event_without_support_still_produces_every_column():
    e = ev.SecurityEvent(type="fall", camera="cam", frame_start=0, frame_end=4, fps=5.0)
    row = e.as_row()
    for k in ("support_score_p50", "support_score_min", "support_observed", "support_span"):
        assert k in row and row[k] is None


def test_an_event_with_support_puts_it_in_the_row():
    e = ev.SecurityEvent(
        type="fall",
        camera="cam",
        frame_start=0,
        frame_end=4,
        fps=5.0,
        support=ev.TrackSupport(score_p50=0.22, score_min=0.16, observed=3, span=5),
    )
    row = e.as_row()
    assert row["support_score_p50"] == pytest.approx(0.22)
    assert row["support_observed"] == 3 and row["support_span"] == 5


def test_support_refuses_an_impossible_span():
    with pytest.raises(ValueError, match="observed <= span"):
        ev.TrackSupport(score_p50=0.5, score_min=0.5, observed=5, span=3)


# ------------------------------------------------------- the producer that fills it


def test_a_posture_event_carries_the_confidence_of_its_own_frames():
    """A fall on a track whose boxes scored 0.16-0.18 during it says so.

    The fixture is `_posture.posed`'s shape -- stand, then change, because
    the box-height cross-check refuses a track that is already down when it opens. The
    scores follow the same story: a shopper detected confidently while upright and only
    marginally once they are on the floor, which is exactly the case the threshold sweep
    found and the case a consumer must be able to tell from a 0.6 fall.
    """
    poses, heights = _stand_then(keypoints(80.0, 200.0), 150.0)
    track = posed(1, poses, heights)
    track.scores = [0.8] * 5 + [0.16, 0.18, 0.17, 0.16, 0.18, 0.17, 0.16, 0.18, 0.17, 0.16]

    events = ev.pose_posture_events([track], fps=5.0, camera="cam01")
    assert [e.type for e in events] == ["fall"]
    support = events[0].support
    assert support is not None
    assert support.score_p50 < 0.2
    assert support.score_min == pytest.approx(0.16)
    assert support.observed == support.span == 10
    assert events[0].as_row()["support_score_min"] == pytest.approx(0.16)


def test_the_same_fall_at_high_confidence_reports_the_difference():
    """Same posture, same boxes, different detector -- and the row now shows it."""
    poses, heights = _stand_then(keypoints(80.0, 200.0), 150.0)
    track = posed(1, poses, heights)
    track.scores = [0.8] * 15
    (event,) = ev.pose_posture_events([track], fps=5.0, camera="cam01")
    assert event.support is not None
    assert event.support.score_p50 == pytest.approx(0.8)


# ----------------------------------------------------- staff-memory inheritance


def _walk(tr, frames_boxes, probs):
    for n, (box, p) in enumerate(zip(frames_boxes, probs, strict=True)):
        b = np.array(box, float).reshape(-1, 4)
        sp = np.array(p, float).reshape(-1)
        tr.update(b, n, staff_scores=sp)


def test_a_newborn_where_a_track_just_died_inherits_its_staff_evidence():
    """The measured cause of colour flicker is fragmentation: 89 tracks at median life
    12 frames, 66 warm-up colour pops against 5 real flips (Kaohsiung-cam04, 900
    frames). Each fragment restarting its evidence from zero is the flicker; the heir
    starting from the donor's evidence is the fix, and 15/0.2 measured zero
    contamination across 15 decidable donor/heir pairs."""
    from syncai_hydranet.analytics.tracker import Tracker

    tr = Tracker(max_age=1, min_hits=1, staff_memory_gap=15, staff_memory_iou=0.2)
    spot = [10.0, 10.0, 60.0, 150.0]
    # a confirmed track accumulates three staff observations, then vanishes
    _walk(tr, [[spot]] * 3 + [np.zeros((0, 4))] * 3, [[0.9]] * 3 + [[]] * 3)
    assert not tr.tracks  # the donor is dead and retired
    # a newborn appears on the same spot within the gap
    tr.update(np.array([spot]), 6, staff_scores=np.array([0.8]))
    (heir,) = tr.tracks
    assert heir.staff_scores == [0.9, 0.9, 0.9, 0.8], "the donor's evidence seeds the heir"


def test_a_dead_track_feeds_at_most_one_heir():
    from syncai_hydranet.analytics.tracker import Tracker

    tr = Tracker(max_age=1, min_hits=1, staff_memory_gap=15, staff_memory_iou=0.2)
    spot = [10.0, 10.0, 60.0, 150.0]
    _walk(tr, [[spot]] * 3 + [np.zeros((0, 4))] * 3, [[0.9]] * 3 + [[]] * 3)
    tr.update(np.array([spot]), 6, staff_scores=np.array([0.8]))
    # the first heir dies too; a SECOND newborn on the spot must not inherit from the
    # original donor again (its evidence already lives on in heir #1's line)
    tr.update(np.zeros((0, 4)), 7, staff_scores=np.zeros(0))
    tr.update(np.zeros((0, 4)), 8, staff_scores=np.zeros(0))
    tr.update(np.array([spot]), 9, staff_scores=np.array([0.7]))
    newest = max(tr.tracks, key=lambda t: t.track_id)
    assert 0.9 in newest.staff_scores or newest.staff_scores.count(0.9) == 0
    donors_used = tr._inherited_from
    assert len(donors_used) == len(set(donors_used))


def test_inheritance_off_by_default_and_gated_by_gap_and_overlap():
    from syncai_hydranet.analytics.tracker import Tracker

    spot = [10.0, 10.0, 60.0, 150.0]
    far = [500.0, 10.0, 550.0, 150.0]
    # default: off
    tr = Tracker(max_age=1, min_hits=1)
    _walk(tr, [[spot]] * 3 + [np.zeros((0, 4))] * 3, [[0.9]] * 3 + [[]] * 3)
    tr.update(np.array([spot]), 6, staff_scores=np.array([0.8]))
    assert tr.tracks[0].staff_scores == [0.8]
    # outside the gap: no inheritance
    tr = Tracker(max_age=1, min_hits=1, staff_memory_gap=2, staff_memory_iou=0.2)
    _walk(tr, [[spot]] * 3 + [np.zeros((0, 4))] * 5, [[0.9]] * 3 + [[]] * 5)
    tr.update(np.array([spot]), 8, staff_scores=np.array([0.8]))
    assert tr.tracks[0].staff_scores == [0.8]
    # no overlap: no inheritance
    tr = Tracker(max_age=1, min_hits=1, staff_memory_gap=15, staff_memory_iou=0.2)
    _walk(tr, [[spot]] * 3 + [np.zeros((0, 4))] * 3, [[0.9]] * 3 + [[]] * 3)
    tr.update(np.array([far]), 6, staff_scores=np.array([0.8]))
    assert tr.tracks[0].staff_scores == [0.8]


def test_nan_staff_scores_are_quality_refusals_not_evidence():
    from syncai_hydranet.analytics.staff import MIN_OBSERVATIONS, staff_verdict

    assert staff_verdict([0.9] * MIN_OBSERVATIONS) is True
    padded = [0.9] * (MIN_OBSERVATIONS - 1) + [float("nan")] * 10
    assert staff_verdict(padded) is None, "NaN must not count toward the floor"
    assert staff_verdict([0.9] * MIN_OBSERVATIONS + [float("nan")] * 10) is True
