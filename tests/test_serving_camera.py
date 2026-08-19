"""Per-camera serving state: isolation between cameras, EMA semantics, per-class
thresholds, and track-box consumption."""

from __future__ import annotations

import json

import numpy as np
import pytest

from syncai_hydranet.serving.camera import (
    BIRTH_REF,
    DEFAULT_THRESHOLDS,
    CameraState,
    ClassThresholds,
    confirmed_track_boxes,
)

HW = (8, 10)
DET = ["person", "bag", "boxed_stock", "device"]


def make_state(name="cam00", **kw) -> CameraState:
    kw.setdefault("num_terrain_classes", 6)
    kw.setdefault("canvas_hw", HW)
    kw.setdefault("det_classes", DET)
    return CameraState(camera=name, **kw)


class StubTracker:
    """Records what the state feeds it; exposes the .tracks the consumer reads."""

    def __init__(self):
        self.calls = []
        self.tracks = []

    def update(self, boxes, scores, frame_idx):
        self.calls.append((np.asarray(boxes).copy(), np.asarray(scores).copy(), frame_idx))


class StubTrack:
    def __init__(self, frag_id, box, age=0, confirmed=True, frames=(0,)):
        class _K:
            pass

        self.frag_id = frag_id
        self.boxes = [np.asarray(box, np.float64)]
        self.kalman = _K()
        self.kalman.box = np.asarray(box, np.float64) + 5.0  # distinguishable prediction
        self.age = age
        self.confirmed = confirmed
        self.frames = list(frames)


# -- isolation ---------------------------------------------------------------


def test_two_cameras_share_no_state():
    a, b = (
        make_state("a", tracker_factory=StubTracker),
        make_state("b", tracker_factory=StubTracker),
    )
    assert a.tracker is not b.tracker
    assert a.thresholds is not b.thresholds
    labels_a = np.zeros(HW, np.uint8)
    labels_b = np.full(HW, 3, np.uint8)
    empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64))
    a.update(1, labels_a, *empty)
    b.update(1, labels_b, *empty)
    assert (a.ema_labels(labels_a) == 0).all()
    assert (b.ema_labels(labels_b) == 3).all()
    assert a.frames_seen == 1 and b.frames_seen == 1


def test_default_thresholds_are_copied_not_aliased():
    a, b = make_state("a"), make_state("b")
    a.thresholds["person"] = ClassThresholds(birth=0.9, keep=0.5)
    assert b.thresholds["person"] == DEFAULT_THRESHOLDS["person"]


# -- EMA semantics -----------------------------------------------------------


def test_ema_first_frame_is_taken_as_is():
    s = make_state()
    labels = np.arange(80, dtype=np.uint8).reshape(HW) % 6
    assert (s.ema_labels(labels) == labels).all()


def test_ema_flip_takes_two_frames_at_alpha_035():
    """argmax flips once (1-alpha)^n < 0.5: at alpha=0.35 that is n=2, the measured
    0.4 s lag runs/stable01 accepted for the 1.50% -> 0.67% flip-rate win."""
    s = make_state(ema_alpha=0.35)
    zeros = np.zeros(HW, np.uint8)
    ones = np.ones(HW, np.uint8)
    s.ema_labels(zeros)
    assert (s.ema_labels(ones) == 0).all()  # one observation is out-voted by history
    assert (s.ema_labels(ones) == 1).all()  # the second carries the flip


def test_ema_rejects_the_wrong_canvas_shape():
    s = make_state()
    with pytest.raises(ValueError, match="cam00"):
        s.ema_labels(np.zeros((4, 4), np.uint8))


# -- per-class thresholds ----------------------------------------------------


def test_thresholds_validate():
    with pytest.raises(ValueError):
        ClassThresholds(birth=0.2, keep=0.3)  # keep above birth
    with pytest.raises(ValueError):
        ClassThresholds(birth=0.5, keep=0.0)


def test_filter_uses_each_class_own_keep_threshold():
    s = make_state(
        thresholds={
            "person": ClassThresholds(birth=0.35, keep=0.20),
            "boxed_stock": ClassThresholds(birth=0.14, keep=0.08),
        }
    )
    boxes = np.array([[0, 0, 10, 10]] * 3, np.float64)
    scores = np.array([0.10, 0.10, 0.05])
    labels = np.array([0, 2, 2], np.int64)  # person, boxed_stock, boxed_stock
    _fb, fs, fl = s.filter_and_scale(boxes, scores, labels)
    # person at 0.10 is under its 0.20 keep; boxed_stock at 0.10 survives its 0.08.
    assert fl.tolist() == [2]
    # Scaled onto the tracker's band: 0.10 * (0.35 / 0.14) = 0.25.
    assert fs == pytest.approx([0.25])


def test_scaling_maps_a_class_birth_score_onto_the_reference_edge():
    s = make_state(thresholds={"device": ClassThresholds(birth=0.7, keep=0.4)})
    boxes = np.array([[0, 0, 5, 5]], np.float64)
    _, fs, _ = s.filter_and_scale(boxes, np.array([0.7]), np.array([3], np.int64))
    assert fs == pytest.approx([BIRTH_REF])


def test_update_feeds_the_tracker_scaled_scores_and_counts_frames():
    s = make_state(tracker_factory=StubTracker)
    labels = np.zeros(HW, np.uint8)
    boxes = np.array([[0, 0, 4, 4]], np.float64)
    s.update(7, labels, boxes, np.array([0.35]), np.array([0], np.int64))
    assert len(s.tracker.calls) == 1
    _, scores, frame_idx = s.tracker.calls[0]
    assert frame_idx == 0 and scores == pytest.approx([0.35])
    assert s.frames_seen == 1 and s.last_seq == 7


# -- calibration handle ------------------------------------------------------


def test_missing_calib_is_none_and_wrong_schema_raises(tmp_path):
    assert make_state(calib_path=tmp_path / "absent.json").calib is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "something-else"}))
    with pytest.raises(ValueError, match="schema"):
        make_state(calib_path=bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"schema": "hydranet-onboard-calib/v1", "height_m": 2.5}))
    assert make_state(calib_path=good).calib["height_m"] == 2.5


# -- track-box consumption ---------------------------------------------------


def test_confirmed_track_boxes_observed_vs_coasting_and_majority_vote():
    tr = StubTracker()
    observed = StubTrack(1, [2, 2, 6, 6], age=0)
    observed.label_votes = [0, 0, 1]  # majority person
    coasting = StubTrack(2, [1, 1, 4, 4], age=2)
    coasting.label_votes = [2]
    unconfirmed = StubTrack(3, [0, 0, 3, 3], confirmed=False)
    tr.tracks = [observed, coasting, unconfirmed]
    boxes = np.array([[2, 2, 6, 6]], np.float64)
    out = confirmed_track_boxes(tr, np.array([1], np.int64), boxes, HW)
    assert [o["id"] for o in out] == [1, 2]
    assert out[0]["coasting"] is False and out[0]["label"] == 0
    # The coasting track renders its Kalman prediction, clipped to the canvas.
    assert out[1]["coasting"] is True
    assert out[1]["box"].tolist() == [6.0, 6.0, 9.0, 8.0]  # +5 prediction, clip (10, 8)


def test_incremental_ema_matches_the_naive_recursion_over_a_long_sequence():
    """The scaled/incremental implementation must be indistinguishable from the
    plain decay + one-hot add + argmax it replaced, including across the scale
    renormalisation (~46 frames at alpha 0.35)."""
    rng = np.random.default_rng(11)
    s = make_state(ema_alpha=0.35)
    alpha, c = 0.35, 6
    naive = None
    for _ in range(120):
        labels = rng.integers(0, c, size=HW).astype(np.uint8)
        flat = labels.reshape(-1)
        if naive is None:
            naive = np.zeros((c, labels.size), np.float64)
            naive[flat, np.arange(labels.size)] = 1.0
        else:
            naive *= 1 - alpha
            naive[flat, np.arange(labels.size)] += alpha
        expected = naive.argmax(axis=0).astype(np.uint8).reshape(HW)
        np.testing.assert_array_equal(s.ema_labels(labels), expected)
