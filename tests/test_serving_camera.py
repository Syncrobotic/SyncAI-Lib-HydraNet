"""Per-camera serving state: isolation between cameras, EMA semantics, per-class
thresholds, and track-box consumption."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from syncai_hydranet.serving.camera import (
    BIRTH_REF,
    DEFAULT_THRESHOLDS,
    CameraState,
    ClassThresholds,
    confirmed_track_boxes,
    load_thresholds,
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
    good.write_text(
        json.dumps(
            {
                "schema": "hydranet-onboard-calib/v1",
                "height_m": 2.5,
                "scale": 0.64,
                "scale_source": "person_height_median_vs_1.7m_prior_n30",
            }
        )
    )
    assert make_state(calib_path=good).calib["height_m"] == 2.5


def test_a_scan_with_no_measured_scale_is_not_calibrated(tmp_path):
    """The refusal `syncai_bev3d.commissioning` makes, arriving one step earlier.

    That converter declines to write a `camera.json` from a scan whose scale is
    unmeasured, because the scan has a shape and no metres and a file written from it
    "would put confident wrong metres under every downstream event". This path read the
    *upstream* artefact and checked only the schema string, so twelve of the fleet's
    twenty-three scans loaded here and `serve_pilot.py` counted them in its
    "N with calibration" readiness line.
    """
    import warnings

    scan = tmp_path / "unmeasured.json"
    scan.write_text(
        json.dumps(
            {
                "schema": "hydranet-onboard-calib/v1",
                "height_m": 2.5,
                "pitch_deg": 42.0,
                "scale": None,
                "scale_source": "unmeasured",
            }
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = make_state(calib_path=scan)
    assert state.calib is None, "a scan with no metres must not read as calibrated"
    said = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert said and "no metres" in said[0], f"demoted in silence: {caught}"
    # The orientation it does carry is real; the refusal is about scale alone, so a
    # scan that gains a measured scale later loads unchanged.
    scan.write_text(
        json.dumps(
            {
                "schema": "hydranet-onboard-calib/v1",
                "height_m": 2.5,
                "pitch_deg": 42.0,
                "scale": 0.61,
                "scale_source": "person_height_median_vs_1.7m_prior_n28",
            }
        )
    )
    assert make_state(calib_path=scan).calib["scale"] == 0.61


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


# ----------------------------------------------------- the threshold book, per camera


def _book(tmp_path, cameras: dict) -> Path:
    body = {
        "schema": "hydranet-thresholds/v1",
        "checkpoint": "runs/whatever/last.pt",
        "default": {
            "person": {"birth": 0.15, "keep": 0.10},
            "bag": {"birth": 0.35, "keep": 0.20},
        },
        "cameras": cameras,
    }
    p = tmp_path / "thresholds.json"
    p.write_text(json.dumps(body))
    return p


def test_a_camera_without_an_entry_runs_the_fleet_default(tmp_path):
    book = load_thresholds(_book(tmp_path, {}))
    assert book.for_camera("Taichung-cam01")["person"].birth == pytest.approx(0.15)
    assert book.basis_for("Taichung-cam01") is None


def test_an_override_replaces_only_the_classes_it_names(tmp_path):
    """Kaohsiung-cam04's person calibration is under investigation; its bag is not."""
    book = load_thresholds(
        _book(
            tmp_path,
            {
                "Kaohsiung-cam04": {
                    "basis": "runs/thr_sweep01: 3,353 -> 13,595 detections at 0.15",
                    "person": {"birth": 0.35, "keep": 0.20},
                }
            },
        )
    )
    cam = book.for_camera("Kaohsiung-cam04")
    assert cam["person"].birth == pytest.approx(0.35)  # held back
    assert cam["bag"].birth == pytest.approx(0.35)  # inherited, not copied
    assert book.for_camera("Taichung-cam01")["person"].birth == pytest.approx(0.15)


def test_the_default_is_not_mutated_by_reading_a_camera(tmp_path):
    book = load_thresholds(
        _book(tmp_path, {"cam": {"basis": "measured", "person": {"birth": 0.9, "keep": 0.5}}})
    )
    book.for_camera("cam")
    assert book.default["person"].birth == pytest.approx(0.15)


def test_an_override_without_a_stated_measurement_is_refused(tmp_path):
    p = _book(tmp_path, {"cam": {"person": {"birth": 0.5, "keep": 0.3}}})
    with pytest.raises(ValueError, match="without a `basis`"):
        load_thresholds(p)


def test_a_blank_basis_is_not_a_basis(tmp_path):
    p = _book(tmp_path, {"cam": {"basis": "   ", "person": {"birth": 0.5, "keep": 0.3}}})
    with pytest.raises(ValueError, match="without a `basis`"):
        load_thresholds(p)


def test_a_basis_that_overrides_nothing_is_refused(tmp_path):
    p = _book(tmp_path, {"cam": {"basis": "measured somewhere"}})
    with pytest.raises(ValueError, match="overrides nothing"):
        load_thresholds(p)


def test_a_book_of_overrides_alone_is_not_a_book(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"schema": "hydranet-thresholds/v1", "cameras": {}}))
    with pytest.raises(ValueError, match="no default thresholds"):
        load_thresholds(p)


def test_an_unknown_schema_is_refused(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"schema": "something-else/v9", "default": {}}))
    with pytest.raises(ValueError, match="unexpected thresholds schema"):
        load_thresholds(p)


def test_a_class_missing_half_its_hysteresis_is_refused(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(
        json.dumps({"schema": "hydranet-thresholds/v1", "default": {"person": {"birth": 0.35}}})
    )
    with pytest.raises(ValueError, match="needs both birth and keep"):
        load_thresholds(p)


def test_the_shipped_book_loads_and_holds_cam04_at_the_shipped_working_point():
    """The file that actually ships, read as a test rather than trusted."""
    book = load_thresholds("configs/serving/thresholds_retail_security.json")
    assert book.for_camera("Kaohsiung-cam04")["person"].birth == pytest.approx(0.35)
    basis = book.basis_for("Kaohsiung-cam04")
    assert basis is not None and "13,595" in basis


def test_a_camera_state_takes_one_camera_s_thresholds():
    book = load_thresholds("configs/serving/thresholds_retail_security.json")
    s = make_state(thresholds=book.for_camera("Kaohsiung-cam04"))
    assert s.thresholds["person"].birth == pytest.approx(0.35)


# -- the appearance gate reaches the injected tracker (2026-09-10) ----------------------


def test_appearance_rows_are_filtered_by_the_same_keep_mask_as_the_boxes():
    """Row i must describe box i after the keep threshold drops some boxes."""
    import numpy as np

    class Looking(StubTracker):
        def update(self, boxes, scores, frame_idx, appearance=None):
            super().update(boxes, scores, frame_idx)
            self.looks = appearance

    s = make_state(tracker_factory=Looking)
    terrain = np.zeros(HW, dtype=np.uint8)
    boxes = np.array([[0, 0, 10, 20], [20, 0, 30, 20], [40, 0, 50, 20]], float)
    scores = np.array([0.9, 0.05, 0.5])  # the middle box is under every keep threshold
    labels = np.zeros(3, dtype=np.int64)
    looks = np.arange(9, dtype=float).reshape(3, 3)
    s.update(0, terrain, boxes, scores, labels, appearance=looks)
    kept = s.tracker.calls[-1][0]
    assert len(kept) == 2 and np.allclose(s.tracker.looks, looks[[0, 2]])


def test_a_threshold_lands_on_a_tracker_that_has_a_gate_and_is_refused_on_one_that_does_not():
    import pytest

    from syncai_hydranet.analytics.bytetrack import OfflineForward

    s = make_state(
        tracker_factory=lambda: OfflineForward(0.35, 0.2, 0.3, 0.4, 5, 1, 5.0),
        appearance_thr=0.39,
    )
    assert s.tracker.appearance_thr == 0.39 and s.appearance_thr == 0.39
    with pytest.raises(ValueError, match="no appearance gate"):
        make_state(tracker_factory=StubTracker, appearance_thr=0.39)


def test_the_dense_vouch_is_computed_on_the_frame_map_and_filtered_like_the_boxes():
    import numpy as np

    from syncai_hydranet.analytics.bytetrack import OfflineForward

    class Vouched(OfflineForward):
        def update(self, boxes, scores, frame_idx, appearance=None, confirmed=None):
            self.last_confirmed = confirmed
            super().update(boxes, scores, frame_idx, appearance, confirmed)

    s = make_state(
        tracker_factory=lambda: Vouched(0.35, 0.2, 0.3, 0.4, 5, 1, 5.0, dense_birth_thr=0.15),
        dense_person_id=5,
    )
    terrain = np.zeros(HW, dtype=np.uint8)
    terrain[0:20, 0:10] = 5  # person pixels under the first box only
    boxes = np.array([[0, 0, 10, 20], [20, 0, 30, 20], [40, 0, 50, 20]], float)
    scores = np.array([0.25, 0.05, 0.25])  # the middle box is under every keep threshold
    labels = np.zeros(3, dtype=np.int64)
    s.update(0, terrain, boxes, scores, labels)
    assert list(s.tracker.last_confirmed) == [True, False]
    s.update(1, terrain, boxes, scores, labels)  # persistence: the second frame births
    assert s.tracker.dense_births == 1 and s.tracker.tracks[0].born_confirmed


def test_a_person_id_on_a_tracker_without_the_birth_is_refused():
    import pytest

    with pytest.raises(ValueError, match="dense-confirmed"):
        make_state(tracker_factory=StubTracker, dense_person_id=5)
