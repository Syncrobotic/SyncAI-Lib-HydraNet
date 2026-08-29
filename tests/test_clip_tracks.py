"""The clip loop that four scripts each kept a copy of, and the one thing they disagreed on.

These tests exist because the divergence was known and recorded -- `scripts/ty_ratchet.sh`
named it -- and then left alone for a day, which is long enough for a measurement to be
published off the wrong copy. What is pinned here is not that undistortion is correct; it
is that **the choice cannot be made silently any more**.
"""

from __future__ import annotations

import numpy as np
import pytest

from syncai_hydranet.analytics.clip_tracks import (
    PERSON,
    ClipTracks,
    to_source_pixels,
    track_clip,
    undistort_boxes,
)
from syncai_hydranet.analytics.tracker import Tracker


class _Tracker:
    """Records what it was fed, which is the only thing these tests are about."""

    def __init__(self):
        self.seen: list[np.ndarray] = []
        self.scores: list[np.ndarray] = []

    def update(self, boxes, _frame_index, scores=None):
        self.seen.append(np.asarray(boxes, dtype=float).copy())
        self.scores.append(np.asarray([] if scores is None else scores, dtype=float))

    def finished(self):
        return [f"track-{len(self.seen)}"]


class _Det:
    """The slice of a torch tensor's API that `track_clip` actually touches."""

    def __init__(self, arr):
        self._a = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def __len__(self):
        return len(self._a)


class _Model:
    def __init__(self, boxes_per_frame, scores_per_frame=None):
        self._boxes = boxes_per_frame
        self._scores = scores_per_frame

    def predict(self, x, score_thr):  # noqa: ARG002  (stub returns fixed boxes)
        i = int(x)
        boxes = self._boxes[i]
        scores = self._scores[i] if self._scores is not None else np.full(len(boxes), 0.9)
        return {
            "detection": [
                {
                    "labels": _Det(np.full(len(boxes), PERSON)),
                    "boxes": _Det(np.asarray(boxes, dtype=float).reshape(-1, 4)),
                    "scores": _Det(np.asarray(scores, dtype=float).reshape(-1)),
                }
            ]
        }


def _harness(boxes_per_frame, src=(1920, 1080)):
    """The three callables `track_clip` takes instead of importing the video layer.

    They are arguments rather than imports because `analytics` may not depend on `cli`,
    and the useful side effect is that this test needs no video, no torch and no model.
    A 1x1 pixel carrying the frame index stands in for the decoded frame.
    """

    def frames(_clip, _w, _h, _fps):
        for i in range(len(boxes_per_frame)):
            yield np.full((1, 1, 3), i, dtype=np.uint8)

    def preprocess(image, _size):
        return _Canvas(int(np.asarray(image).flat[0])), None, (0, 0, 640, 512)

    def probe(_clip):
        return (*src, 30.0)

    return frames, preprocess, probe


class _Canvas:
    """Stands in for the preprocessed tensor; carries only the frame index."""

    def __init__(self, i):
        self.i = i

    def to(self, _device):
        return self.i

    def __int__(self):
        return self.i


def _run(boxes_per_frame, *, k1, max_frames=0, scores_per_frame=None, tracker=None):
    frames, preprocess, probe = _harness(boxes_per_frame)
    tracker = _Tracker() if tracker is None else tracker
    out = track_clip(
        "clip.mp4",
        _Model(boxes_per_frame, scores_per_frame),
        (512, 640),
        "cpu",
        tracker,
        frames=frames,
        preprocess=preprocess,
        probe=probe,
        fps=5.0,
        score_thr=0.2,
        k1=k1,
        max_frames=max_frames,
    )
    return out, tracker


BOXES = [[[100.0, 50.0, 200.0, 300.0]], [[110.0, 55.0, 210.0, 305.0]]]


def test_k1_is_keyword_only_and_has_no_default():
    """The whole point: a caller cannot forget to state its lens correction.

    The two copies this module replaced differed by exactly this, and neither said so.
    A default would put that back.
    """
    frames, preprocess, probe = _harness(BOXES)
    with pytest.raises(TypeError):
        track_clip(  # ty: ignore[missing-argument]
            "clip.mp4",
            _Model(BOXES),
            (512, 640),
            "cpu",
            _Tracker(),
            frames=frames,
            preprocess=preprocess,
            probe=probe,
            fps=5.0,
            score_thr=0.2,
        )


def test_none_and_zero_both_mean_no_lens_correction():
    """`0.0` is the identity of the division model, so it must equal `None` exactly.

    Stated as a test because a caller reproducing an older run passes `0.0` and needs to
    know it reproduces it, rather than inferring that from the algebra.
    """
    _, none_tracker = _run(BOXES, k1=None)
    _, zero_tracker = _run(BOXES, k1=0.0)
    for a, b in zip(none_tracker.seen, zero_tracker.seen, strict=True):
        np.testing.assert_allclose(a, b)


def test_a_real_k1_moves_the_boxes_the_tracker_sees():
    """The divergence, made visible: same clip, same detections, different associations.

    The tracker matches on IoU, so boxes that differ are boxes that can link differently.
    This is the failure the two copies could produce on the same footage.
    """
    _, plain = _run(BOXES, k1=None)
    _, lensed = _run(BOXES, k1=-0.225)
    assert not np.allclose(plain.seen[0], lensed.seen[0])


def test_max_frames_stops_early_and_counts_what_it_saw():
    out, tracker = _run(BOXES, k1=None, max_frames=1)
    assert out.frames == 1
    assert out.detections == 1
    assert len(tracker.seen) == 1


def test_the_return_carries_what_all_four_call_sites_needed():
    """One tuple, because the copies returned three different ones in three orders."""
    out, _ = _run(BOXES, k1=None)
    assert isinstance(out, ClipTracks)
    assert (out.frames, out.detections, out.src_w, out.src_h) == (2, 2, 1920, 1080)


def test_a_frame_with_no_detections_still_advances_the_tracker():
    """A gap is information: `max_age` counts frames, so a skipped one ages nothing."""
    out, tracker = _run([[], [[100.0, 50.0, 200.0, 300.0]]], k1=None)
    assert out.frames == 2
    assert len(tracker.seen) == 2
    assert tracker.seen[0].shape == (0, 4)


def test_empty_input_is_shaped_not_just_empty():
    """`(0, 4)` rather than `(0,)`: the tracker indexes columns even on an empty batch."""
    assert to_source_pixels(np.zeros((0, 4)), (0, 0, 640, 512), 1920, 1080).shape == (0, 4)
    assert undistort_boxes(np.zeros((0, 4)), -0.225, 1920, 1080).shape == (0, 4)


def test_undistortion_keeps_a_box_a_box():
    """The axis-aligned hull is an approximation, but it must stay ordered.

    x0 < x1 and y0 < y1 after the radial map, or every downstream IoU is negative area.
    """
    boxes = np.array([[100.0, 50.0, 200.0, 300.0], [1500.0, 800.0, 1700.0, 1000.0]])
    out = undistort_boxes(boxes, -0.225, 1920, 1080)
    assert (out[:, 2] > out[:, 0]).all()
    assert (out[:, 3] > out[:, 1]).all()


def test_every_track_carries_the_score_of_each_box_it_was_built_from():
    """The shared loop is where four callers get confidence for free, or none of them do.

    `events.TrackSupport` records the measurement: at a 0.15 birth threshold the fleet's
    posture events quadrupled and the extra ones were real people detected at low
    confidence. A consumer can only tell those apart if the score survived association,
    and this loop is the only place all four call sites pass through. The real `Tracker`
    runs here rather than the recording stub, because what is being checked is that the
    score reaches a `Track`.
    """
    boxes = [[[100.0, 100.0, 160.0, 300.0]], [[104.0, 100.0, 164.0, 300.0]]]
    out, _ = _run(
        boxes,
        k1=None,
        scores_per_frame=[[0.16], [0.62]],
        tracker=Tracker(min_hits=1),
    )
    (track,) = out.tracks
    assert track.scores == pytest.approx([0.16, 0.62])
    assert len(track.scores) == len(track.frames) == len(track.boxes)


def test_a_frame_with_no_person_still_passes_scores_so_the_latch_holds():
    """An empty frame must pass an empty score array, not nothing.

    `Tracker.update` latches on the first call and refuses a later one that disagrees, so
    a loop that skipped `scores=` whenever a frame was empty would raise on the first
    quiet frame -- which on this footage is most of them.
    """
    out, _ = _run([[[100.0, 100.0, 160.0, 300.0]], []], k1=None, tracker=Tracker(min_hits=1))
    assert out.frames == 2
