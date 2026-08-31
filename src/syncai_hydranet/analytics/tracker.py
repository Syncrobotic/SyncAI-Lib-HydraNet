"""Tracking by detection, for a camera that does not move.

SORT's shape without SORT's Kalman filter, and the omission is deliberate rather than
lazy. A Kalman filter is optimal fusion *given a measured noise model*; this project has
no hand-labelled site boxes, so both the process and measurement covariances would be
invented. Tuned-looking constants that were guessed are worse than an honest constant
velocity step, because they make the result look calibrated. Constant velocity is what
is actually known about a shopper between two frames 200 ms apart.

The association is greedy on IoU rather than Hungarian, because scipy is not a
dependency here and, on a fixed camera with a handful of people in frame, the two agree
almost always. They diverge when two tracks compete for one detection -- a crowded
aisle, or two shoppers crossing -- and there greedy takes the higher-IoU pair and leaves
the other track to coast. `SIMPLIFICATIONS` records this at module level so it reaches a
reader of the numbers rather than only a reader of the code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

SIMPLIFICATIONS = (
    "constant velocity, and for a stationary person that velocity is box jitter -- "
    "mitigated by matching against the last observed box too, not removed",
    "greedy IoU association by default; optimal assignment is available and measured at "
    "no difference on the clip that motivated looking",
    "constant velocity, no Kalman: no measured noise model exists to fit one to",
    "no appearance model: two shoppers who swap places while overlapping will swap ids",
)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` (N,4) against every box in ``b`` (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def iou_pair(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """IoU of one xyxy box against one other.

    **Deliberately not a wrapper around `iou` above, and the cost is measured rather than
    assumed**: routing a single pair through the vectorised form takes 17.8 us against
    1.2 us here, because it allocates two input arrays and a (1,1) result per call. The
    callers that hold one pair are loops -- 100k comparisons is 1.65 s of that difference
    -- and a loop is exactly the shape that made three of them write this out by hand
    instead of importing anything.

    Two implementations of one formula is a thing this repository keeps finding, so
    `tests/test_bytetrack.py` holds these two equal over random and degenerate boxes
    rather than trusting that they read alike. A caller comparing many boxes against many
    should use `iou` and not a loop over this.
    """
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x0 >= x1 or y0 >= y1:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _latch(seen: bool | None, now: bool, what: str) -> bool:
    """Hold a tracker to its first answer about whether it is given ``what``.

    One helper for keypoints and scores because the failure is identical and it is not a
    failure that announces itself: a list that skipped three frames still zips against
    `frames` without error, it just describes different frames than it claims to.
    """
    if seen is None:
        return now
    if seen != now:
        raise ValueError(
            f"this tracker was updated {'with' if seen else 'without'} {what} and is now "
            f"being updated {'with' if now else 'without'} them. `Track.{what}` has to stay "
            "index-aligned with `Track.frames`; a gap cannot be recovered later, so it is "
            "refused here rather than found by a consumer."
        )
    return now


@dataclass
class Track:
    """One identity. ``confirmed`` gates whether it is allowed to count."""

    track_id: int
    box: np.ndarray  # xyxy, the most recent observation or prediction
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hits: int = 1
    age: int = 0  # frames since the last observation
    frames: list[int] = field(default_factory=list)  # frame indices actually observed
    boxes: list[np.ndarray] = field(default_factory=list)  # observed boxes, one per frame
    confirmed: bool = False
    # (17, 3) COCO keypoints per observed frame -- x, y in image pixels and a score --
    # or empty, which is the state of every track this tracker produces today.
    #
    # It is a field rather than a parallel structure because the alignment with `frames`
    # and `boxes` is the whole contract: `events.pose_posture_events` reads keypoints[i]
    # against frames[i], and two containers that have to stay index-aligned across a
    # module boundary drift the first time one of them is filtered.
    #
    # **Filled by the second stage's pose model, which does not exist yet.** Empty is
    # therefore the normal case and every consumer has to say what it does with it;
    # `require_keypoints` in events.py is that refusal, in one place, naming the missing
    # model rather than returning no events as if none had happened.
    keypoints: list[np.ndarray] = field(default_factory=list)
    # The detector score of each observed box, index-aligned with `frames` and `boxes`
    # for the same reason `keypoints` is a field rather than a parallel structure.
    #
    # **This is the quantity the event layer had no way to see.** The 2026-08-26 threshold
    # sweep lowered the person birth threshold 0.35 -> 0.15 and measured +51% detections
    # and +141% tracks on seven healthy cameras, at four times the posture events -- and
    # the extra events were not false boxes. They were real people detected at low
    # confidence, whose keypoints are noisier and so produce more posture runs. Filtering
    # the boxes harder does not reach that: the dense-head confirmation dropped 12-14% of
    # boxes and produced *exactly* the unfiltered arm's events. What reaches it is a
    # consumer that can tell a track built from 0.15 boxes from one built from 0.6 boxes,
    # and until this field existed nothing downstream could.
    #
    # Empty when the caller did not supply scores, which is legal and is the state of
    # every track produced before 2026-08-26. A consumer that needs it says so; see
    # `events.support_for`.
    scores: list[float] = field(default_factory=list)
    # P(staff) for the crop of each observed box, index-aligned with `frames` and `boxes`
    # for the third time and the same reason.
    #
    # **Evidence, not a verdict.** `staff/customer` is a property of a person and not of a
    # frame -- `analytics/track_attributes.py` measured the same staff member labelled `F`
    # in one frame and `M` in the next -- so what a consumer wants is one answer per
    # track, and `analytics.staff.track_staff` is where that reduction lives. It is not a
    # property here on purpose: this module stays free of the classifier, its crop
    # geometry and its minimum-observation rule, all three of which are `staff.py`'s to
    # state and to change.
    #
    # Empty when the caller supplied no staff scores, which is every track produced
    # before 2026-08-28 and every track from a camera the classifier is not licensed for.
    staff_scores: list[float] = field(default_factory=list)

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def foot(self) -> np.ndarray:
        """Bottom-centre: where the person meets the floor.

        The one range a single camera can recover, and the same assumption the BEV
        renderer already makes for detections. It is wrong when the feet are occluded by
        a fixture, which is common in a shop -- so a track whose box bottom sits on a
        counter edge reports the counter's distance, not the shopper's.
        """
        return np.array([(self.box[0] + self.box[2]) / 2, self.box[3]])


class Tracker:
    """Greedy IoU tracker with a confirmation delay.

    ``min_hits`` is the parameter that matters for analytics and it defaults high on
    purpose. A track confirmed on its first detection turns every one-frame false
    positive into a shopper; requiring three consecutive observations costs a genuine
    shopper roughly half a second of dwell at the head of their track and removes the
    single largest source of over-counting. Under-count over over-count, always: the
    first is visible against a manual audit, the second reads as a good day's trading.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        min_hits: int = 3,
        assignment: str = "greedy",
        match_against: str = "both",
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        if assignment not in ("hungarian", "greedy"):
            raise ValueError(f"assignment must be hungarian or greedy, got {assignment!r}")
        # Default unchanged: optimal assignment was measured at 76 -> 78 tracks on the
        # clip that motivated it, i.e. nothing. Available, not imposed.
        self.assignment = assignment
        if match_against not in ("both", "predicted"):
            raise ValueError(f"match_against must be both or predicted, got {match_against!r}")
        # `predicted` reproduces every track number this project published before
        # 2026-08-18. `both` is the default because it more than halved fragmentation on
        # the one case where the right answer was independently known.
        self.match_against = match_against
        self.tracks: list[Track] = []
        # Retired tracks, kept because dwell is computed after the clip ends and a
        # shopper who left the frame is exactly the one whose visit is complete.
        self.retired: list[Track] = []
        self._next_id = 1
        # None until the first update decides. See `update` -- keypoints and scores are
        # all frames or none, because a gap in either cannot be recovered afterwards.
        self._keypoints_seen: bool | None = None
        self._scores_seen: bool | None = None
        self._staff_seen: bool | None = None

    def update(
        self,
        boxes: np.ndarray,
        frame_idx: int,
        keypoints: np.ndarray | None = None,
        scores: np.ndarray | None = None,
        staff_scores: np.ndarray | None = None,
    ) -> list[Track]:
        """Advance one frame. ``boxes`` is (N,4) xyxy for one class. Returns live tracks.

        ``keypoints`` is (N,17,3) in image pixels, one row per box in the same order --
        x, y, score. Supplying it is what makes `events.pose_posture_events` and
        `reach_to_shelf_events` runnable: they read ``track.keypoints[i]`` against
        ``track.frames[i]`` and `require_keypoints` refuses the pair when the lengths
        disagree. Until the pose head existed nothing could fill it and the refusal was
        the whole story; now the producer exists and this is the wire.

        ``scores`` is (N,) detector confidences, one per box in the same order. It is
        recorded rather than used: nothing here associates on score -- that is
        `bytetrack`'s two-band mechanism and this tracker deliberately has no equivalent
        -- and the reason to carry it is stated on `Track.scores`.

        ``staff_scores`` is (N,) values of P(staff) from `analytics.staff`, one per box in
        the same order. Recorded rather than used, like ``scores``: nothing here
        associates on appearance, and the reduction to one verdict per person is
        `staff.track_staff`.

        **All frames or none, per tracker**, for keypoints, scores and staff scores
        independently.
        Passing either on some calls and not others silently misaligns its list against
        `frames`, which is exactly the drift `Track.keypoints` documents as its reason for
        being a field. A tracker that has seen keypoints refuses a later call without
        them, and the reverse, rather than producing a track whose pose belongs to
        different frames than it claims.
        """
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        if keypoints is not None:
            keypoints = np.asarray(keypoints, dtype=float).reshape(-1, 17, 3)
            if len(keypoints) != len(boxes):
                raise ValueError(
                    f"{len(keypoints)} keypoint sets for {len(boxes)} boxes: they are "
                    "matched by position, so a mismatch has no safe interpretation"
                )
        if scores is not None:
            scores = np.asarray(scores, dtype=float).reshape(-1)
            if len(scores) != len(boxes):
                raise ValueError(
                    f"{len(scores)} scores for {len(boxes)} boxes: they are matched by "
                    "position, so a mismatch has no safe interpretation"
                )
        if staff_scores is not None:
            staff_scores = np.asarray(staff_scores, dtype=float).reshape(-1)
            if len(staff_scores) != len(boxes):
                raise ValueError(
                    f"{len(staff_scores)} staff scores for {len(boxes)} boxes: they are "
                    "matched by position, so a mismatch has no safe interpretation"
                )
        self._keypoints_seen = _latch(self._keypoints_seen, keypoints is not None, "keypoints")
        self._scores_seen = _latch(self._scores_seen, scores is not None, "scores")
        self._staff_seen = _latch(self._staff_seen, staff_scores is not None, "staff scores")

        # Predict: constant velocity on the box centre, size held.
        for t in self.tracks:
            t.box = t.box + np.concatenate([t.velocity, t.velocity])
            t.age += 1

        matched = self._match(boxes)
        for ti, di in matched.items():
            t = self.tracks[ti]
            new = boxes[di]
            old_centre = t.centre
            t.box = new
            t.velocity = t.centre - old_centre
            t.hits += 1
            t.age = 0
            t.frames.append(frame_idx)
            t.boxes.append(new.copy())
            if keypoints is not None:
                t.keypoints.append(keypoints[di].copy())
            if scores is not None:
                t.scores.append(float(scores[di]))
            if staff_scores is not None:
                t.staff_scores.append(float(staff_scores[di]))
            if t.hits >= self.min_hits:
                t.confirmed = True

        for di in set(range(len(boxes))) - set(matched.values()):
            t = Track(
                self._next_id,
                boxes[di].copy(),
                frames=[frame_idx],
                boxes=[boxes[di].copy()],
                keypoints=[] if keypoints is None else [keypoints[di].copy()],
                scores=[] if scores is None else [float(scores[di])],
                staff_scores=[] if staff_scores is None else [float(staff_scores[di])],
            )
            t.confirmed = self.min_hits <= 1
            self.tracks.append(t)
            self._next_id += 1

        live, gone = [], []
        for t in self.tracks:
            (live if t.age <= self.max_age else gone).append(t)
        self.tracks = live
        self.retired.extend(t for t in gone if t.confirmed)
        return [t for t in self.tracks if t.confirmed and t.age == 0]

    def _match(self, boxes: np.ndarray) -> dict[int, int]:
        """Assign detections to tracks. Returns {track index: detection index}.

        Two things were changed here and only one of them mattered. Both are recorded
        because the one that did not matter was the confident hypothesis.

        **What was wrong.** On Taichung-cam01 a shopper standing at a counter for 110
        seconds came out as three tracks, breaking at box IoU **0.797** and **0.686**
        against a 0.3 threshold -- no detection gap, no threshold too tight, no `max_age`
        too small, and 48 of that clip's 76 tracks ended with a detection available.

        **The hypothesis, and its refutation.** `SIMPLIFICATIONS` blamed greedy assignment
        for exactly this shape, and `reid_metrics._hungarian` was already in the tree. It
        was wired in and measured: 76 tracks became **78**, median length 26.5 to 26.0.
        Optimal assignment changes nothing here, so the option stays and the default does
        not, because a default changed without evidence is how a codebase stops being
        reproducible.

        **What actually broke it: matching against the predicted box alone.** `update`
        applies constant velocity before matching, and for a person standing still that
        "velocity" is box jitter, which displaces the prediction far enough to lose an
        otherwise obvious match. Matching against the predicted box **or** the last
        observed one, whichever agrees better, keeps prediction where it earns its place
        -- a shopper walking at 5 fps moves a long way between frames -- and removes its
        cost where it does not:

            predicted only          76 tracks, median 26.5, p90 134, max  236 (47 s)
            predicted or observed   34 tracks, median 34.5, p90 262, max  551 (110 s)

        551 frames is the 110-second shopper, one track. **One verified case is not a
        general result**: fewer tracks can also mean wrong merges, and nothing here can
        tell the difference until a site clip carries ground-truth ids. What is verified
        is that this particular merge is correct, because those three tracks sat within
        9 cm of each other on the floor.
        """
        if not self.tracks or len(boxes) == 0:
            return {}
        m = iou(np.stack([t.box for t in self.tracks]), boxes)
        if self.match_against == "both":
            observed = np.stack([t.boxes[-1] if t.boxes else t.box for t in self.tracks])
            m = np.maximum(m, iou(observed, boxes))
        if self.assignment == "hungarian":
            from .reid_metrics import _hungarian

            pairs = {}
            for ti, di in _hungarian(-m):
                if m[ti, di] >= self.iou_threshold:
                    pairs[int(ti)] = int(di)
            return pairs
        pairs: dict[int, int] = {}
        used_det: set[int] = set()
        order = np.dstack(np.unravel_index(np.argsort(m, axis=None)[::-1], m.shape))[0]
        for ti, di in order:
            ti, di = int(ti), int(di)
            if m[ti, di] < self.iou_threshold:
                break
            if ti in pairs or di in used_det:
                continue
            pairs[ti] = di
            used_det.add(di)
        return pairs

    def finished(self) -> list[Track]:
        """Every confirmed track, retired and still live.

        Call after the last frame. A track still live at the end of a clip is a shopper
        who had not left when the recording stopped, so its dwell is a lower bound --
        `dwell_table` flags those rather than averaging them in as if they were complete.
        """
        return self.retired + [t for t in self.tracks if t.confirmed]
