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
    "no appearance model BY DEFAULT: two shoppers who swap places while overlapping will "
    "swap ids. `appearance_thr` gates re-association after a gap and is off unless a "
    "caller sets it",
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


def appearance_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Chi-squared distance between two normalised appearance histograms.

    Chi-squared rather than L2 because these are histograms: it weights a difference by
    how much mass is there, so two crops that disagree about a rare colour are not scored
    like two that disagree about the dominant one.

    The tracker does not say what the descriptor is -- only that it is a non-negative
    vector that sums to about one and that the caller builds the same way every frame.
    The measurement this was written for used a coarse HSV histogram of the upper 45% of
    the box, where the clothing is.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"appearance vectors differ in length: {a.shape} vs {b.shape}")
    # A NaN row -- `appearance.torso_histograms` for a box too small to describe -- gives
    # NaN here, and `NaN > thr` is False at the gate: an undescribed box is never split
    # from its track, only carried. Absence of evidence is not a different shopper.
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)))


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
    # or empty, when the caller runs detection without the pose head.
    #
    # It is a field rather than a parallel structure because the alignment with `frames`
    # and `boxes` is the whole contract: `events.pose_posture_events` reads keypoints[i]
    # against frames[i], and two containers that have to stay index-aligned across a
    # module boundary drift the first time one of them is filtered.
    #
    # **Filled by the P3 pose head via `update(..., keypoints=...)`** -- see the wire
    # note on `update` below; `tools/pose/pose_overlay.py` is a live caller. Empty is
    # still a legal state and every consumer has to say what it does with it;
    # `require_keypoints` in `analytics/events/pose.py` is that refusal, in one place.
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
    # The caller's appearance descriptor for each observed box, index-aligned with
    # `frames` and `boxes` for the same reason the three fields above are.
    #
    # Written whenever the caller supplies it, and *used* only when `appearance_thr` is
    # set. Recorded-but-unused is the same choice `scores` made: a consumer that wants to
    # ask "did this track change person" needs the series, and nothing downstream could
    # ask before the field existed.
    appearance: list[np.ndarray] = field(default_factory=list)
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
        staff_memory_gap: int = 0,
        staff_memory_iou: float = 0.3,
        birth_thr: float | None = None,
        appearance_thr: float | None = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        # **A survival band, and deliberately nothing else.** With `birth_thr` set, a box
        # below it may continue a track it matches and may not start one; the caller
        # decodes at the lower edge so those boxes reach here at all. `None` keeps the
        # single-threshold behaviour every number this project has published was measured
        # under.
        #
        # This exists to separate two things `bytetrack` moves together. That file brings
        # the high/low band *and* a Kalman filter, and this tracker refuses the filter on
        # stated grounds -- no measured noise model exists for this footage, and tuned-
        # looking constants that were guessed are worse than an honest constant velocity
        # step. 7.11 measured the pair against the single-threshold arm (202 tracks -> 100,
        # coasted fraction 0.0643 -> 0.0159) and could not say which half carried it. If
        # the band alone carries the gain, the filter never has to be argued about.
        #
        # Association itself is untouched: one round, greedy IoU, all boxes eligible.
        # ByteTrack associates the high band first and the low band second against what is
        # left, which is a *matching* change on top of the band -- a third variable, and
        # the point here is to move one.
        if birth_thr is not None and not 0.0 <= birth_thr <= 1.0:
            raise ValueError(f"birth_thr must be a score in [0, 1], got {birth_thr!r}")
        self.birth_thr = birth_thr
        # **An appearance gate on RE-association only, and deliberately nothing else.**
        # When a track has coasted through a gap, a match whose appearance distance from
        # the track's last observed descriptor exceeds this is refused: the track is left
        # unmatched and the detection is free to start its own. Matches on consecutive
        # frames are never gated -- the tracker still has no appearance model for the
        # ordinary case, which is what `SIMPLIFICATIONS` says.
        #
        # Measured on Taichung-cam04's 14:31 clip, 2026-09-08. Appearance distance between
        # consecutive observations of one track: p50 0.021, p99 0.225. Between two
        # different shoppers in the same frame: p10 0.376. The two do not overlap, which
        # is what makes a threshold between them meaningful rather than tuned. 29 of
        # 13,066 steps sit in the between-people range and 48% of those are the step right
        # after a gap, against 1% of steps overall; 19 are corroborated by an impossible
        # floor speed or a doubled box as well. That is 17 of 108 tracks carrying at least
        # one identity change -- a track whose attribute vote is then two people's.
        #
        # `None` keeps the behaviour every number this project has published was measured
        # under. Turning it on SPLITS tracks, so track counts rise; that is two shoppers
        # being counted as two rather than as one, not a regression.
        if appearance_thr is not None and appearance_thr < 0:
            raise ValueError(f"appearance_thr must be a distance >= 0, got {appearance_thr!r}")
        self.appearance_thr = appearance_thr
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
        # A new track born where a confirmed one just died inherits its accumulated
        # staff evidence, when `staff_memory_gap` > 0 frames. Fragmentation is the
        # measured cause of colour flicker (Kaohsiung-cam04, 900 frames: 89 tracks at
        # median life 12 frames produced 66 warm-up colour pops and 33 tracks that
        # never reached a verdict, against 5 genuine flips), and each fragment
        # restarting its evidence from zero is what turns one shopper into a chain of
        # colour changes. Inheritance requires the dead track be CONFIRMED, its last
        # box overlap the newborn's at `staff_memory_iou`, and each dead track feeds
        # at most one heir -- the same evidence must not colour two people.
        self.staff_memory_gap = int(staff_memory_gap)
        self.staff_memory_iou = float(staff_memory_iou)
        self.tracks: list[Track] = []
        # Retired tracks, kept because dwell is computed after the clip ends and a
        # shopper who left the frame is exactly the one whose visit is complete.
        self.retired: list[Track] = []
        self._inherited_from: set[int] = set()  # retired track ids already consumed
        self._next_id = 1
        # None until the first update decides. See `update` -- keypoints and scores are
        # all frames or none, because a gap in either cannot be recovered afterwards.
        self._keypoints_seen: bool | None = None
        self._scores_seen: bool | None = None
        self._staff_seen: bool | None = None
        self._appearance_seen: bool | None = None

    def update(
        self,
        boxes: np.ndarray,
        frame_idx: int,
        keypoints: np.ndarray | None = None,
        scores: np.ndarray | None = None,
        staff_scores: np.ndarray | None = None,
        appearance: np.ndarray | None = None,
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

        ``appearance`` is (N, D) descriptors, one row per box in the same order. It is
        recorded like the three above, and additionally *used* when `appearance_thr` is
        set: a track re-associating after a gap must look like itself. See the constructor
        for the measurement that fixed the threshold's meaning.

        **All frames or none, per tracker**, for keypoints, scores, staff scores and
        appearance independently.
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
        if appearance is not None:
            appearance = (
                np.asarray(appearance, dtype=float).reshape(len(boxes), -1)
                if len(boxes)
                else np.zeros((0, 0))
            )
            if len(appearance) != len(boxes):
                raise ValueError(
                    f"{len(appearance)} appearance vectors for {len(boxes)} boxes: they "
                    "are matched by position, so a mismatch has no safe interpretation"
                )
        if self.appearance_thr is not None and appearance is None:
            raise ValueError(
                "appearance_thr is set and this frame carried no appearance vectors. The "
                "gate can only refuse a re-association it can measure, so a frame without "
                "them would silently let through exactly the matches it exists to check."
            )
        self._keypoints_seen = _latch(self._keypoints_seen, keypoints is not None, "keypoints")
        self._scores_seen = _latch(self._scores_seen, scores is not None, "scores")
        self._staff_seen = _latch(self._staff_seen, staff_scores is not None, "staff scores")
        self._appearance_seen = _latch(
            self._appearance_seen, appearance is not None, "appearance"
        )

        # Predict: constant velocity on the box centre, size held.
        for t in self.tracks:
            t.box = t.box + np.concatenate([t.velocity, t.velocity])
            t.age += 1

        matched = self._match(boxes)
        # The gate. `age` was incremented by the predict step above, so a track observed
        # on the immediately preceding frame arrives here at age 1 and one that coasted
        # through a gap at 2 or more. Only the second kind is checked: consecutive-frame
        # association is where IoU is trustworthy and where gating would cost tracks for
        # a person who merely turned round.
        # `appearance is not None` is already guaranteed by the refusal above; it is
        # restated so the type checker can see it, rather than asserted away.
        if self.appearance_thr is not None and appearance is not None:
            for ti in [ti for ti in matched if self.tracks[ti].age > 1]:
                t = self.tracks[ti]
                if not t.appearance:
                    continue
                if appearance_distance(t.appearance[-1], appearance[matched[ti]]) > (
                    self.appearance_thr
                ):
                    # Refuse the pair rather than reassign it: the track is left to coast
                    # or die, and the detection falls through to `unmatched` below, where
                    # it starts its own track. Two shoppers then read as two.
                    del matched[ti]
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
            if appearance is not None:
                t.appearance.append(np.asarray(appearance[di], dtype=float).copy())
            if t.hits >= self.min_hits:
                t.confirmed = True

        unmatched = set(range(len(boxes))) - set(matched.values())
        if self.birth_thr is not None:
            if scores is None:
                raise ValueError(
                    "birth_thr is set and this frame carried no scores. The band decides "
                    "which boxes may start a track, so a scoreless frame has no safe "
                    "reading -- `clip_tracks.track_clip` supplies them on every call."
                )
            unmatched = {di for di in unmatched if float(scores[di]) >= self.birth_thr}
        for di in unmatched:
            inherited: list[float] = []
            if self.staff_memory_gap > 0 and staff_scores is not None:
                donor, best = None, self.staff_memory_iou
                for rt in self.retired:
                    if rt.track_id in self._inherited_from or not rt.staff_scores:
                        continue
                    if frame_idx - rt.frames[-1] > self.staff_memory_gap:
                        continue
                    overlap = float(iou(boxes[di][None, :], rt.boxes[-1][None, :])[0, 0])
                    if overlap >= best:
                        donor, best = rt, overlap
                if donor is not None:
                    inherited = list(donor.staff_scores)
                    self._inherited_from.add(donor.track_id)
            t = Track(
                self._next_id,
                boxes[di].copy(),
                frames=[frame_idx],
                boxes=[boxes[di].copy()],
                keypoints=[] if keypoints is None else [keypoints[di].copy()],
                scores=[] if scores is None else [float(scores[di])],
                appearance=[]
                if appearance is None
                else [np.asarray(appearance[di], dtype=float).copy()],
                staff_scores=inherited
                + ([] if staff_scores is None else [float(staff_scores[di])]),
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
