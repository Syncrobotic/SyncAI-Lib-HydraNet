"""ByteTrack's forward pass: Kalman prediction plus two-stage hysteresis association.

**This is the repository's second tracker, and it disagrees with the first.** That is
worth stating at the top rather than leaving for a reader to discover, because the
disagreement is about the one thing both files are for.

`tracker.py` refuses a Kalman filter, and says why: "A Kalman filter is optimal fusion
*given a measured noise model*; this project has no hand-labelled site boxes, so both the
process and measurement covariances would be invented. Tuned-looking constants that were
guessed are worse than an honest constant velocity step, because they make the result look
calibrated."

This file runs one anyway, on ByteTrack's published MOT17 weights. It is not an oversight
and it is not a refutation -- it takes the same premise to a different conclusion. The
covariances here are *borrowed and named*, not fitted, and the one adaptation to this
footage is stated in `POS_W`/`VEL_W` below: the velocity prior is rescaled by the frame
rate, because MOT17 is 25-30 fps and these clips sample at 5. Borrowed-and-labelled is a
different thing from guessed, but it is **not** the measured noise model `tracker.py` is
holding out for, and nothing here supplies one.

**The comparison has been run, and it did not settle the question.** `scripts/track_idf1.py`
scores both arms against a labelled clip; on `runs/gt_cam01` (900 frames, 5 identities)
the shipped single-stage arm reads IDF1 0.7388 with 6 switches and the two-stage one
0.7418 with 3 -- three tenths of a point of IDF1, which is inside what one hand-labelled
clip can distinguish, against half the switches, which is not nothing. One clip is one
clip. So the choice is still by what the caller needs:

    tracker.Tracker     greedy IoU, constant velocity, no Kalman. Dwell, ground paths,
                        footfall -- anything whose output is an integral over a clip.
    bytetrack (here)    Kalman + high/low score bands. Offline track mining and the
                        inference-side stabilisers, where coasting through a missed
                        detection is the point and a fabricated box is acceptable
                        because a human reviews the output.

Moved here from `scripts/offline_tracks.py`, where two scripts reached it through a
`sys.path` insert. `tests/test_scripts_are_not_libraries.py` states the cost of that and
it is the whole reason for this file's location: shared code in `scripts/` sits outside
the wheel, outside the type ratchet and outside the coverage floor, so the thing every
caller depends on is the thing nothing checks.

What deliberately did **not** come with it is `stash_crops`. It cuts review thumbnails and
crop-encoder inputs at `track_review.py`'s display geometry, which is a presentation
concern and not a tracking one; it stays in `offline_tracks.py` as a function over a
`Fragment`. `Fragment` keeps the two crop lists as storage the caller may fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .reid_metrics import _hungarian
from .tracker import Track, appearance_distance, iou

# ByteTrack's published noise weights (its kalman_filter.py), tuned on MOT17 at 25-30
# fps. The velocity prior is rescaled at construction by (25 / effective_fps): at the
# 5 fps this samples (content is 7-8 fps real), per-frame displacement is ~5x MOT17's
# for the same walking speed, and an unscaled prior would gate out every walking match.
POS_W = 1.0 / 20.0
VEL_W = 1.0 / 160.0


# Public, unlike the `_cwh`/`_xyxy` they were in `offline_tracks.py`. The stitch pass
# and the motion statistics there both take a box centre, so the conversion was never
# private to the filter -- the underscore only meant "same file".
def to_cwh(box: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dtype=float)


def to_xyxy(s: np.ndarray) -> np.ndarray:
    cx, cy, w, h = s[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)


class Kalman:
    """Constant-velocity Kalman on (cx, cy, w, h) with height-proportional noise."""

    def __init__(self, box: np.ndarray, vel_scale: float) -> None:
        self.vel_scale = vel_scale
        z = to_cwh(box)
        self.x = np.concatenate([z, np.zeros(4)])
        h = z[3]
        std = [2 * POS_W * h] * 4 + [10 * VEL_W * h * vel_scale] * 4
        self.P = np.diag(np.square(std))
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)

    def predict(self) -> None:
        h = max(self.x[3], 1.0)
        q = np.square([POS_W * h] * 4 + [VEL_W * h * self.vel_scale] * 4)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.diag(q)

    def update(self, box: np.ndarray) -> None:
        z = to_cwh(box)
        h = max(z[3], 1.0)
        r = np.square([POS_W * h] * 4)
        innov_cov = self.P[:4, :4] + np.diag(r)
        gain = self.P[:, :4] @ np.linalg.inv(innov_cov)
        self.x = self.x + gain @ (z - self.x[:4])
        self.P = self.P - gain @ self.P[:4, :]

    @property
    def box(self) -> np.ndarray:
        return to_xyxy(self.x)

    @property
    def velocity(self) -> np.ndarray:
        return self.x[4:6].copy()


# ----------------------------------------------------------- forward pass (ByteTrack)


@dataclass
class Fragment:
    frag_id: int
    kalman: Kalman
    hits: int = 1
    age: int = 0
    confirmed: bool = False
    frames: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)
    # One detector score per observed frame, index-aligned with `frames` and `boxes` --
    # the contract `tracker.Track` already states for its own `scores`, and which
    # `test_track_support.py` pins there.
    #
    # **Why a forward tracker has to keep them.** docs/PLAN.md step 4 measured that
    # admitting low-scoring boxes produces real extra events, and concluded that the next
    # mechanism is not another box filter: *the event layer has to see the detection
    # confidence a track was built from*, because a track of 0.15 boxes is not the claim
    # a 0.6 track is. `analytics.world.WorldObject.score` exists to carry exactly that and
    # `events.TrackSupport` to summarise it -- and neither could be filled from this
    # tracker, which took `scores` on every `update` and kept none of them.
    scores: list[float] = field(default_factory=list)
    review_crops: list[Image.Image] = field(default_factory=list)
    embed_crops: list[np.ndarray] = field(default_factory=list)
    obs_count: int = 0  # observations seen, for the embed-crop stride
    # The caller's appearance descriptor per observed frame, index-aligned with `frames`
    # and `boxes` -- the same field `tracker.Track.appearance` carries, for the same gate.
    appearance: list[np.ndarray] = field(default_factory=list)


@dataclass(frozen=True)
class Refusal:
    """One re-association the appearance gate refused: the instrument for reading it.

    The 2026-09-10 four-arm measurement (`runs/endings07`) found the gate moving nine
    endings on one camera and nothing that could say whether each split was two
    shoppers or one, because a refusal was recorded nowhere with a frame. This is that
    record: enough to cut a crop strip either side of the refusal and look.
    """

    frame: int
    frag_id: int
    box: np.ndarray  # the detection refused, xyxy
    distance: float  # appearance distance that exceeded the threshold
    age: int  # frames since the track's last observation, 2 or more by construction
    band: str  # "high" or "low"


# The shipped operating point, in one place. `serving/camera.py` names the two score
# edges (BIRTH_REF / KEEP_REF) and `scripts/serve_pilot.py` used to carry the other four
# as literals; `scripts/step6_events.py` ran a different tracker altogether, so the
# offline log and the live one could not be read against each other (PLAN 9.6). Every
# path now builds from `shipped_forward`, and a number here is the number everywhere.
SHIPPED_IOU = 0.3
SHIPPED_IOU_LOW = 0.4
SHIPPED_MAX_AGE = 5
SHIPPED_MIN_HITS = 2
MOT17_FPS = 25.0  # the frame rate the Kalman noise weights were tuned at


class OfflineForward:
    """The causal half: two-stage (high/low score) association over Kalman predictions.

    ``appearance_thr`` adds the re-association gate `tracker.Tracker` measured on
    2026-09-08: a track that has coasted through a gap is matched only if the detection
    looks like the track's last observation, at a distance the camera's own calibration
    set (`camera.json`'s `appearance_thr`). Consecutive-frame matches are never gated;
    a refused pair is recorded in `refusals` and the detection falls through -- to a
    birth if it was in the high band, to nothing if it was in the low one.
    """

    def __init__(
        self,
        high_thr,
        low_thr,
        iou_thr,
        iou_thr_low,
        max_age,
        min_hits,
        vel_scale,
        appearance_thr: float | None = None,
    ):
        self.high_thr = high_thr
        self.low_thr = low_thr
        self.iou_thr = iou_thr
        self.iou_thr_low = iou_thr_low
        self.max_age = max_age
        self.min_hits = min_hits
        self.vel_scale = vel_scale
        if appearance_thr is not None and appearance_thr < 0:
            raise ValueError(f"appearance_thr must be a distance >= 0, got {appearance_thr!r}")
        self.appearance_thr = appearance_thr
        self.tracks: list[Fragment] = []
        self.retired: list[Fragment] = []
        self.refusals: list[Refusal] = []
        self._next = 1
        self._appearance_seen: bool | None = None

    @staticmethod
    def _associate(tracks, boxes, thr):
        """Hungarian on IoU against predicted-or-last-observed, tracker.py's measured trick."""
        if not tracks or len(boxes) == 0:
            return {}, set(range(len(boxes)))
        pred = np.stack([t.kalman.box for t in tracks])
        m = iou(pred, boxes)
        obs = np.stack([t.boxes[-1] for t in tracks])
        m = np.maximum(m, iou(obs, boxes))
        pairs = {}
        for ti, di in _hungarian(-m):
            if m[ti, di] >= thr:
                pairs[ti] = di
        return pairs, set(range(len(boxes))) - set(pairs.values())

    def _gate(self, tracks, pairs, boxes, looks, frame_idx: int, band: str):
        """Drop every pair the appearance gate refuses, recording each. In place."""
        if self.appearance_thr is None or looks is None:
            return pairs
        for ti in list(pairs):
            t = tracks[ti]
            # `age` was incremented by the predict step: 1 means observed on the previous
            # frame and is never gated; 2 or more means the track coasted through a gap.
            if t.age <= 1 or not t.appearance:
                continue
            d = appearance_distance(t.appearance[-1], looks[pairs[ti]])
            if d > self.appearance_thr:
                self.refusals.append(
                    Refusal(
                        frame=int(frame_idx),
                        frag_id=t.frag_id,
                        box=np.asarray(boxes[pairs[ti]], float).copy(),
                        distance=float(d),
                        age=int(t.age),
                        band=band,
                    )
                )
                del pairs[ti]
        return pairs

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        frame_idx: int,
        appearance: np.ndarray | None = None,
    ) -> None:
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        if self.appearance_thr is not None and appearance is None:
            raise ValueError(
                "appearance_thr is set and this frame carried no appearance vectors; the "
                "gate can only refuse a re-association it can measure"
            )
        now = appearance is not None
        if self._appearance_seen is not None and self._appearance_seen != now:
            raise ValueError(
                "this tracker was updated with appearance vectors on one frame and "
                "without on another; `Fragment.appearance` must stay index-aligned with "
                "`frames`, so a gap cannot be recovered later"
            )
        self._appearance_seen = now
        looks = None
        if appearance is not None:
            looks = np.asarray(appearance, float)
            # An empty frame carries an empty (0, D) array; reshape(0, -1) is ambiguous.
            looks = looks.reshape(0, 0) if len(boxes) == 0 else looks.reshape(len(boxes), -1)
        for t in self.tracks:
            t.kalman.predict()
            t.age += 1
        hi_idx = np.where(scores >= self.high_thr)[0]
        lo_idx = np.where((scores >= self.low_thr) & (scores < self.high_thr))[0]
        hi_looks = None if looks is None else looks[hi_idx]
        lo_looks = None if looks is None else looks[lo_idx]

        # Stage 1: every live track against the high band.
        pairs, _ = self._associate(self.tracks, boxes[hi_idx], self.iou_thr)
        pairs = self._gate(self.tracks, pairs, boxes[hi_idx], hi_looks, frame_idx, "high")
        un_hi = set(range(len(hi_idx))) - set(pairs.values())
        matched_tracks = set()
        for ti, di in pairs.items():
            self._observe(
                self.tracks[ti],
                boxes[hi_idx[di]],
                frame_idx,
                float(scores[hi_idx[di]]),
                None if hi_looks is None else hi_looks[di],
            )
            matched_tracks.add(ti)

        # Stage 2: still-unmatched tracks against the low band. Stricter IoU: a
        # low-score box is a noisier box, and this stage exists to bridge occlusion,
        # not to grow the box into the fixture that caused it.
        rest = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        rest_tracks = [self.tracks[i] for i in rest]
        pairs2, _ = self._associate(rest_tracks, boxes[lo_idx], self.iou_thr_low)
        pairs2 = self._gate(rest_tracks, pairs2, boxes[lo_idx], lo_looks, frame_idx, "low")
        for ti, di in pairs2.items():
            self._observe(
                self.tracks[rest[ti]],
                boxes[lo_idx[di]],
                frame_idx,
                float(scores[lo_idx[di]]),
                None if lo_looks is None else lo_looks[di],
            )

        # Births: unmatched high-band detections only.
        for di in sorted(un_hi):
            box = boxes[hi_idx[di]].copy()
            t = Fragment(self._next, Kalman(box, self.vel_scale))
            t.frames.append(frame_idx)
            t.boxes.append(box)
            t.scores.append(float(scores[hi_idx[di]]))
            if hi_looks is not None:
                t.appearance.append(np.asarray(hi_looks[di], float).copy())
            t.confirmed = self.min_hits <= 1
            self.tracks.append(t)
            self._next += 1

        live, gone = [], []
        for t in self.tracks:
            (live if t.age <= self.max_age else gone).append(t)
        self.tracks = live
        self.retired.extend(t for t in gone if t.confirmed)

    def _observe(
        self, t: Fragment, box: np.ndarray, frame_idx: int, score: float, look=None
    ) -> None:
        t.kalman.update(box)
        t.hits += 1
        t.age = 0
        t.obs_count += 1
        t.frames.append(frame_idx)
        t.boxes.append(box.copy())
        t.scores.append(float(score))
        if look is not None:
            t.appearance.append(np.asarray(look, float).copy())
        if t.hits >= self.min_hits:
            t.confirmed = True

    def finished(self) -> list[Fragment]:
        return self.retired + [t for t in self.tracks if t.confirmed]


def as_track(fragment: Fragment) -> Track:
    """A forward tracker's `Fragment`, in the shape the L1 producer consumes.

    `analytics.world.world_frame` takes `tracker.Track`, and the serving path's tracker
    is this one, which produces `Fragment`. The two carry the same observations under
    different names -- `frag_id` against `track_id`, and a current box that a `Fragment`
    keeps in two places rather than one.

    **An adapter rather than one shared type**, because the difference is not accidental:
    a `Fragment` owns a Kalman filter, review crops and embedding crops, which are the
    offline re-identification pass's working state and have no business in a payload
    every consumer of the vector space reads. Merging them would put that state on the
    type `events`, `journey` and `dwell` all consume.

    **The current box follows `serving.camera.confirmed_track_boxes`'s rule**: an observed
    track contributes its observation, a coasting one its Kalman prediction, so a missed
    detection does not blink the position off. `world_frame` reads `age` for the same
    distinction and reports it as `WorldObject.observed`, so the two agree by
    construction rather than by coincidence.
    """
    observed = fragment.age == 0 and bool(fragment.boxes)
    box = np.asarray(fragment.boxes[-1] if observed else fragment.kalman.box, float)
    return Track(
        track_id=fragment.frag_id,
        box=box.copy(),
        hits=fragment.hits,
        age=fragment.age,
        frames=list(fragment.frames),
        boxes=[np.asarray(b, float).copy() for b in fragment.boxes],
        scores=list(fragment.scores),
        appearance=[np.asarray(a, float).copy() for a in fragment.appearance],
        confirmed=fragment.confirmed,
    )


def shipped_forward(fps: float, appearance_thr: float | None = None) -> OfflineForward:
    """The tracker that ships, at the rate the caller consumes frames.

    The score edges are `serving/camera.py`'s; the rest are `SHIPPED_*` above. ``fps`` is
    the *consumption* rate -- 5 on every analytics path -- and it scales the Kalman's
    velocity prior from MOT17's 25; passing the decode rate of a stream that is consumed
    slower would gate out every walking match (see `POS_W` / `VEL_W`).
    """
    from ..serving.camera import BIRTH_REF, KEEP_REF  # the edges live with the book

    if not fps > 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return OfflineForward(
        BIRTH_REF,
        KEEP_REF,
        SHIPPED_IOU,
        SHIPPED_IOU_LOW,
        SHIPPED_MAX_AGE,
        SHIPPED_MIN_HITS,
        MOT17_FPS / fps,
        appearance_thr=appearance_thr,
    )


class TwoStageForClip:
    """`OfflineForward` behind the interface `clip_tracks.track_clip` calls.

    `track_clip` calls ``update(boxes, frame_idx, scores=..., appearance=...)`` and reads
    ``finished()`` as `tracker.Track`s; `OfflineForward` takes ``(boxes, scores,
    frame_idx)`` and hands back `Fragment`s. This is the same adapter
    `scripts/track_endings.py` kept privately as `RecordingByteTrack`, in the package so
    `scripts/step6_events.py` can run the shipped tracker offline -- which is the whole
    point: the L3 log was produced by a tracker the serving path did not run.
    """

    def __init__(self, inner: OfflineForward) -> None:
        self.inner = inner

    @property
    def tracks(self) -> list[Fragment]:
        return self.inner.tracks

    @property
    def refusals(self) -> list[Refusal]:
        return self.inner.refusals

    def update(self, boxes, frame_idx, scores=None, appearance=None, **_ignored) -> None:
        b = np.asarray(boxes, float).reshape(-1, 4)
        if scores is None:
            raise ValueError("the two-stage tracker associates on score; pass scores=")
        self.inner.update(b, np.asarray(scores, float).reshape(-1), int(frame_idx), appearance)

    def finished(self) -> list[Track]:
        return [as_track(f) for f in self.inner.finished()]
