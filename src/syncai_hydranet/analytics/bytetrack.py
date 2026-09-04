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
from .tracker import iou

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
    review_crops: list[Image.Image] = field(default_factory=list)
    embed_crops: list[np.ndarray] = field(default_factory=list)
    obs_count: int = 0  # observations seen, for the embed-crop stride


class OfflineForward:
    """The causal half: two-stage (high/low score) association over Kalman predictions."""

    def __init__(self, high_thr, low_thr, iou_thr, iou_thr_low, max_age, min_hits, vel_scale):
        self.high_thr = high_thr
        self.low_thr = low_thr
        self.iou_thr = iou_thr
        self.iou_thr_low = iou_thr_low
        self.max_age = max_age
        self.min_hits = min_hits
        self.vel_scale = vel_scale
        self.tracks: list[Fragment] = []
        self.retired: list[Fragment] = []
        self._next = 1

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

    def update(self, boxes: np.ndarray, scores: np.ndarray, frame_idx: int) -> None:
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        for t in self.tracks:
            t.kalman.predict()
            t.age += 1

        hi_idx = np.where(scores >= self.high_thr)[0]
        lo_idx = np.where((scores >= self.low_thr) & (scores < self.high_thr))[0]

        # Stage 1: every live track against the high band.
        pairs, un_hi = self._associate(self.tracks, boxes[hi_idx], self.iou_thr)
        matched_tracks = set()
        for ti, di in pairs.items():
            self._observe(self.tracks[ti], boxes[hi_idx[di]], frame_idx)
            matched_tracks.add(ti)

        # Stage 2: still-unmatched tracks against the low band. Stricter IoU: a
        # low-score box is a noisier box, and this stage exists to bridge occlusion,
        # not to grow the box into the fixture that caused it.
        rest = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        pairs2, _ = self._associate(
            [self.tracks[i] for i in rest], boxes[lo_idx], self.iou_thr_low
        )
        for ti, di in pairs2.items():
            self._observe(self.tracks[rest[ti]], boxes[lo_idx[di]], frame_idx)

        # Births: unmatched high-band detections only.
        for di in sorted(un_hi):
            box = boxes[hi_idx[di]].copy()
            t = Fragment(self._next, Kalman(box, self.vel_scale))
            t.frames.append(frame_idx)
            t.boxes.append(box)
            t.confirmed = self.min_hits <= 1
            self.tracks.append(t)
            self._next += 1

        live, gone = [], []
        for t in self.tracks:
            (live if t.age <= self.max_age else gone).append(t)
        self.tracks = live
        self.retired.extend(t for t in gone if t.confirmed)

    def _observe(self, t: Fragment, box: np.ndarray, frame_idx: int) -> None:
        t.kalman.update(box)
        t.hits += 1
        t.age = 0
        t.obs_count += 1
        t.frames.append(frame_idx)
        t.boxes.append(box.copy())
        if t.hits >= self.min_hits:
            t.confirmed = True

    def finished(self) -> list[Fragment]:
        return self.retired + [t for t in self.tracks if t.confirmed]
