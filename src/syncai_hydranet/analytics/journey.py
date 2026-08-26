"""The first consumer of the vector space: one track's route across the floor, in metres.

`world.py` typed the metre side and shipped a producer, and said in its own words what it
still was not: "**adopted**. One producer, no consumer yet." This is the consumer, and it
is a new reader rather than a rewrite of an old one on purpose. `dwell.track_ground_path`
and `events/zones.py` each project their own foot points, and `world.py` records why they
are left alone: the producer here undistorts and `track_ground_path` does not, so moving
them changes every dwell, path and heatmap number already reported. That is a
re-baseline, it belongs next to a measurement of what moved, and it is not this.

---------------------------------------------------------------------------
WHAT THIS ANSWERS, AND WHAT IT MUST NOT BE READ AS

The question is "this person walked from A to B and stood at C for how long". A `Journey`
answers it as an ordered list of `Visit`s with durations, plus the transitions between
them and the floor distance walked.

**It is a track's journey, not a customer's, and the difference is measured.** The events
package records what person tracks are worth on this footage today: a 4.6-minute clip
fragments into **1,234 tracks**. A shopper who walks A -> B -> C and is re-identified
three times on the way produces three journeys, each of them true and none of them the
visit. So `Journey.track_id` is a track id, the field is not named `customer_id`, and
nothing here merges two journeys into one. Association is what fixes that -- PLAN step 5
-- and `reid_metrics.py` is where it will be shown to have worked.

**One space, always.** Every frame handed in must carry the same `space`. Today the only
space is `camera_floor(<camera_id>)` with its origin under the camera, so two cameras'
`x_m` are two different x and a route stitched across them would be arithmetic on
unrelated axes. That is refused here rather than discovered in a dashboard: `world.py`
carries `space` per frame for exactly this, and a consumer that ignores it wastes the
field.

---------------------------------------------------------------------------
THREE DECISIONS

**A re-entry is a second visit, not a longer one.** A shopper who leaves a zone and comes
back has done two things, and summing them into one hides the leaving. `time_in` sums
them for the caller who wants the total; the list keeps what happened.

**A position that was never measured contributes nothing, and says so.** `Zone.contains`
calls a NaN row False deliberately -- "counting that as inside would put a shopper in a
zone on the strength of a refusal to measure" -- and the path length here skips the same
rows instead of treating a refusal as a jump to the origin. `Visit.observed` against
`Visit.span` is what a reader checks: a 40-second dwell measured in four frames is a
different claim from one measured in forty.

**Duration comes from the clock the caller supplies, or from nothing.** `world.py`
measured that every clip in the site corpus writes `30/1` into `r_frame_rate` regardless
of its true rate, so `seconds` is None unless the frames carry `time_s` or the caller
knowingly passes an fps -- the same trade, made in the same place, as `world_frame`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .events._types import Zone
from .world import WorldFrame


@dataclass(frozen=True)
class Visit:
    """One continuous stay of one track inside one zone.

    `span` counts frames from entry to exit inclusive; `observed` counts the frames in
    that span where this track actually appeared with a measured position. They differ
    when the track coasted or its foot point went above the horizon.
    """

    zone: str
    frame_start: int
    frame_end: int
    span: int
    observed: int
    seconds: float | None

    @property
    def seen_fraction(self) -> float:
        return self.observed / self.span


@dataclass(frozen=True)
class Journey:
    """One track's route: where it went, in what order, and how long it stayed.

    `path_m` is the floor distance between consecutive *measured* positions. It is a
    lower bound on the distance walked, twice over: an unmeasured frame is skipped rather
    than bridged, and a track that ends at a re-identification ends the journey with it.
    """

    track_id: int
    camera_id: str
    space: str
    frame_start: int
    frame_end: int
    observed: int
    path_m: float
    seconds: float | None
    visits: tuple[Visit, ...]
    score_p50: float | None

    @property
    def route(self) -> tuple[str, ...]:
        """The zones in the order they were entered, re-entries included."""
        return tuple(v.zone for v in self.visits)

    @property
    def transitions(self) -> tuple[tuple[str, str], ...]:
        """Consecutive zone pairs -- "walked from A to B", one per move."""
        return tuple(pairwise(self.route))

    def time_in(self, zone: str) -> float | None:
        """Total seconds spent in ``zone`` across every visit, or None if unclocked."""
        parts = [v.seconds for v in self.visits if v.zone == zone]
        if not parts or any(p is None for p in parts):
            return None
        return float(sum(p for p in parts if p is not None))

    def as_row(self) -> dict:
        """Flat dict for JSONL or a dashboard. No numpy, no NaN."""
        return {
            "track_id": int(self.track_id),
            "camera_id": self.camera_id,
            "space": self.space,
            "frame_start": int(self.frame_start),
            "frame_end": int(self.frame_end),
            "observed": int(self.observed),
            "path_m": round(float(self.path_m), 3),
            "seconds": None if self.seconds is None else round(self.seconds, 3),
            "route": list(self.route),
            "score_p50": self.score_p50,
            "visits": [
                {
                    "zone": v.zone,
                    "frame_start": int(v.frame_start),
                    "frame_end": int(v.frame_end),
                    "span": int(v.span),
                    "observed": int(v.observed),
                    "seconds": None if v.seconds is None else round(v.seconds, 3),
                }
                for v in self.visits
            ],
        }


def journeys(
    frames: Sequence[WorldFrame],
    zones: Sequence[Zone],
    *,
    name: str = "person",
    fps: float | None = None,
) -> list[Journey]:
    """Vector-space frames -> one `Journey` per track, ordered by first appearance.

    ``frames`` need not be sorted and need not be contiguous; they are read in frame
    order and a track missing from a frame is simply not seen in it. ``name`` filters the
    objects, because a bag's route is not a shopper's and the payload carries both.

    ``fps`` is used only where a frame has no `time_s`, and it is the caller's knowing
    trade -- see the module docstring on why a nominal rate is not a clock.
    """
    if not frames:
        return []
    spaces = {f["space"] for f in frames}
    if len(spaces) > 1:
        raise ValueError(
            f"journeys over {len(spaces)} metric spaces: {sorted(spaces)}. Each "
            "`camera_floor(...)` has its origin under its own camera, so a route across "
            "two of them is arithmetic on unrelated axes. Build one journey per space "
            "until a store frame exists."
        )
    ordered = sorted(frames, key=lambda f: f["frame_index"])
    space = ordered[0]["space"]
    camera_id = ordered[0]["camera_id"]

    seen: dict[int, list[tuple[int, float | None, np.ndarray | None, float | None]]] = {}
    for f in ordered:
        idx = int(f["frame_index"])
        t_s = f["time_s"]
        for o in f["objects"]:
            if o["name"] != name:
                continue
            x, z = float(o["x_m"]), float(o["z_m"])
            pos = np.array([x, z]) if math.isfinite(x) and math.isfinite(z) else None
            seen.setdefault(int(o["track_id"]), []).append((idx, t_s, pos, o["score"]))

    out: list[Journey] = []
    for track_id, rows in sorted(seen.items(), key=lambda kv: kv[1][0][0]):
        out.append(_journey(track_id, camera_id, space, rows, zones, fps))
    return out


def _seconds(f0: int, f1: int, t0: float | None, t1: float | None, fps: float | None):
    """Inclusive duration of frames f0..f1, from the clock if there is one."""
    if t0 is not None and t1 is not None:
        # +1 frame so a single-frame visit is one frame long rather than zero. The frame
        # interval is taken from the clock where two frames give one, else from fps.
        step = (t1 - t0) / (f1 - f0) if f1 > f0 else (1.0 / fps if fps else 0.0)
        return float(t1 - t0 + step)
    if fps:
        return (f1 - f0 + 1) / float(fps)
    return None


def _journey(track_id, camera_id, space, rows, zones, fps) -> Journey:
    frames = [r[0] for r in rows]
    times = {r[0]: r[1] for r in rows}
    scores = [r[3] for r in rows if r[3] is not None]
    measured = [(r[0], r[2]) for r in rows if r[2] is not None]

    path = 0.0
    for (_, a), (_, b) in pairwise(measured):
        path += float(np.linalg.norm(b - a))

    visits: list[Visit] = []
    for zone in zones:
        inside = [
            fi
            for fi, pos in measured
            if bool(zone.contains(np.asarray(pos, dtype=float).reshape(1, 2))[0])
        ]
        for start, end in _contiguous(inside, frames):
            span_frames = [fi for fi in frames if start <= fi <= end]
            visits.append(
                Visit(
                    zone=zone.name,
                    frame_start=start,
                    frame_end=end,
                    span=end - start + 1,
                    observed=len(span_frames),
                    seconds=_seconds(start, end, times.get(start), times.get(end), fps),
                )
            )
    visits.sort(key=lambda v: (v.frame_start, v.zone))

    return Journey(
        track_id=track_id,
        camera_id=camera_id,
        space=space,
        frame_start=frames[0],
        frame_end=frames[-1],
        observed=len(frames),
        path_m=path,
        seconds=_seconds(
            frames[0], frames[-1], times.get(frames[0]), times.get(frames[-1]), fps
        ),
        visits=tuple(visits),
        score_p50=float(np.median(scores)) if scores else None,
    )


def _contiguous(inside: list[int], frames: list[int]) -> list[tuple[int, int]]:
    """Group frames inside a zone into stays, breaking only where the track was seen out.

    A frame the track was not seen in does **not** end a visit: a tracker that missed one
    observation has not reported the shopper leaving, and splitting there would turn one
    40-second dwell into two 20-second ones. A frame where it *was* seen and was outside
    does end it, because that is an observation of leaving.
    """
    if not inside:
        return []
    inside_set = set(inside)
    runs: list[tuple[int, int]] = []
    start = prev = inside[0]
    for fi in inside[1:]:
        left = any(f > prev and f < fi and f not in inside_set for f in frames)
        if left:
            runs.append((start, prev))
            start = fi
        prev = fi
    runs.append((start, prev))
    return runs
