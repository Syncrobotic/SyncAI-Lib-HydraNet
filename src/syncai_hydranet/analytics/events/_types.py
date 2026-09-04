"""What an event *is*: the type list, the tiers, and the row a consumer receives.

Split out of one 1,425-line module and named with a leading underscore because it is
not the import surface -- `analytics.events` re-exports every name here, and
`tests/test_dispositions.py` and `scripts/mine_fall_candidates.py` both import off the
package rather than off this file. The package docstring carries the argument for why
none of this is a model output; this file carries the shapes that argument produces.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ._geometry import _point_in_polygon

# Grouped by the instrument that can resolve them, which is the same judgement that
# moved `product` out of the dense head and into detection: ask what can answer the
# question before asking what to call it. `TIERS` below states it per type, and a
# consumer that wants to know how much to trust a row reads that rather than a score.
EVENT_TYPES = (
    # --- track geometry: floor positions in metres over time. No new model, no new data.
    "zone_intrusion",  # a person's floor position entered a restricted polygon
    "line_cross",  # a person's floor path crossed a counting line, with a direction
    "occupancy_exceeded",  # more people in a zone than it is allowed to hold
    "loitering",  # one person, one zone, longer than the store permits
    "running",  # floor speed over a window, in m/s
    "tailgating",  # two tracks through one line inside a few seconds, same direction
    "crowd_forming",  # several people within a radius of each other, sustained
    "object_left",  # a bag that stopped moving and has no person near it
    "stock_removed",  # merchandise count in a zone dropped and stayed down
    "fall_candidate",  # box-shape proxy for a fall. NOT an alarm -- see the producer.
    # --- pose: 17 keypoints per person per frame, from a second-stage crop model.
    "fall",  # torso angle and hip collapse, sustained
    "crouch",  # the thing a box-shape proxy cannot tell a fall from
    "reach_to_shelf",  # a wrist over the terrain head's `fixture` region
    # --- temporal model over a whole track. Unbuilt; see UNBUILT.
    "fight",
)


TIERS = {
    "zone_intrusion": "track geometry",
    "line_cross": "track geometry",
    "occupancy_exceeded": "track geometry",
    "loitering": "track geometry",
    "running": "track geometry",
    "tailgating": "track geometry",
    "crowd_forming": "track geometry",
    "object_left": "track geometry",
    "stock_removed": "track geometry",
    "fall_candidate": "track geometry",
    "fall": "pose",
    "crouch": "pose",
    "reach_to_shelf": "pose",
    "fight": "temporal model",
}


# The blocker per unbuilt type, in the form "what would have to exist first". Read by
# `require_buildable`, so asking for one of these is an error naming the gap rather than
# an empty result that reads as "nothing happened".
UNBUILT = {
    "fight": (
        "needs a temporal model over a whole track, and the blocker is association "
        "rather than data: track length is a median 9-16 frames at 5 fps "
        "(RETAIL_DATA.md), and a clip-level classifier never receives a whole "
        "person. RWF-2000 is the plausible source and buying it first would train a "
        "model on an input this pipeline cannot deliver. Association metric, then "
        "association, then this -- ARCHITECTURE.md section 5. Nothing in "
        "datasets/studioa_clips has been checked for a single instance either, and "
        "`fall_candidate` below is how that gets checked without paying for it first"
    ),
}


def require_buildable(event_type: str) -> None:
    """Refuse an event type that has no producer, naming what it needs."""
    if event_type in UNBUILT:
        raise NotImplementedError(f"{event_type}: {UNBUILT[event_type]}")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event type {event_type!r}; known: {', '.join(EVENT_TYPES)}")


@dataclass(frozen=True)
class Zone:
    """A polygon on the floor, in **metres**, plus the thresholds that apply inside it.

    Metres and not pixels, and the polygon is per camera mount rather than per frame. A
    zone drawn on an image is invalidated by a knock to the bracket; this one is
    invalidated by nothing the camera does, because `pixel_to_ground` re-derives the
    pixels from the pose every time.
    """

    name: str
    polygon: np.ndarray  # (N, 2) x-z metres, ordered, implicitly closed
    max_occupancy: int | None = None
    loiter_seconds: float | None = None
    restricted: bool = False  # any presence at all is an event

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean per row of ``points`` (N, 2). NaN rows are False, deliberately.

        `track_ground_path` returns NaN where the foot point is at or above the horizon,
        which is a position the geometry declines to invent. Counting that as "inside"
        would put a shopper in a zone on the strength of a refusal to measure.
        """
        return _point_in_polygon(points, self.polygon)


@dataclass(frozen=True)
class CountingLine:
    """A directed segment on the floor, in metres. Crossing a->b right-hand side is `in`."""

    name: str
    a: np.ndarray  # (2,)
    b: np.ndarray  # (2,)


def zones_from_camera(cam_file: Any, kinds: Sequence[str] | None = None) -> list[Zone]:
    """A commissioned camera's polygon zones, as the event layer's `Zone`.

    `camera.json` and this module both call a floor region a "zone" and they are not the
    same object: `camera_json.Zone` is what the *camera* is -- a named polygon in metres,
    valid until the mount or the store moves -- and `Zone` here carries the thresholds a
    store manager changes on a Tuesday. The package docstring's third reason for keeping
    events out of the weights is the same reason for keeping policy out of `camera.json`,
    so this bridge deliberately carries **geometry only**: every policy field comes back
    at its default and the caller sets it.

    `entrance_line` is skipped because it is a line, not a region -- `counting_lines`
    below returns those. Without the split, a two-point polyline would arrive here as a
    degenerate polygon that `contains` calls False for every point on the floor, which is
    a zone that never fires and never errors.
    """
    wanted = None if kinds is None else set(kinds)
    return [
        Zone(name=z.name, polygon=np.asarray(z.points_m, dtype=float))
        for z in cam_file.zones
        if z.kind != "entrance_line" and (wanted is None or z.kind in wanted)
    ]


def counting_lines(cam_file: Any) -> list[CountingLine]:
    """A commissioned camera's `entrance_line` zones, as directed segments.

    A polyline with more than two points is refused rather than truncated: which pair a
    crossing test should use is a decision, and taking the first two silently would make
    a three-point entrance mean whatever its first segment happens to mean.
    """
    out = []
    for z in cam_file.zones:
        if z.kind != "entrance_line":
            continue
        if len(z.points_m) != 2:
            raise ValueError(
                f"entrance_line {z.name!r} has {len(z.points_m)} points; a counting line "
                "is one directed segment, and choosing which pair of a polyline to cross "
                "against is a decision this function must not make silently"
            )
        a, b = (np.asarray(pt, dtype=float) for pt in z.points_m)
        out.append(CountingLine(name=z.name, a=a, b=b))
    return out


@dataclass(frozen=True)
class TrackSupport:
    """How much detector confidence stands behind one event, over its own frames.

    **Why an event needs this at all**, measured 2026-08-26. Lowering the person birth
    threshold from 0.35 to 0.15 bought +51% detections and +141% tracks on seven healthy
    cameras -- real recall, on shoppers whose lower body is behind a counter -- and it
    also multiplied posture events by four. Making the dense head vouch for the low
    boxes did not help: it dropped 12-14% of them and produced *exactly* the unfiltered
    arm's events, because the boxes it removed were never the ones firing. The extra
    events come from real people detected at low confidence, whose keypoints are noisier
    and so produce more posture runs. A better box filter cannot reach that. What reaches
    it is a consumer that can see a 0.15-built track is not a 0.6-built one, which is
    what this carries.

    **It is not a confidence for the event.** The package docstring's rule stands -- an
    event is a threshold crossing and inventing a probability for it would be a number
    whose production nobody can name. These are the detector's own scores, reported as
    what they are, next to how much of the event the detector actually saw.

    `observed` counts the frames in the span the tracker matched a detection to; `span`
    counts the frames the event covers. They differ when a track coasted, and the
    difference is the point: a two-second `fall` seen in three frames out of ten is a
    different claim from one seen in all ten, and nothing before this could tell a
    consumer which it was holding.
    """

    score_p50: float
    score_min: float
    observed: int
    span: int

    def __post_init__(self):
        if self.observed < 1 or self.span < self.observed:
            raise ValueError(
                f"need 1 <= observed <= span, got observed={self.observed} span={self.span}"
            )

    @property
    def seen_fraction(self) -> float:
        """Frames of this event the detector actually produced a box for."""
        return self.observed / self.span


def support_for(track: Any, i0: int, i1: int) -> TrackSupport | None:
    """The detection support behind ``track``'s observations ``i0..i1`` inclusive.

    ``i0``/``i1`` index the track's *observed* frames -- `Track.frames`, `Track.boxes`
    and `Track.scores` are all in that space -- while `span` is computed in frame
    numbers, so a track that coasted through the middle of an event reports it.

    Returns None when the track carries no scores. That is legal and it is the state of
    every track this repository produced before 2026-08-26: `Tracker.update` takes them
    and does not require them. None means "not recorded", never "low" -- the distinction
    this project keeps everywhere, and the one a defaulted 0.0 would destroy.
    """
    scores = list(getattr(track, "scores", ()))
    if not scores:
        return None
    frames = list(track.frames)
    if len(scores) != len(frames):
        raise ValueError(
            f"track {getattr(track, 'track_id', '?')} has {len(scores)} scores for "
            f"{len(frames)} observed frames; they are index-aligned by contract, so a "
            "mismatch has no safe interpretation"
        )
    window = [float(x) for x in scores[i0 : i1 + 1]]
    if not window:
        return None
    return TrackSupport(
        score_p50=float(np.median(window)),
        score_min=float(min(window)),
        observed=len(window),
        span=int(frames[i1] - frames[i0] + 1),
    )


@dataclass(frozen=True)
class SecurityEvent:
    """One row of the security output. Stable across versions; new fields append.

    `value` and `threshold` are in the units named by `basis`, so a consumer can render
    "5.2 m of a 4.0 m line" or "7 people against a limit of 6" without a lookup table --
    and can see that an event at 6.01 is not the same claim as one at 14.
    """

    type: str
    camera: str
    frame_start: int
    frame_end: int
    fps: float
    track_ids: tuple[int, ...] = ()
    zone: str | None = None
    value: float | None = None
    threshold: float | None = None
    basis: str = ""
    # The detector confidence behind this event's own frames, or None when the producer
    # had none to give. See `TrackSupport` -- it is deliberately not a score for the
    # event, and `None` means "not recorded" rather than "low".
    support: TrackSupport | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # When the clip's first frame was recorded, timezone-aware. Optional and defaulted so
    # no builder signature changes: the wall clock is not something an event *detector*
    # knows, it is something the run knows, and `with_clip_start` attaches it at the edge.
    # See `started_at` for why a row without it is not deliverable.
    clip_start: datetime | None = None

    @property
    def seconds(self) -> float:
        """Duration in wall-clock seconds. Frames are the unit of record; this converts.

        `stage.StageFrame` states why the index is authoritative: a dropped frame changes
        the index and does not change the clock, and it is the index that decides whether
        two boxes are adjacent observations of one shopper.
        """
        return (self.frame_end - self.frame_start + 1) / self.fps

    def at(self, frame: int) -> datetime | None:
        """The wall clock at ``frame``, or None if this event was never stamped."""
        if self.clip_start is None:
            return None
        return self.clip_start + timedelta(seconds=frame / self.fps)

    @property
    def started_at(self) -> datetime | None:
        """When this event began, in real time.

        Frames stay the unit of record for every computation above -- that argument is in
        `stage.StageFrame` and it has not changed. This is the conversion at the edge, and
        without it the output is not deliverable: a security operator handed
        `frame_start: 5312` cannot pull the footage. Nothing in this package produced an
        absolute time until now, while `scripts/pull_studioa.py` had been parsing one out
        of every clip's name since the corpus was first pulled.
        """
        return self.at(self.frame_start)

    @property
    def ended_at(self) -> datetime | None:
        """When this event stopped. `frame_end` is inclusive, hence the +1."""
        return self.at(self.frame_end + 1)

    def as_row(self) -> dict:
        """Flat dict for JSONL, a database, or a dashboard. No numpy types survive it."""
        return {
            "type": self.type,
            "camera": self.camera,
            "frame_start": int(self.frame_start),
            "frame_end": int(self.frame_end),
            "seconds": round(self.seconds, 3),
            # ISO-8601 with an offset, so a reader never has to ask which zone a time is
            # in -- the mistake `pull_studioa` names in capitals, having pulled a closed
            # store at "16:00" because the filenames are UTC and the shops are UTC+8.
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at),
            "track_ids": [int(t) for t in self.track_ids],
            "zone": self.zone,
            "value": None if self.value is None else float(self.value),
            "threshold": None if self.threshold is None else float(self.threshold),
            "basis": self.basis,
            # Flattened with a prefix rather than nested, because `as_row` promises a
            # flat dict a database column can be made from. The keys are always present
            # so the shape does not depend on whether the producer had scores.
            "support_score_p50": None if self.support is None else self.support.score_p50,
            "support_score_min": None if self.support is None else self.support.score_min,
            "support_observed": None if self.support is None else self.support.observed,
            "support_span": None if self.support is None else self.support.span,
            **dict(self.extra),
        }


def _iso(when: datetime | None) -> str | None:
    return None if when is None else when.isoformat()


CLIP_NAME = re.compile(r"archive_(\d{8}-\d{6})_")


def clip_start_from_name(path: str | Path, tz: timezone = UTC) -> datetime:
    """`archive_20260816-113012_20260816-113518.mp4` -> when recording started.

    **The stamp in the name is UTC and the stores are UTC+8.** That is not a detail; it
    is the failure `scripts/pull_studioa.py` documents in capitals, having asked for
    "16:00, the busy hour" and received a greyscale IR clip of a closed shop. So this
    reads the name as UTC and converts to ``tz``, and every timestamp downstream carries
    its offset rather than relying on the reader to know.

    Raises rather than guessing: a clip whose name does not carry a time cannot be given
    one, and an invented start would put every event in the report at the wrong moment
    while looking exactly as authoritative as a correct one.
    """
    m = CLIP_NAME.search(Path(path).name)
    if m is None:
        raise ValueError(
            f"{Path(path).name!r} does not carry a recording time. Expected the site "
            f"corpus's `archive_YYYYmmdd-HHMMSS_...` naming; pass the start explicitly "
            f"for a clip from anywhere else."
        )
    utc = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    return utc.astimezone(tz)


def with_clip_start(
    events: Sequence[SecurityEvent], clip_start: datetime | None
) -> list[SecurityEvent]:
    """Attach a clip's wall-clock start to a batch of events.

    At the edge rather than threaded through ten builder signatures: an event detector
    knows about frames and a run knows about clocks, and `stage.StageFrame` is explicit
    that the conversion belongs here.
    """
    if clip_start is None:
        return list(events)
    if clip_start.tzinfo is None:
        raise ValueError(
            f"clip_start {clip_start!r} has no timezone. A naive time in a report that "
            f"crosses a UTC+8 store and a UTC bucket is a time nobody can act on."
        )
    return [replace(e, clip_start=clip_start) for e in events]
