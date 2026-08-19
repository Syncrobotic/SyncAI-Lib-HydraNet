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
from datetime import datetime, timedelta, timezone
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
            **dict(self.extra),
        }


def _iso(when: datetime | None) -> str | None:
    return None if when is None else when.isoformat()


CLIP_NAME = re.compile(r"archive_(\d{8}-\d{6})_")


def clip_start_from_name(path: str | Path, tz: timezone = timezone.utc) -> datetime:
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
    utc = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
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
