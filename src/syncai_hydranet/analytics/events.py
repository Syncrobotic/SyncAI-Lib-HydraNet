"""Security events: the second stage's *output* contract, and why none of it is a head.

`stage.py` names this stage and types what enters it (`StageFrame`). This types what
leaves it, for the security half of the deployment: a shopper entered a restricted zone,
too many people are at the counter, someone has stood in one place for four minutes, a
bag has been left, stock has gone off a shelf.

---------------------------------------------------------------------------
WHY THESE ARE NOT MODEL OUTPUTS, AND THE ARGUMENT IS ALREADY IN THIS REPOSITORY

`RETAIL.md` section 2 refuses to make "walkable area within 5 m" a class, and the
three reasons transfer to every event here without modification:

1. **Annotators cannot label it consistently.** Where a restricted zone ends is a
   decision, not an appearance. Two annotators disagree by a metre and the model trains
   their average.
2. **It dies when the camera moves.** A zone drawn in pixels is wrong the moment the
   mount is knocked, and nothing in training reports it. A zone in metres on the floor
   survives, because the homography absorbs the change.
3. **It freezes a product parameter into the weights.** "Four minutes is loitering" and
   "six people is too many" are settings a store manager changes on a Tuesday. In a
   config they are a number; in a checkpoint they are a retraining run.

So every threshold below is an argument to a function, every zone is a polygon **in
metres on the ground plane**, and the model contributes exactly two things: boxes and a
terrain map.

---------------------------------------------------------------------------
WHAT AN EVENT CARRIES, AND WHAT IT DELIBERATELY DOES NOT

No score. These are threshold crossings on measured quantities, so an event carries the
`value` it measured and the `threshold` it crossed, and a reader can see how marginal it
was without a number that looks like a probability and is not one. A confidence here
would be invented, and this project has a standing rule against numbers whose production
nobody can name (`ARCHITECTURE.md` section 1).

`basis` names the instrument instead: which measurement produced this event. That is the
same rule applied to the event itself -- "before attributing a number to a mechanism,
check what produced the number".

---------------------------------------------------------------------------
WHAT IS NOT BUILT, AND IT IS NOT AN OVERSIGHT

`fall` and `fight` are in `EVENT_TYPES` because the schema is a contract with whatever
consumes these rows and adding a field later is more expensive than reserving one. They
have no producer here. Both need a temporal model over crops, and
`ARCHITECTURE.md`'s "what not to build" names the precondition nobody has met:
**confirm the behaviour occurs on these cameras before paying for the annotation.** MERL
Shopping is a grocery aisle of closed shelving; these are open display tables. A
plausible detector shipped unmeasured is the failure this project ranks worst.

`UNBUILT` below states the blocker per type, so a consumer that asks for one gets a
refusal naming what is missing rather than an empty list that reads like "no events".

---------------------------------------------------------------------------
THE PRECONDITION EVERY ROW HERE INHERITS

All of it stands on person tracks, and `RETAIL_DATA.md` measured what those are
worth today: a 4.6-minute clip fragments into **1234 tracks**. Occupancy counts tracks,
loitering measures how long one lasts, and an intrusion is one track crossing a line --
so fragmentation does not add noise to these numbers, it biases them, and in a known
direction. Nothing here fixes that; association does, and `reid_metrics.py` is where it
will be shown to have worked.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry.ground import Camera, GroundPlane
from .dwell import track_ground_path
from .tracker import Track

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
    "reach_to_shelf",  # a wrist inside the terrain head's `fixture` region
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


# --------------------------------------------------------------------------- geometry


def _point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd ray casting, vectorised over points. No matplotlib, no shapely.

    Both are heavier dependencies than 12 lines, and this project already refuses a
    dependency for a small numeric job (`tracker.py` on scipy and the Hungarian
    assignment). The boundary case is left as-is rather than tuned: a track sitting
    exactly on a zone edge for several frames flips, and that is visible in the frame
    span rather than hidden by an epsilon whose value nobody measured.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    poly = np.asarray(polygon, dtype=float)
    if len(poly) < 3:
        raise ValueError(f"a zone needs at least 3 corners, got {len(poly)}")
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1, strict=True):
        straddles = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at_y = (bx - ax) * (y - ay) / (by - ay) + ax
        inside ^= straddles & (x < x_at_y)
    return inside & np.isfinite(x) & np.isfinite(y)


def _cross2(d: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The z component of a 2-D cross product, which is all the side tests need.

    `np.cross` on 2-D vectors is deprecated in NumPy 2.0 and scheduled for removal;
    writing the one component out is smaller than a deprecation shim and says what is
    actually being computed.
    """
    p = np.atleast_2d(p)
    return d[0] * p[..., 1] - d[1] * p[..., 0]


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Index spans where ``flags`` is True, as inclusive (start, end) pairs."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


# --------------------------------------------------------------------------- detectors


def zone_events(
    tracks: list[Track],
    zones: list[Zone],
    cam: Camera,
    plane: GroundPlane,
    fps: float,
    camera: str,
    min_seconds: float = 1.0,
) -> list[SecurityEvent]:
    """`zone_intrusion` for restricted zones, `loitering` for zones with a dwell limit.

    One pass because both read the same quantity -- how long one track's floor position
    stayed inside one polygon -- and computing it twice invites the two to disagree.

    `min_seconds` exists because a foot point jitters across a boundary. It is a
    *duration*, not a smoothing constant: at 5 fps a shopper walking past a zone edge is
    inside for one or two frames, and a rule that fires on those reports the boundary
    rather than the behaviour.
    """
    events: list[SecurityEvent] = []
    for track in tracks:
        if not track.frames:
            continue
        path = track_ground_path(track, cam, plane)
        frames = np.asarray(track.frames)
        for zone in zones:
            inside = zone.contains(path)
            for i0, i1 in _runs(inside):
                seconds = (frames[i1] - frames[i0] + 1) / fps
                if zone.restricted and seconds >= min_seconds:
                    events.append(
                        SecurityEvent(
                            type="zone_intrusion",
                            camera=camera,
                            frame_start=int(frames[i0]),
                            frame_end=int(frames[i1]),
                            fps=fps,
                            track_ids=(track.track_id,),
                            zone=zone.name,
                            value=seconds,
                            threshold=min_seconds,
                            basis="seconds a tracked foot point stayed inside the polygon",
                        )
                    )
                if zone.loiter_seconds is not None and seconds >= zone.loiter_seconds:
                    events.append(
                        SecurityEvent(
                            type="loitering",
                            camera=camera,
                            frame_start=int(frames[i0]),
                            frame_end=int(frames[i1]),
                            fps=fps,
                            track_ids=(track.track_id,),
                            zone=zone.name,
                            value=seconds,
                            threshold=zone.loiter_seconds,
                            basis="seconds a single track stayed inside the polygon",
                            # The number this event is most likely to be wrong by, stated
                            # in the row rather than in a footnote: a fragmenting tracker
                            # splits one four-minute shopper into twenty tracks and each
                            # falls under the threshold. Loitering is the event most
                            # damaged by that, because it is the only one that needs a
                            # track to survive.
                            extra={"track_frames": len(track.frames)},
                        )
                    )
    return events


def occupancy_events(
    tracks: list[Track],
    zone: Zone,
    cam: Camera,
    plane: GroundPlane,
    fps: float,
    camera: str,
    min_seconds: float = 2.0,
) -> list[SecurityEvent]:
    """One event per span where the count inside ``zone`` exceeded its limit.

    Counts **tracks**, which is the honest unit and not the wanted one. Two fragments of
    one shopper are two tracks, so this over-counts by exactly the fragmentation rate --
    the same bias `RETAIL_DATA.md` describes for demographics, arriving at a rule
    that raises an alarm. Until association is fixed, read a queue alarm as an upper
    bound.
    """
    if zone.max_occupancy is None:
        raise ValueError(f"zone {zone.name!r} has no max_occupancy, so nothing to exceed")
    per_frame: dict[int, int] = {}
    for track in tracks:
        if not track.frames:
            continue
        inside = zone.contains(track_ground_path(track, cam, plane))
        for frame, is_in in zip(track.frames, inside, strict=True):
            if is_in:
                per_frame[frame] = per_frame.get(frame, 0) + 1
    if not per_frame:
        return []
    frames = np.arange(min(per_frame), max(per_frame) + 1)
    counts = np.array([per_frame.get(int(f), 0) for f in frames])
    events = []
    for i0, i1 in _runs(counts > zone.max_occupancy):
        if (frames[i1] - frames[i0] + 1) / fps < min_seconds:
            continue
        events.append(
            SecurityEvent(
                type="occupancy_exceeded",
                camera=camera,
                frame_start=int(frames[i0]),
                frame_end=int(frames[i1]),
                fps=fps,
                zone=zone.name,
                value=float(counts[i0 : i1 + 1].max()),
                threshold=float(zone.max_occupancy),
                basis="distinct track ids whose foot point was inside the polygon",
            )
        )
    return events


def line_events(
    tracks: list[Track],
    line: CountingLine,
    cam: Camera,
    plane: GroundPlane,
    fps: float,
    camera: str,
) -> list[SecurityEvent]:
    """One event per crossing of ``line``, with the direction it was crossed in.

    Direction comes from the sign of the cross product, so `in` and `out` are properties
    of how the line was drawn and not of where the door is. Draw it once per camera and
    write down which side is which; a line whose orientation is guessed produces a
    footfall count with the sign flipped, which looks like a plausible number.
    """
    a, b = np.asarray(line.a, dtype=float), np.asarray(line.b, dtype=float)
    d = b - a
    events = []
    for track in tracks:
        path = track_ground_path(track, cam, plane)
        if len(path) < 2:
            continue
        frames = np.asarray(track.frames)
        side = np.sign(_cross2(d, path - a))
        for i in range(len(path) - 1):
            if not (np.isfinite(path[i]).all() and np.isfinite(path[i + 1]).all()):
                continue
            if side[i] == 0 or side[i + 1] == 0 or side[i] == side[i + 1]:
                continue
            if not _segments_cross(a, b, path[i], path[i + 1]):
                continue  # the same side flip, but off the end of the segment
            events.append(
                SecurityEvent(
                    type="line_cross",
                    camera=camera,
                    frame_start=int(frames[i]),
                    frame_end=int(frames[i + 1]),
                    fps=fps,
                    track_ids=(track.track_id,),
                    zone=line.name,
                    value=float(side[i + 1]),
                    threshold=0.0,
                    basis="sign change of the cross product across the counting line",
                    extra={"direction": "in" if side[i + 1] > 0 else "out"},
                )
            )
    return events


def _segments_cross(p0, p1, q0, q1) -> bool:
    """Do segments p0-p1 and q0-q1 properly intersect? Collinear counts as no."""

    def orient(a, b, c):
        return np.sign(_cross2(b - a, c - a))

    o1, o2 = orient(p0, p1, q0), orient(p0, p1, q1)
    o3, o4 = orient(q0, q1, p0), orient(q0, q1, p1)
    return bool(o1 != o2 and o3 != o4 and o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0)


def object_left_events(
    bag_tracks: list[Track],
    person_tracks: list[Track],
    cam: Camera,
    plane: GroundPlane,
    fps: float,
    camera: str,
    still_seconds: float = 30.0,
    still_radius_m: float = 0.5,
    owner_radius_m: float = 2.0,
) -> list[SecurityEvent]:
    """A bag that stopped moving and had nobody near it for ``still_seconds``.

    Two thresholds in metres and one in seconds, all arguments, all a store's decision --
    a 30-second unattended bag is an airport rule and a shop's is usually minutes.

    **Its recall on these cameras is unknown and that is a property of the data, not of
    this rule.** `bag` is trained on COCO alone (see `label_maps_retail_security`): web
    photography, eye level, uncompressed, against overhead h.264 store cameras. The rule
    is deterministic; what feeds it is a bootstrap detector nobody has scored on site
    footage. Do not put a number on this event's precision until a site clip carries bag
    boxes.
    """
    events = []
    for bag in bag_tracks:
        path = track_ground_path(bag, cam, plane)
        frames = np.asarray(bag.frames)
        if len(frames) < 2:
            continue
        finite = np.asarray(np.isfinite(path).all(axis=1))
        for i0, i1 in _runs(finite):
            span = path[i0 : i1 + 1]
            seconds = (frames[i1] - frames[i0] + 1) / fps
            if seconds < still_seconds:
                continue
            spread = float(np.linalg.norm(span - span.mean(axis=0), axis=1).max())
            if spread > still_radius_m:
                continue
            nearest = _nearest_person_distance(
                span.mean(axis=0), person_tracks, frames[i0], frames[i1], cam, plane
            )
            if nearest <= owner_radius_m:
                continue
            events.append(
                SecurityEvent(
                    type="object_left",
                    camera=camera,
                    frame_start=int(frames[i0]),
                    frame_end=int(frames[i1]),
                    fps=fps,
                    track_ids=(bag.track_id,),
                    value=seconds,
                    threshold=still_seconds,
                    basis=(
                        "seconds a bag track stayed within still_radius_m with no "
                        "person inside owner_radius_m"
                    ),
                    extra={
                        "spread_m": round(spread, 3),
                        "nearest_person_m": (
                            None if np.isinf(nearest) else round(float(nearest), 3)
                        ),
                    },
                )
            )
    return events


def _nearest_person_distance(
    point: np.ndarray,
    person_tracks: list[Track],
    frame_start: int,
    frame_end: int,
    cam: Camera,
    plane: GroundPlane,
) -> float:
    """Closest any person's floor position came to ``point`` in the frame span. inf if none."""
    best = np.inf
    for person in person_tracks:
        if not person.frames:
            continue
        path = track_ground_path(person, cam, plane)
        frames = np.asarray(person.frames)
        window = (frames >= frame_start) & (frames <= frame_end)
        if not window.any():
            continue
        d = np.linalg.norm(path[window] - point, axis=1)
        d = d[np.isfinite(d)]
        if len(d):
            best = min(best, float(d.min()))
    return best


def stock_removed_events(
    counts: dict[int, int],
    zone: Zone,
    fps: float,
    camera: str,
    drop: int = 1,
    sustained_seconds: float = 10.0,
) -> list[SecurityEvent]:
    """Merchandise count in a zone fell by ``drop`` and stayed down.

    ``counts`` is frame index -> number of merchandise boxes whose foot point was inside
    the zone; `zone_stock_counts` below builds it from stage frames. Kept as an argument
    rather than computed here so the same rule can run over a smoothed or a raw series
    and the caller can see which it used.

    **Sustained is the whole rule.** A shopper standing in front of a shelf hides stock
    for a few seconds, and a detector that fires on that reports occlusion as theft. The
    baseline is the median over the frames before the drop rather than the immediately
    preceding frame, for the same reason.
    """
    if not counts:
        return []
    frames = np.arange(min(counts), max(counts) + 1)
    series = np.array([counts.get(int(f), 0) for f in frames], dtype=float)
    hold = max(round(sustained_seconds * fps), 1)
    events = []
    i = hold
    while i + hold <= len(series):
        before = float(np.median(series[max(0, i - hold) : i]))
        after = series[i : i + hold]
        if before - float(after.max()) >= drop:
            events.append(
                SecurityEvent(
                    type="stock_removed",
                    camera=camera,
                    frame_start=int(frames[i]),
                    frame_end=int(frames[min(i + hold, len(frames)) - 1]),
                    fps=fps,
                    zone=zone.name,
                    value=before - float(after.max()),
                    threshold=float(drop),
                    basis=(
                        "median merchandise box count over the preceding "
                        f"{sustained_seconds:g}s minus the maximum over the following "
                        f"{sustained_seconds:g}s"
                    ),
                )
            )
            i += hold  # one event per drop, not one per frame of it
        else:
            i += 1
    return events


def zone_stock_counts(
    frames: list[dict],
    zone: Zone,
    cam: Camera,
    plane: GroundPlane,
    classes: tuple[str, ...] = ("boxed_stock", "device"),
    score_thr: float = 0.3,
) -> dict[int, int]:
    """Merchandise boxes inside ``zone`` per frame, from `stage.StageFrame` payloads.

    Reads `class_names` off each frame rather than assuming the vocabulary, which is the
    rule `StageFrame` states: an exported engine narrowed with `--detection-classes`
    keeps only its own binding names, so a consumer that assumes a fixed list silently
    renames every box.
    """
    out: dict[int, int] = {}
    for frame in frames:
        names = frame["class_names"]
        wanted = {i for i, n in enumerate(names) if n in classes}
        boxes, scores, labels = frame["boxes"], frame["scores"], frame["labels"]
        keep = np.array(
            [
                bool(lab in wanted and sc >= score_thr)
                for lab, sc in zip(labels, scores, strict=True)
            ],
            dtype=bool,
        )
        if not keep.any():
            out[int(frame["frame_index"])] = 0
            continue
        sel = boxes[keep]
        foot_u = (sel[:, 0] + sel[:, 2]) / 2
        foot_v = sel[:, 3]
        from ..geometry.ground import pixel_to_ground

        x, z = pixel_to_ground(foot_u, foot_v, cam, plane)
        out[int(frame["frame_index"])] = int(zone.contains(np.stack([x, z], axis=-1)).sum())
    return out


# ------------------------------------------------- behaviour, tier 1: track geometry
#
# Everything below this line answers a *behaviour* question, and the three sections are
# ordered by the instrument that can resolve one. This first section needs no model that
# does not already exist and no data that has not already been captured: a floor path in
# metres and a clock. That is worth stating plainly, because the behaviour ask usually
# arrives attached to an annotation budget, and a good share of it is geometry.
#
# The unit is metres per second and not pixels per frame, and that is the whole reason
# these transfer between cameras. A pixel speed threshold tuned on Kaohsiung-cam08 is a
# statement about that lens and that mount.


def speed_events(
    tracks: list[Track],
    cam: Camera,
    plane: GroundPlane,
    fps: float,
    camera: str,
    max_speed_mps: float = 2.5,
    window_seconds: float = 1.0,
    min_seconds: float = 0.6,
) -> list[SecurityEvent]:
    """`running`: floor speed over a sliding window exceeded ``max_speed_mps``.

    Speed is **net displacement over the window divided by its duration**, not the sum of
    per-frame steps. The difference matters and the choice is deliberate: foot points
    jitter by tens of centimetres between frames because the box bottom moves with the
    gait, and summing steps turns that jitter into speed. A shopper pacing on the spot
    reads as fast under the summed version and as still under this one, which is the
    correct answer for `running` and the wrong one for "how much did this person move" --
    a different question, and it should get a different function rather than a flag.

    `window_seconds` is therefore doing two jobs, smoothing and defining the event, and
    both are the store's to set. 2.5 m/s is a brisk run and not a measured threshold; it
    is the kind of absolute this project keeps warning about, so it is an argument with a
    default rather than a constant, and the first site clip with a real runner in it is
    what should set it.
    """
    events: list[SecurityEvent] = []
    for track in tracks:
        if len(track.frames) < 2:
            continue
        path = track_ground_path(track, cam, plane)
        frames = np.asarray(track.frames)
        span = max(round(window_seconds * fps), 1)
        speed = np.full(len(frames), np.nan)
        for i in range(len(frames)):
            j = i
            while j + 1 < len(frames) and frames[j + 1] - frames[i] <= span:
                j += 1
            dt = (frames[j] - frames[i]) / fps
            if j == i or dt <= 0:
                continue
            if np.isfinite(path[i]).all() and np.isfinite(path[j]).all():
                speed[i] = float(np.linalg.norm(path[j] - path[i]) / dt)
        fast = np.nan_to_num(speed, nan=0.0) > max_speed_mps
        for i0, i1 in _runs(fast):
            if (frames[i1] - frames[i0] + 1) / fps < min_seconds:
                continue
            events.append(
                SecurityEvent(
                    type="running",
                    camera=camera,
                    frame_start=int(frames[i0]),
                    frame_end=int(frames[i1]),
                    fps=fps,
                    track_ids=(track.track_id,),
                    value=float(np.nanmax(speed[i0 : i1 + 1])),
                    threshold=max_speed_mps,
                    basis=f"net floor displacement over {window_seconds:g}s, in m/s",
                )
            )
    return events


def tailgating_events(
    tracks: list[Track],
    line: CountingLine,
    cam: Camera,
    plane: GroundPlane,
    fps: float,
    camera: str,
    within_seconds: float = 2.0,
) -> list[SecurityEvent]:
    """Two different tracks through one line, same direction, inside ``within_seconds``.

    Built on `line_events` rather than beside it, so a change to what counts as a
    crossing cannot make the two disagree -- which is the failure mode of every pair of
    functions that recompute the same geometry.

    The false positive it cannot avoid is a couple shopping together, and no threshold
    separates them from a tailgater. This is a *reporting* output for a door or a stock
    room, not an alarm, unless the store says otherwise.
    """
    crossings = line_events(tracks, line, cam, plane, fps, camera)
    events: list[SecurityEvent] = []
    for direction in ("in", "out"):
        same = sorted(
            (e for e in crossings if e.extra.get("direction") == direction),
            key=lambda e: e.frame_start,
        )
        for first, second in pairwise(same):
            if first.track_ids == second.track_ids:
                continue
            gap = (second.frame_start - first.frame_start) / fps
            if gap > within_seconds:
                continue
            events.append(
                SecurityEvent(
                    type="tailgating",
                    camera=camera,
                    frame_start=first.frame_start,
                    frame_end=second.frame_end,
                    fps=fps,
                    track_ids=first.track_ids + second.track_ids,
                    zone=line.name,
                    value=gap,
                    threshold=within_seconds,
                    basis="seconds between two tracks crossing one line in one direction",
                    extra={"direction": direction},
                )
            )
    return events


def crowd_events(
    tracks: list[Track],
    cam: Camera,
    plane: GroundPlane,
    fps: float,
    camera: str,
    radius_m: float = 1.5,
    min_people: int = 4,
    min_seconds: float = 5.0,
) -> list[SecurityEvent]:
    """`crowd_forming`: ``min_people`` floor positions within ``radius_m`` of one another.

    A **density proxy, not a clustering algorithm**, and the docstring says so because
    the number it emits looks like a group size. What is computed per frame is the
    largest count of tracks within `radius_m` of any single track -- so a line of people
    two metres apart end to end can report a smaller group than they form, and a tight
    knot reports correctly. Anything better needs a real clustering pass, and nothing yet
    justifies the dependency.

    It counts tracks, so it inherits the fragmentation bias in `occupancy_events`: one
    shopper broken into three is three people standing in the same place.
    """
    per_frame: dict[int, list[np.ndarray]] = {}
    for track in tracks:
        if not track.frames:
            continue
        path = track_ground_path(track, cam, plane)
        for frame, point in zip(track.frames, path, strict=True):
            if np.isfinite(point).all():
                per_frame.setdefault(int(frame), []).append(point)
    if not per_frame:
        return []
    frames = np.arange(min(per_frame), max(per_frame) + 1)
    counts = np.zeros(len(frames), dtype=int)
    for i, frame in enumerate(frames):
        points = per_frame.get(int(frame), [])
        if len(points) < 2:
            counts[i] = len(points)
            continue
        pts = np.stack(points)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        counts[i] = int((d <= radius_m).sum(axis=1).max())
    events = []
    for i0, i1 in _runs(counts >= min_people):
        if (frames[i1] - frames[i0] + 1) / fps < min_seconds:
            continue
        events.append(
            SecurityEvent(
                type="crowd_forming",
                camera=camera,
                frame_start=int(frames[i0]),
                frame_end=int(frames[i1]),
                fps=fps,
                value=float(counts[i0 : i1 + 1].max()),
                threshold=float(min_people),
                basis=f"largest count of tracks within {radius_m:g} m of one track",
            )
        )
    return events


def fall_candidates(
    tracks: list[Track],
    fps: float,
    camera: str,
    aspect_thr: float = 1.0,
    height_ratio: float = 0.6,
    sustained_seconds: float = 1.5,
) -> list[SecurityEvent]:
    """A box that went wide and short and stayed that way. **A miner, not an alarm.**

    The distinction is the entire point of this function, so it is in the type name --
    `fall_candidate`, not `fall` -- and not only in this docstring, because a type name
    survives being copied into a dashboard and a docstring does not.

    What it cannot tell a fall from, in a shop specifically:

    * **a shopper bending to reach a low shelf** -- the single most common posture in
      retail, and geometrically identical from a down-pitched camera;
    * **someone crouching to look at bottom stock**, same;
    * **a person whose legs are occluded by a fixture**, which shortens the box without
      anybody moving. `dwell.py` already records that fixtures are exactly where shoppers
      stand.

    So its precision as an alarm is somewhere near useless and its recall as a *filter*
    is what makes it worth having. Run it over the site clips already on disk, look at
    what it returns, and that answers the question ARCHITECTURE.md insists is
    answered before any behaviour annotation is paid for: **do these events occur on
    these cameras at all.** A tier-1 proxy is the cheapest instrument that can ask.

    The standing baseline is the track's **90th percentile** box height rather than its
    median, and that is forced rather than tuned. A fall lasting longer than the person
    was upright makes the median the *fallen* height, so the rule compares the fall
    against itself and never fires -- and at a median track length of 9-16 frames a fall
    routinely fills its own track, so the median version fails on the central case rather
    than at the edges. The test that found this had six upright frames and twelve down.

    What survives even at the 90th percentile: a track containing no upright frame at all
    has no baseline, because the person was already down before this camera saw them.
    Only the aspect-ratio half of the rule speaks there.
    """
    events = []
    for track in tracks:
        if len(track.boxes) < 2:
            continue
        boxes = np.stack(track.boxes)
        frames = np.asarray(track.frames)
        w = boxes[:, 2] - boxes[:, 0]
        h = np.clip(boxes[:, 3] - boxes[:, 1], 1e-6, None)
        standing_h = float(np.percentile(h, 90))
        down = ((w / h) >= aspect_thr) & (h <= height_ratio * standing_h)
        for i0, i1 in _runs(down):
            seconds = (frames[i1] - frames[i0] + 1) / fps
            if seconds < sustained_seconds:
                continue
            events.append(
                SecurityEvent(
                    type="fall_candidate",
                    camera=camera,
                    frame_start=int(frames[i0]),
                    frame_end=int(frames[i1]),
                    fps=fps,
                    track_ids=(track.track_id,),
                    value=seconds,
                    threshold=sustained_seconds,
                    basis=(
                        "box aspect >= "
                        f"{aspect_thr:g} and height <= {height_ratio:g} x the track's "
                        "90th-percentile height, sustained -- a proxy that cannot "
                        "separate a fall from bending to a low shelf"
                    ),
                    extra={"box_aspect": float((w / h)[i0 : i1 + 1].max())},
                )
            )
    return events


# ------------------------------------------------------- behaviour, tier 2: pose
#
# WHY POSE AND NOT A CLIP-LEVEL ACTION CLASSIFIER, WHICH IS WHAT THE ASK SOUNDS LIKE
#
# Three measurements decide it, all of them already in this repository:
#
#   person box height   244-336 px median      a crop encoder is comfortable
#   head region         31-42 px               anything routed through a face is noise
#   track length        median 9-16 frames     at 5 fps, and this is the one that decides
#
# A clip-level action model wants 16-64 consecutive frames of one person. At 5 fps the
# median track does not contain a whole person for long enough, so such a model would be
# trained on an input this pipeline cannot deliver -- and the fix for that is association
# (ARCHITECTURE.md section 5), not a bigger action model.
#
# **Pose is per-frame, so fragmentation shortens it instead of invalidating it.** Nine
# frames of a track is nine poses, and a fall is a change of torso angle over one or two
# seconds. That is the whole argument, and it is why this tier is buildable now while
# tier 3 is not.
#
# Two further things it gets for free: `person_keypoints_train2017.json` is already on
# disk, so this direction needs no dataset purchase to start; and the pose model is a
# second-stage crop model with its own small ONNX export, which is exactly the boundary
# `stage.py` and ARCHITECTURE.md section 2 describe. It is not a HydraNet head
# and cannot be one -- a keypoint head needs the detection head's boxes, and the `Head`
# protocol's `forward_into(out, feats, size)` has no way to say that.

# COCO's 17, in order. Named rather than indexed at the call site, because 11 and 12
# being the hips is the kind of fact that a reader cannot check and a typo cannot fail.
KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def require_keypoints(tracks: list[Track]) -> None:
    """Refuse a pose event on tracks that carry no pose, naming the missing model.

    An empty `Track.keypoints` is the normal state today -- nothing fills it, because the
    second-stage pose model is not built. Returning no events for that would be
    indistinguishable from "nobody fell", which is the failure this project ranks worst:
    plausible output, no error.
    """
    for track in tracks:
        if len(track.keypoints) != len(track.frames):
            raise NotImplementedError(
                f"track {track.track_id} carries {len(track.keypoints)} keypoint sets for "
                f"{len(track.frames)} observed frames. Pose events need one (17, 3) array "
                "per observed frame, written by the second-stage pose model -- which does "
                "not exist yet. Tier-1 `fall_candidates` is what runs without it."
            )


def _torso(kps: np.ndarray, score_thr: float) -> tuple[float, float, float]:
    """(angle from vertical, hip-to-ankle extent, torso length). Degrees and px, NaN if unseen.

    Image coordinates, y down. The angle is taken from the shoulder-hip vector, which is
    the segment least disturbed by an occluded lower body -- and a lower body behind a
    display table is the normal case here rather than the exception.
    """

    def mid(a: str, b: str) -> np.ndarray:
        pair = kps[[KP[a], KP[b]]]
        if (pair[:, 2] < score_thr).any():
            return np.array([np.nan, np.nan])
        return pair[:, :2].mean(axis=0)

    shoulder = mid("left_shoulder", "right_shoulder")
    hip = mid("left_hip", "right_hip")
    ankle = mid("left_ankle", "right_ankle")
    if not np.isfinite(shoulder).all() or not np.isfinite(hip).all():
        return float("nan"), float("nan"), float("nan")
    dx, dy = hip[0] - shoulder[0], hip[1] - shoulder[1]
    angle = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
    extent = float(abs(ankle[1] - hip[1])) if np.isfinite(ankle).all() else float("nan")
    return angle, extent, float(np.hypot(dx, dy))


def pose_posture_events(
    tracks: list[Track],
    fps: float,
    camera: str,
    fall_angle_deg: float = 55.0,
    crouch_ratio: float = 0.6,
    sustained_seconds: float = 1.0,
    score_thr: float = 0.3,
) -> list[SecurityEvent]:
    """`fall` and `crouch` from keypoints -- and they are one function on purpose.

    Separating the two is the entire reason this tier exists. `fall_candidates` cannot,
    because from a down-pitched camera a fallen person and a shopper reaching a bottom
    shelf produce the same box. The torso angle does separate them: crouching keeps the
    trunk roughly upright over folded legs, falling puts it near horizontal. So the two
    types are computed from one pass over one quantity, which is also what stops a future
    edit from making them disagree about the same frame.

    `crouch` needs the ankles and `fall` does not, which is deliberate -- ankles are the
    first thing a fixture hides, so the safety-relevant type is the one that survives
    occlusion.

    **Both quantities are measured inside a single frame.** The torso angle needs no
    reference at all, and `crouch` compares the hip-to-ankle extent against the torso
    length of the same person in the same frame rather than against that track's own
    history. Two things follow and the second is why the first is not a detail: it is
    scale-free, so a shopper at 3 m and one at 12 m are judged alike with no calibration;
    and it does not go blind when the posture fills the track. A self-baseline over a
    9-16 frame fragment -- the measured median track length -- compares a crouch against a
    crouch and reports nothing, which is precisely the case that matters.

    Every threshold here is a default and none is measured. They are stated as arguments
    for the reason RETAIL.md gives about the 5 m rule: a number that belongs to a
    store belongs in a config, not in weights, and not in a constant either.
    """
    require_keypoints(tracks)
    events: list[SecurityEvent] = []
    for track in tracks:
        if not track.frames:
            continue
        frames = np.asarray(track.frames)
        angles, ratios = [], []
        for kps in track.keypoints:
            angle, extent, torso = _torso(np.asarray(kps, dtype=float), score_thr)
            angles.append(angle)
            ratios.append(extent / torso if np.isfinite(extent) and torso > 0 else np.nan)
        angle_arr = np.asarray(angles)
        ratio = np.asarray(ratios)
        upright = np.nan_to_num(angle_arr, nan=0.0) < fall_angle_deg
        fallen = np.nan_to_num(angle_arr, nan=0.0) >= fall_angle_deg
        low = upright & (np.nan_to_num(ratio, nan=np.inf) <= crouch_ratio)
        for i0, i1 in _runs(fallen):
            events += _posture_event(
                track,
                frames,
                i0,
                i1,
                fps,
                camera,
                sustained_seconds,
                etype="fall",
                value=float(np.nanmax(angle_arr[i0 : i1 + 1])),
                threshold=fall_angle_deg,
                basis="shoulder-to-hip angle from vertical, in degrees, sustained",
            )
        for i0, i1 in _runs(low):
            events += _posture_event(
                track,
                frames,
                i0,
                i1,
                fps,
                camera,
                sustained_seconds,
                etype="crouch",
                value=float(np.nanmin(ratio[i0 : i1 + 1])),
                threshold=crouch_ratio,
                basis="hip-to-ankle extent over torso length in the same frame, trunk upright",
            )
    return events


def _posture_event(
    track: Track,
    frames: np.ndarray,
    i0: int,
    i1: int,
    fps: float,
    camera: str,
    sustained_seconds: float,
    *,
    etype: str,
    value: float,
    threshold: float,
    basis: str,
) -> list[SecurityEvent]:
    """One posture span as an event, or nothing if it was too short to be one.

    A module-level function rather than a closure inside `pose_posture_events`, which is
    where it started: a closure over the loop variables reads fine and is one refactor
    away from being called after the loop has moved on. Returning a list rather than
    appending keeps the caller's `events` the only mutable thing in that function.
    """
    if (frames[i1] - frames[i0] + 1) / fps < sustained_seconds:
        return []
    return [
        SecurityEvent(
            type=etype,
            camera=camera,
            frame_start=int(frames[i0]),
            frame_end=int(frames[i1]),
            fps=fps,
            track_ids=(track.track_id,),
            value=value,
            threshold=threshold,
            basis=basis,
        )
    ]


def reach_to_shelf_events(
    tracks: list[Track],
    frames: list[dict],
    fps: float,
    camera: str,
    fixture_id: int,
    min_seconds: float = 0.6,
    score_thr: float = 0.3,
) -> list[SecurityEvent]:
    """A wrist inside the terrain head's `fixture` region, sustained.

    **This is the one output where the retail model and the security model are the same
    model**, and it is worth saying because "shared trunk" is usually a claim about
    parameters rather than about answers. Here the segmentation head's `fixture` channel
    and the second stage's wrist keypoint are both required to produce one row: neither
    the terrain map nor the pose model can emit it alone.

    `fixture_id` is passed rather than imported because the id is a property of the
    taxonomy a config chose -- 4 under `RETAIL_OBJECTS`, 4 under the six-class surfaces
    scheme, and a hard-coded 4 would be wrong the first time a config reorders its
    classes. `data.terrain_classes.index("fixture")` is where a caller gets it.

    What it is not: a purchase, a pick-up, or an interaction with a specific product. It
    is a wrist over a shelf. MERL Shopping's "reach to shelf" is the nearest labelled
    thing and ARCHITECTURE.md's warning about it applies here too -- those are
    closed grocery shelves and these are open display tables, so do not assume the
    behaviour has the same shape before counting it here.
    """
    require_keypoints(tracks)
    terrain_by_frame = {
        int(f["frame_index"]): f["terrain"] for f in frames if f.get("terrain") is not None
    }
    if not terrain_by_frame:
        raise ValueError(
            "no frame carries a `terrain` map, so a wrist cannot be tested against the "
            "fixture region. A detection-only config has no terrain head; this event "
            "needs one."
        )
    events: list[SecurityEvent] = []
    for track in tracks:
        idx = np.asarray(track.frames)
        touching = np.zeros(len(idx), dtype=bool)
        for i, (frame, kps) in enumerate(zip(track.frames, track.keypoints, strict=True)):
            terrain = terrain_by_frame.get(int(frame))
            if terrain is None:
                continue
            h, w = terrain.shape[-2:]
            kps = np.asarray(kps, dtype=float)
            for wrist in ("left_wrist", "right_wrist"):
                x, y, score = kps[KP[wrist]]
                if score < score_thr:
                    continue
                xi, yi = round(x), round(y)
                if 0 <= xi < w and 0 <= yi < h and int(terrain[yi, xi]) == fixture_id:
                    touching[i] = True
                    break
        for i0, i1 in _runs(touching):
            seconds = (idx[i1] - idx[i0] + 1) / fps
            if seconds < min_seconds:
                continue
            events.append(
                SecurityEvent(
                    type="reach_to_shelf",
                    camera=camera,
                    frame_start=int(idx[i0]),
                    frame_end=int(idx[i1]),
                    fps=fps,
                    track_ids=(track.track_id,),
                    value=seconds,
                    threshold=min_seconds,
                    basis="wrist keypoint inside the terrain head's fixture region",
                )
            )
    return events
