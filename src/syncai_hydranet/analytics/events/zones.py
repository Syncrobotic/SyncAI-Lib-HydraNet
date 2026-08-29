"""Events a polygon or a line on the floor decides: intrusion, occupancy, stock.

The first of the three instrument tiers `TIERS` names, and the cheapest: a floor
position in metres and a clock. Every threshold is an argument, and every zone is a
polygon in metres on the ground plane rather than in pixels -- see the package
docstring for why that is not a modelling choice that was skipped.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...geometry.ground import Camera, GroundPlane, pixel_to_ground
from ..dwell import track_ground_path
from ..stage import BoxFrame
from ..tracker import Track
from ._geometry import _cross2, _runs, _segments_cross
from ._types import CountingLine, SecurityEvent, Zone

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
    frames: Sequence[BoxFrame],
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

    **A vocabulary that cannot express `classes` is refused, not counted as zero.**
    Reading names off the frame makes the mismatch visible; it did not make it *loud*.
    Matching by name against a vocabulary holding neither `boxed_stock` nor `device`
    leaves `wanted` empty, every frame counts 0, and the return value is a perfectly
    well-formed mapping that a consumer reads as "no merchandise moved" -- so the
    stock-removal alarm stops firing and nothing anywhere says why. That is not
    hypothetical: `tools/site30k/box_pass.py` had to map a `person`/`product` campaign
    taxonomy back onto `retail_security` specifically to avoid it, and its comment names
    this function and this line. PLAN section 7.1 carries the same risk for a `stack`
    class arriving in the vocabulary.

    An empty `frames` is not this failure and returns `{}`: a clip with no frames is a
    decode question, and it is `data.video.frames` that raises on one.
    """
    out: dict[int, int] = {}
    for frame in frames:
        names = frame["class_names"]
        wanted = {i for i, n in enumerate(names) if n in classes}
        if not wanted:
            raise ValueError(
                f"none of {list(classes)} is in this frame's detection vocabulary "
                f"({list(names)}), so merchandise cannot be counted at all. Returning 0 "
                "per frame would read as 'no stock moved' and silence the stock-removal "
                "alarm. Pass `classes=` naming what this vocabulary actually calls "
                "merchandise, or export an engine that carries these channels."
            )
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
        x, z = pixel_to_ground(foot_u, foot_v, cam, plane)
        out[int(frame["frame_index"])] = int(zone.contains(np.stack([x, z], axis=-1)).sum())
    return out
