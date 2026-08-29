"""Behaviour that a floor path and a clock can settle, with no new model and no new data.

This is the second half of tier 1 in `TIERS`, kept apart from `zones.py` because the
question is different: `zones.py` asks where a person is, this asks what they are doing,
and the ask usually arrives attached to an annotation budget. A good share of it is
geometry and costs nothing.

The unit is metres per second and not pixels per frame, and that is the whole reason
these transfer between cameras. A pixel speed threshold tuned on Kaohsiung-cam08 is a
statement about that lens and that mount.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from ...geometry.ground import Camera, GroundPlane
from ..dwell import track_ground_path
from ..tracker import Track
from ._geometry import _runs
from ._types import CountingLine, SecurityEvent

# `tailgating_events` is a second reading of the crossings `line_events` already
# produced, rather than a second traversal of the paths -- so it asks that function
# rather than restating the crossing test. That is the one edge from this tier to
# `zones`, and it is one-way: nothing in `zones` reads behaviour.
from .zones import line_events

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
