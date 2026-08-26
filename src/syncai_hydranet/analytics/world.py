"""The vector space: the store in metres at one instant, as a payload rather than a
computation.

`stage.py` typed what enters the second stage and said so in its own words: what had no
shape was "what one frame hands the stage". It typed the **pixel** side -- `BoxFrame`
opens with "Boxes are in image pixels, xyxy". The metre side never got the same
treatment, and the cost is already visible in three places that each rebuild it by hand:
`dwell.track_ground_path`, `events/zones.py` and `cli/scene.py` all call
`pixel_to_ground` themselves and each keeps the result in a shape of its own. That is the
same failure `stage.py` records as the reason it exists, one coordinate system later.

---------------------------------------------------------------------------
WHY IT IS HERE AND NOT IN `syncai_bev3d.scene_types`

`PlaneObject` and `DepthObject` already describe most of this shape -- `x_m`, `z_m`,
`width_m`, `height_m`, and `DepthObject` even carries `yaw_rad`. They are the right
design and this file does not restate them for fun. They are simply on the wrong side of
a boundary for this job: they live in `syncai_bev3d`, **the serving path is forbidden
from importing that package** (`tests/test_package_boundaries.py`), and this payload is
produced every frame on the serving path.

The precedent is `geometry/camera_json.py`, which lives on the hydranet side for exactly
this reason: a reader barred from the producer's package still has to read the file. When
a runtime and a commissioning object mean the same thing, the runtime one is the one that
has to be reachable from serving.

---------------------------------------------------------------------------
WHY A PRODUCER SHIPS IN THE SAME FILE

`tests/test_stage_contract.py` records what happened the last time a contract was written
on its own: "zero producers, zero consumers, zero coverage, eight docstring references",
and when it was finally checked against the code, "the contract was wrong, not merely
unadopted". So `world_frame` below is not a convenience -- it is the evidence that the
keys are the keys something can actually fill, and the tests read the type off it.

What this still is not: **adopted**. One producer, no consumer yet. `dwell` and
`events/zones.py` take `Track` and will keep taking `Track` until someone measures that
moving them changes no number. That is a separate change and it is not this one.

---------------------------------------------------------------------------
THREE DECISIONS WORTH THE LINES

**Every key is required and the unfillable ones are `None`.** A `total=False` key makes
"this producer did not supply it" and "this quantity was not measured" the same fact, and
this project separates those everywhere else -- `pixel_to_ground` returns NaN rather than
a large number, `Zone.contains` calls NaN False on purpose, `SecurityEvent` carries the
`value` it measured next to the `threshold` it crossed. A consumer here always finds the
key, and `None` is a stated refusal rather than an absence it has to interpret.

**`space` names which metric frame these coordinates live in**, carried per frame rather
than assumed global -- the same rule, and the same failure, as `BoxFrame.class_names`.
Today the only answer is one camera's own floor frame, because there is no store frame:
nothing in `CameraFile` maps a camera's metres onto a store plan, and `BevGrid` says
plainly that the origin is under the camera. Until that transform exists, two cameras'
`x_m` are two different x. A consumer that fuses them has to read this field to find out
it must not.

**`basis` names the instrument that produced the position**, for the same reason
`SecurityEvent.basis` does. A foot point through the homography and a foot point
extrapolated through an occlusion are not the same claim, and a consumer that cannot tell
them apart will average them.

---------------------------------------------------------------------------
THE LENS, WHICH IS A CORRECTION AND NOT A FEATURE

`camera_json.py` states the contract in its header: the lens "applies to *points* on
their way to the floor (`undistort_points`), never to the pixel-space artefacts", and
`undistort_points` itself says "every runtime consumer of `camera.json` must undo the
lens with the exact same model or the metres drift silently". Nothing on the serving path
does. `dwell.track_ground_path` projects the raw foot point, and the only callers of
`undistort_points` in the tree are two commissioning modules.

This producer undistorts. `track_ground_path` is **left alone deliberately**: correcting
it changes every dwell, path and heatmap number already reported, which is a re-baseline
and belongs in its own commit next to a measurement of what moved. Named here so it is a
known difference rather than a discovered one -- and `dwell.py` already quotes the
journal on what a missing distortion fit costs, having once put 142 people at 1.0-1.2 m
tall.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np

from ..geometry.camera_json import CameraFile
from ..geometry.ground import pixel_to_ground, undistort_points
from .tracker import Track

# How a floor position was arrived at. Grouped by what could be wrong with it, which is
# the judgement a consumer needs and a score cannot carry.
BASES = (
    # The box bottom-centre, undistorted, through the ground plane. The only one this
    # module produces today, and `Track.foot` states its failure: a shopper behind a
    # counter reports the counter's distance.
    "foot_point",
    # Ankle keypoints instead of the box bottom, for exactly that case. Needs the pose
    # head, which is the next one to land (docs/PLAN.md section 2.2).
    "keypoint_ankle",
    # Head or shoulder keypoints plus a height prior, when the ankles are missing or low
    # confidence -- the fixture-occlusion fallback docs/PLAN.md section 2.3 names as an L1 rule.
    "keypoint_prior",
    # The ray at or above the horizon: `pixel_to_ground` declines to turn it into a
    # distance, so `x_m`/`z_m` are NaN and this says which refusal produced them.
    "above_horizon",
)


class WorldObject(TypedDict):
    """One thing on the floor, in metres, at one instant.

    Every key required, unfillable values `None` -- see the module docstring. `width_m`
    is absent rather than `None` because recovering a person's footprint width from one
    camera needs an assumed facing, which is research; `yaw_rad` and `height_m` are
    present-and-`None` because both have a named producer that is a function rather than
    a project (shoulder line for yaw once the pose head lands; foot-point depth against
    the box top for height, the arithmetic `syncai_bev3d.meshes` already does offline).
    """

    track_id: int
    name: str  # the detection class, spelled as `scene_types.PlaneObject` spells it
    x_m: float  # lateral. NaN when `basis` is "above_horizon"
    z_m: float  # forward
    vx_ms: float | None  # None unless the caller supplied a time base -- see `world_frame`
    vz_ms: float | None
    yaw_rad: float | None  # None until the pose head lands
    height_m: float | None  # None until a serving-side producer exists
    observed: bool  # False when this frame's box is the tracker's prediction, not a detection
    basis: str  # one of BASES
    # The detector score behind this frame's box, or None when the producer had none.
    # Present for the reason `events.TrackSupport` states: the 2026-08-26 sweep measured
    # that a track built from 0.15 boxes is not the same claim as one built from 0.6, and
    # a consumer of this payload has exactly the same need as a consumer of an event.
    # None is "not recorded" and never "low"; a coasting frame has no score at all,
    # which is `observed=False` saying it in the other field.
    score: float | None


class WorldFrame(TypedDict):
    """The vector space at one instant, for one camera.

    Deliberately not a `StageFrame` in metres: that payload is keyed by what the network
    emitted and this one by what the world is. The two are related by `camera.json` and
    by nothing else, which is the property that lets a consumer of this file be written
    without knowing which heads a config declared.
    """

    frame_index: int  # the unit of record, for the reason `stage.FrameRef` gives
    time_s: float | None  # PTS seconds when the caller has them; see `world_frame`
    camera_id: str
    space: str  # `camera_floor(<camera_id>)` -- see the module docstring on `space`
    objects: list[WorldObject]


def camera_floor_space(camera_id: str) -> str:
    """The only metric frame that exists today: one camera's floor, origin under it.

    A function rather than an f-string at each call site so that the day a store frame
    exists, every producer and every test names the new one by changing one place.
    """
    return f"camera_floor({camera_id})"


def world_frame(
    tracks: Sequence[Track],
    cam_file: CameraFile,
    frame_index: int,
    *,
    name: str,
    times_s: Mapping[int, float] | None = None,
    fps: float | None = None,
    confirmed_only: bool = True,
    source_size_px: tuple[int, int] | None = None,
) -> WorldFrame:
    """Live tracks -> the vector space, in metres on this camera's floor.

    ``name`` is passed rather than read off the track because `Tracker.update` takes
    "boxes for one class" -- a `Track` has no class field, and inventing one here would
    put a second answer next to the tracker's.

    **The time base.** ``times_s`` maps a frame index to its PTS seconds and is the only
    honest source: docs/PLAN.md section 2.3 measured that every clip in the site corpus writes
    `30/1` into `r_frame_rate` regardless of its true, variable rate, and a speed over a
    nominal fps is how a walk becomes a run alert nobody can explain. ``fps`` is accepted
    for callers that have already made that trade knowingly (`dwell_table` takes one) and
    is used only when ``times_s`` is absent. With neither, velocities are ``None`` --
    not zero, and not a number derived from a rate nobody measured.

    ``source_size_px`` is the frame the boxes are in, when that is not the frame
    `camera.json` was calibrated on -- `clip_tracks.track_clip` returns boxes in the
    decoded stream's pixels and several commissioned cameras are calibrated at half that.
    See `_to_calibrated_pixels`: stating it scales the points, and omitting it is checked
    rather than trusted.

    ``confirmed_only`` keeps the tracker's own gate: `Tracker.min_hits` defaults high
    because "a track confirmed on its first detection turns every one-frame false
    positive into a shopper", and a vector space that shows them has undone that.
    """
    objects: list[WorldObject] = []
    live = [t for t in tracks if t.confirmed or not confirmed_only]
    if live:
        feet = np.stack([t.foot for t in live])
        x, z = _to_ground(feet, cam_file, source_size_px)
        for i, t in enumerate(live):
            vx, vz = _velocity_ms(t, cam_file, times_s, fps, source_size_px)
            objects.append(
                WorldObject(
                    track_id=t.track_id,
                    name=name,
                    x_m=float(x[i]),
                    z_m=float(z[i]),
                    vx_ms=vx,
                    vz_ms=vz,
                    yaw_rad=None,
                    height_m=None,
                    observed=t.age == 0,
                    basis="above_horizon" if not math.isfinite(x[i]) else "foot_point",
                    score=float(t.scores[-1]) if t.scores and t.age == 0 else None,
                )
            )
    return WorldFrame(
        frame_index=frame_index,
        time_s=None if times_s is None else times_s.get(frame_index),
        camera_id=cam_file.camera_id,
        space=camera_floor_space(cam_file.camera_id),
        objects=objects,
    )


def world_frames(
    tracks: Sequence[Track],
    cam_file: CameraFile,
    *,
    name: str,
    times_s: Mapping[int, float] | None = None,
    fps: float | None = None,
    source_size_px: tuple[int, int] | None = None,
) -> list[WorldFrame]:
    """Finished tracks -> one `WorldFrame` per frame they cover, in frame order.

    `world_frame` above is the live shape: called once a frame with whatever the tracker
    currently holds. This is the offline one, for a clip that has already been tracked --
    `clip_tracks.track_clip` returns `Tracker.finished()`, and every consumer of the
    vector space would otherwise have to rebuild this replay itself.

    **A replay cannot show coasting, and says so by construction.** `Track.frames` records
    only the frames a detection was matched in, so every object here has `observed=True`
    and a frame the track coasted through simply has no row for it. That is not the same
    payload the live producer emits and the difference is not cosmetic: a consumer
    counting `observed=False` to find extrapolated positions finds none here, and
    `Visit.observed` against `Visit.span` is where the gap shows up instead.

    Each frame's objects are built from the track truncated at that frame, so a velocity
    is the one that was knowable then rather than the one the whole clip ended with.
    """
    frames = sorted({int(f) for t in tracks for f in t.frames})
    out: list[WorldFrame] = []
    for f in frames:
        live: list[Track] = []
        for t in tracks:
            if f not in t.frames:
                continue
            i = t.frames.index(f)
            live.append(
                Track(
                    track_id=t.track_id,
                    box=np.asarray(t.boxes[i]).copy(),
                    hits=i + 1,
                    age=0,
                    frames=list(t.frames[: i + 1]),
                    boxes=[np.asarray(b).copy() for b in t.boxes[: i + 1]],
                    keypoints=list(t.keypoints[: i + 1]),
                    scores=list(t.scores[: i + 1]),
                    confirmed=t.confirmed,
                )
            )
        out.append(
            world_frame(
                live,
                cam_file,
                f,
                name=name,
                times_s=times_s,
                fps=fps,
                source_size_px=source_size_px,
            )
        )
    return out


def as_rows(frame: WorldFrame) -> list[dict]:
    """Flat dicts for JSONL, a database or a dashboard. No numpy and no NaN survive it.

    NaN becomes `None` because NaN is not JSON: `json.dumps` emits a bare `NaN` token
    that most parsers reject, and the ones that accept it disagree about what it means.
    `None` is the same claim this module already makes everywhere else -- not measured --
    and `basis` still says which refusal produced it.
    """
    rows = []
    for o in frame["objects"]:
        rows.append(
            {
                "frame_index": int(frame["frame_index"]),
                "time_s": frame["time_s"],
                "camera_id": frame["camera_id"],
                "space": frame["space"],
                "track_id": int(o["track_id"]),
                "name": o["name"],
                "x_m": _finite(o["x_m"]),
                "z_m": _finite(o["z_m"]),
                "vx_ms": _finite(o["vx_ms"]),
                "vz_ms": _finite(o["vz_ms"]),
                "yaw_rad": _finite(o["yaw_rad"]),
                "height_m": _finite(o["height_m"]),
                "observed": bool(o["observed"]),
                "basis": o["basis"],
                "score": _finite(o["score"]),
            }
        )
    return rows


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _to_calibrated_pixels(
    points_px: np.ndarray, cam_file: CameraFile, source_size_px: tuple[int, int] | None
) -> np.ndarray:
    """Put pixels into the frame `camera.json`'s intrinsics were fitted on.

    **The failure this exists for, found 2026-08-26 on the first real run.**
    `Taichung-cam01.camera.json` declares `image_size_px` (960, 540) and its clips decode
    at 1920x1080. `clip_tracks.track_clip` returns boxes in *source* pixels, so feeding
    them straight to `pixel_to_ground` used a calibration at half the scale of the boxes
    -- and produced metres. Not an error, not a NaN: three shoppers standing on the shop
    floor came back at x 0.4 to 8.8 m, several metres outside the commissioned walkable
    polygon, with a 38 m walk in 60 seconds and nothing anywhere saying the frame was
    wrong. `camera_json.py`'s header states the contract ("pixels on the raw stream frame
    at `image_size_px`") and nothing enforced it.

    So a caller may state the frame its pixels are in and get them scaled, and a caller
    who states nothing is checked: points far outside the calibrated canvas cannot be a
    lens correction or a clipped box, and are refused with the resolution mismatch named
    rather than projected into confident nonsense.
    """
    pts = np.asarray(points_px, dtype=float).reshape(-1, 2)
    w, h = cam_file.image_size_px
    if source_size_px is not None:
        sw, sh = source_size_px
        if (sw, sh) != (w, h):
            pts = pts * np.array([w / float(sw), h / float(sh)])
        return pts
    if len(pts):
        # 1.5x the canvas: undistortion moves an edge point by pixels and a clipped box
        # sits on the boundary, while a 2x or 4x decode mismatch lands here every time.
        over_x = float(np.nanmax(np.abs(pts[:, 0]))) / max(w, 1)
        over_y = float(np.nanmax(np.abs(pts[:, 1]))) / max(h, 1)
        if max(over_x, over_y) > 1.5:
            raise ValueError(
                f"{cam_file.camera_id}: a point at "
                f"({np.nanmax(pts[:, 0]):.0f}, {np.nanmax(pts[:, 1]):.0f}) px is outside "
                f"the {w}x{h} frame this camera.json was calibrated on, by more than a "
                "lens correction or a clipped box can explain. The usual cause is boxes "
                "in the decoded stream's pixels against intrinsics fitted on a smaller "
                "frame -- pass `source_size_px` to say which frame these pixels are in. "
                "Projecting them anyway would return metres, and they would be wrong "
                "without being NaN."
            )
    return pts


def _to_ground(
    points_px: np.ndarray,
    cam_file: CameraFile,
    source_size_px: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(N,2) raw-frame pixels -> floor metres, undoing the lens first if there is one.

    The order is the contract `camera_json.py` states and it is not interchangeable: the
    lens model is defined on raw pixels, so undistorting after projecting corrects the
    wrong quantity.
    """
    pts = _to_calibrated_pixels(points_px, cam_file, source_size_px)
    lens = cam_file.lens
    if lens is not None:
        pts = undistort_points(pts, lens.k1, lens.centre_px, lens.radius_px)
    return pixel_to_ground(pts[:, 0], pts[:, 1], cam_file.camera, cam_file.plane)


def _velocity_ms(
    track: Track,
    cam_file: CameraFile,
    times_s: Mapping[int, float] | None,
    fps: float | None,
    source_size_px: tuple[int, int] | None = None,
) -> tuple[float | None, float | None]:
    """Floor velocity from the last two **observed** boxes, or (None, None).

    Observed and not predicted: a coasting track's current box moves at whatever
    `Track.velocity` last held, so differencing it would report the tracker's own
    extrapolation back as a measurement of the shopper.
    """
    if len(track.frames) < 2 or len(track.boxes) < 2:
        return None, None
    f1, f0 = track.frames[-1], track.frames[-2]
    dt = _dt_seconds(f0, f1, times_s, fps)
    if dt is None or dt <= 0:
        return None, None
    feet = np.stack([[(b[0] + b[2]) / 2, b[3]] for b in (track.boxes[-2], track.boxes[-1])])
    x, z = _to_ground(feet, cam_file, source_size_px)
    if not (math.isfinite(x[0]) and math.isfinite(x[1])):
        return None, None
    return float((x[1] - x[0]) / dt), float((z[1] - z[0]) / dt)


def _dt_seconds(
    f0: int, f1: int, times_s: Mapping[int, float] | None, fps: float | None
) -> float | None:
    if times_s is not None and f0 in times_s and f1 in times_s:
        return float(times_s[f1] - times_s[f0])
    if fps:
        return (f1 - f0) / float(fps)
    return None
