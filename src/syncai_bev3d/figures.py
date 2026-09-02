"""What a commissioning render needs to know about drawing one camera.

Colours, the velocity and smoothing constants those renders were tuned to, and the
handful of `camera.json`-derived geometry calls that turn a person box into a floor
position. **They lived inside `demo_video.py` until 2026-09-02**, which made a CLI script
the de facto library for every other render -- `heads_video` imported thirteen names from
it --
and the split had started to cost: staff colouring existed only in one front-end,
frame-parallel rendering only in the other, and the sidecar `demo_gif` audits against was
written by only one of them, so a verdict could describe a different render than the
figure beside it.

The reasoning behind every constant here was measured on this fleet's own tracks and is
kept with it; none of these numbers is a default anybody chose by feel.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import pixel_to_ground, undistort_points

TRACK_COLORS = [
    (255, 99, 71), (65, 180, 255), (255, 200, 60), (120, 220, 120),
    (220, 120, 255), (255, 150, 100), (100, 230, 210), (250, 100, 160),
]  # fmt: skip
# **Two colours, not three: staff blue, everything else green.** The user's rule, and it
# is a product decision rather than a measurement one, so it is stated here rather than
# argued with. What it costs is worth writing down: a track shorter than
# `staff.MIN_OBSERVATIONS` has no verdict, and this draws it as a customer, so a member of
# staff who crosses the frame in under about a second is green. The alternative -- a third
# grey state -- put a colour on screen that a viewer had to be told how to read, and every
# shopper arrived grey for their first second, which read as a defect.
STAFF_COLOR = (70, 150, 255)
CUSTOMER_COLOR = (105, 215, 120)
# On the staff classifier those colours report for: its torso band is 0.18-0.55 of a
# box and `face_blur.blur_region` covers the top 45% plus padding, so the face blur
# would land squarely on the pixels the classifier reads. That is why features are
# taken from the source frame BEFORE either blur instrument runs -- nothing about the
# order is incidental, and reversing it would silently classify blurred shirts. See
# `_torso_crop` below.
PLACE_MARGIN_M = 2.0  # beyond the commissioned walkable zone a floor position is a guess
# Velocity is measured over a WINDOW, not between adjacent frames, and only asserted
# above a floor. Both numbers come from this camera's own tracks (1,607 steps):
# frame-to-frame heading below 0.1 m/s turns a mean of 92.5 deg per step with 51% of
# turns past 90 -- that is uniform noise, an arrow drawn from it points nowhere. A
# 1.0 s window brings the median turn to 16.9 deg; 2.0 s gives 16.3 and only adds lag.
# The median shopper here moves 0.22 m/s, so most of the time there is no vector to
# draw and the honest thing is to draw none.
VEL_WINDOW_S = 1.0
# Position EMA, chosen against this camera's own tracks rather than by feel. A standing
# shopper's floor point moves a median 3.2 cm per frame with nothing but noise driving
# it -- 0.16 m/s against a median real speed of 0.22, which is why the raw heading was
# indistinguishable from random. Measured over 9 tracks: EMA 0.35 halves the still
# jitter (3.2 -> 1.7 cm) and keeps 92% of the path length on fast segments; 0.25 gets to
# 1.4 cm but cuts 11% of the corners; a trailing median filter scores *above* 100% path
# because it holds still and then jumps, which is a worse artefact than the jitter.
POS_EMA = 0.35
FALLBACK_STATURE_M = 1.70  # only until a track has been measured STATURE_MIN_N times
STATURE_MIN_N = 3
PLUMB_W_M = 0.035  # the drop line under a figure: thin enough not to read as an object
STATURE_RANGE_M = (1.2, 2.6)  # outside this the box top is not a head top
VEL_FLOOR_MS = 0.3
VEL_SECONDS_SHOWN = 1.0  # arrow length IS one second of travel, so it reads in metres


def _torso_crop(frame: np.ndarray, box) -> np.ndarray:
    """The person's pixels for one box, clipped to the frame.

    Clipped rather than trusted: a tracked box can predict past the frame edge on the
    coast frames, and numpy would return an empty or short slice without complaint --
    which `torso_stats` turns into nine zeros, a perfectly usable-looking feature vector
    for a person who is half out of shot. One pixel of margin is kept in each direction so
    the band is always taken over something.
    """
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box)
    ix0 = int(np.clip(x0, 0, w - 2))
    iy0 = int(np.clip(y0, 0, h - 2))
    ix1 = int(np.clip(x1, ix0 + 1, w))
    iy1 = int(np.clip(y1, iy0 + 1, h))
    return frame[iy0:iy1, ix0:ix1]


def stature_m(x_m: float, z_m: float, v_top_px: float, cf: CameraFile) -> float:
    """How tall the person standing at (x, z) must be for their box top to land on v_top.

    The figure was drawn at a hard-coded 1.70 m for everyone, which is the one number in
    the panel that was never measured. A head top at level-frame height `plane.height - h`
    projects linearly in h, so this is one linear solve, not a search. Round-trips
    exactly on synthetic people at nine positions and three heights.

    NaN when the ray is degenerate. The caller decides what an implausible answer means:
    here it means the box top was not a head top -- a merged box, a truncated person --
    and the sample is dropped rather than averaged in.
    """
    rot = cf.plane.rotation
    a = rot @ np.array([x_m, cf.plane.height, z_m])
    b = rot @ np.array([0.0, 1.0, 0.0])
    k = (v_top_px - cf.camera.cy) / cf.camera.fy
    den = b[1] - k * b[2]
    if abs(den) < 1e-9:
        return float("nan")
    return float((a[1] - k * a[2]) / den)


def facing_wedge(radius_m: float = 0.42, half_deg: float = 26.0, sides: int = 7):
    """A compass wedge on the position disc, pointing +z like `human()` does.

    The figure is rotated by its measured heading and this is what makes that visible.
    Measured on this camera's own geometry: turning the mesh a full 180 degrees changes
    0.14% of its pixels, and 90 degrees changes 1.4% -- `human()` is very nearly
    rotationally symmetric at 1.70 m seen from 7 m, and its own docstring says the foot
    is the only feature carrying a facing. A heading that cannot be seen is a heading
    that was not drawn.

    It sits inside the velocity arrow's tail radius so the two never overlap, and it is
    drawn whenever a heading is known -- which includes a stopped shopper, where the
    arrow is deliberately absent.
    """
    a = math.radians(half_deg)
    pts = [(0.0, 0.0)]
    for i in range(sides + 1):
        t = -a + (2 * a) * i / sides
        pts.append((radius_m * math.sin(t), radius_m * math.cos(t)))
    return pts


def velocity_arrow(length_m: float, width_m: float = 0.16, start_m: float = 0.45):
    """Floor arrow pointing +z, starting clear of the figure -- `human()` faces +z too.

    Returned as a footprint for `extrude`, because that is how every other object in
    this scene is built and an arrow that is a decal on the floor cannot be mistaken
    for a measured piece of furniture. `start_m` is the position disc's radius: an
    arrow drawn from the figure's own feet is under the figure and invisible from the
    one fixed viewpoint, which is the mistake this argument exists to correct.
    """
    length = max(float(length_m), 0.30)
    head = min(0.26, length * 0.5)
    tip = start_m + length
    shoulder = tip - head
    w, hw = width_m / 2, width_m
    return [
        (-w, start_m), (w, start_m), (w, shoulder), (hw, shoulder),
        (0.0, tip), (-hw, shoulder), (-w, shoulder),
    ]  # fmt: skip


def _point_in_poly(px: float, py: float, poly) -> bool:
    """Ray-cast containment. Shared so pixel zones and metre zones agree on `inside`."""
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def point_in_walkable(cf: CameraFile, x_m: float, z_m: float) -> bool:
    """Is this floor position inside a zone the commissioning actually walked?

    Recorded per placement rather than enforced: a tracked shopper standing outside the
    walkable polygon is evidence about the polygon, not a reason to hide the shopper.
    """
    return any(_point_in_poly(x_m, z_m, z.points_m) for z in cf.zones if z.kind == "walkable")


def in_fp_zone(cf: CameraFile, cx: float, cy: float) -> bool:
    """Ray-cast point-in-polygon, not the polygon's bounding box.

    Today's polygons are axis-aligned 64 px grid cells, for which a bbox test is exactly
    equivalent -- so this changes no current result. It is here because the zone tool
    will draw arbitrary polygons, and a bbox test would then quietly veto detections
    outside the shape the operator drew.
    """
    return any(_point_in_poly(cx, cy, poly) for poly in cf.false_positive_polygons_px)


def walkable_bounds(cf: CameraFile) -> tuple[float, float, float, float]:
    """(x_min, x_max, z_min, z_max) of the commissioned walkable zone, plus a margin.

    Replaces a hard-coded `0 < z < 14, |x| < 12`, which was a guess wide enough to
    admit positions this camera was never commissioned for. On Taichung-cam10 the
    walkable zone is x[-5.8, 3.7] z[0.55, 10.3]; the old constants dropped nothing at
    all over 900 frames, so this tightens a gate rather than opening one.
    """
    pts = [np.asarray(z.points_m, float) for z in cf.zones if z.kind == "walkable"]
    if not pts:
        return (-12.0, 12.0, 0.0, 14.0)
    p = np.vstack(pts)
    m = PLACE_MARGIN_M
    return (
        float(p[:, 0].min()) - m,
        float(p[:, 0].max()) + m,
        max(0.0, float(p[:, 1].min()) - m),
        float(p[:, 1].max()) + m,
    )


class TrackState(NamedTuple):
    """One track's 3D-panel state for one frame, after everything sequential is applied.

    ``h_one`` is the single-frame stature reading (NaN when the box top projects
    outside `STATURE_RANGE_M`) and ``verdict`` the staff call (`None` without a model
    or below `staff.MIN_OBSERVATIONS`); both ride along for `demo_video`'s per-frame
    record and cost `heads_video` nothing.
    """

    track_id: int
    box: np.ndarray
    col: tuple
    sm: tuple
    x_m: float
    z_m: float
    stature: float
    heading: float | None
    speed: float
    h_one: float
    verdict: bool | None


def track_colour(track, verdict_of=None) -> tuple:
    """Staff blue, everyone else green -- and per-track-id colours without a model.

    The cost is stated on STAFF_COLOR above: a track with too few observations has no
    verdict and is drawn as a customer. ``verdict_of`` is `analytics.staff.track_staff`,
    passed in by the caller rather than imported: the staff classifier is student-side
    machinery and `test_package_boundaries` names the only hydranet edges this package
    may import. `None` means no staff model -- colour by track id.
    """
    if verdict_of is None:
        return TRACK_COLORS[track.track_id % len(TRACK_COLORS)]
    return STAFF_COLOR if verdict_of(track) is True else CUSTOMER_COLOR


def track_states(
    tracks, n, cf, state, vel_window, metre_scale, fps, bounds, verdict_of=None
) -> list[TrackState]:
    """The sequential half of the 3D panel: everything that depends on the frames before.

    Split out of the drawing because a chunked render has to **replay** it. A worker that
    draws frames 600-750 must arrive at frame 600 holding the tracker, the smoothed floor
    positions and the stature medians a single-process render would hold there, and the
    only way to be sure of that is for both to run this function rather than two copies of
    it. Two copies agree with themselves and drift from each other, and the drift would
    surface as a track changing colour or jumping half a metre at a chunk boundary --
    visible in the figure, and attributable to nothing. It lived in `heads_video` while
    `demo_video` carried an inline copy, until 2026-09-02; the copies had already diverged
    once (this file's constants against stale literals).

    `state` is mutated: it carries the four dicts that outlive the frame, plus
    ``n_skipped``, the count of track-frames whose floor point was non-finite or outside
    ``bounds`` (`demo_video` reports it as ``n_outside``).
    """
    smoothed, statures, history, last_heading = (
        state["smoothed"],
        state["statures"],
        state["history"],
        state["last_heading"],
    )
    state.setdefault("n_skipped", 0)
    x_lo, x_hi, z_lo, z_hi = bounds
    out = []
    for t in tracks:
        mid_x = (t.box[0] + t.box[2]) / 2 / 2.0
        pts = np.array([[mid_x, t.box[3] / 2.0], [mid_x, t.box[1] / 2.0]])
        if cf.lens is not None:
            pts = undistort_points(pts, cf.lens.k1, cf.lens.centre_px, cf.lens.radius_px)
        fx, fz = pixel_to_ground(pts[:1, 0], pts[:1, 1], cf.camera, cf.plane)
        if not (np.isfinite(fx[0]) and np.isfinite(fz[0])):
            state["n_skipped"] += 1
            continue
        raw = (float(fx[0]), float(fz[0]))
        prev_s = smoothed.get(t.track_id)
        sm = (
            raw
            if prev_s is None
            else (
                POS_EMA * raw[0] + (1 - POS_EMA) * prev_s[0],
                POS_EMA * raw[1] + (1 - POS_EMA) * prev_s[1],
            )
        )
        smoothed[t.track_id] = sm
        if not (x_lo <= sm[0] <= x_hi and z_lo <= sm[1] <= z_hi):
            state["n_skipped"] += 1
            continue
        h_one = stature_m(sm[0], sm[1], float(pts[1, 1]), cf)
        if STATURE_RANGE_M[0] <= h_one <= STATURE_RANGE_M[1]:
            statures.setdefault(t.track_id, []).append(h_one)
        seen_h = statures.get(t.track_id, [])
        stature = (
            float(np.median(seen_h)) if len(seen_h) >= STATURE_MIN_N else FALLBACK_STATURE_M
        ) * metre_scale
        x_m, z_m = sm[0] * metre_scale, sm[1] * metre_scale
        hist = history.setdefault(t.track_id, {})
        hist[n] = (x_m, z_m)
        speed = 0.0
        prev = hist.get(n - vel_window)
        if prev is not None:
            dx, dz = x_m - prev[0], z_m - prev[1]
            speed = math.hypot(dx, dz) / (vel_window / fps)
            if speed >= VEL_FLOOR_MS:
                last_heading[t.track_id] = math.atan2(dx, dz)
        verdict = None if verdict_of is None else verdict_of(t)
        out.append(
            TrackState(
                t.track_id,
                t.box,
                track_colour(t, verdict_of),
                sm,
                x_m,
                z_m,
                stature,
                last_heading.get(t.track_id),
                speed,
                h_one,
                verdict,
            )
        )
    return out


KP_MIN_CONF = 0.2
# The skeleton edges a lifted figure is judged by, and the longest any of them may be.
# Measured 2026-09-02 on both README cameras' gif windows: coherent lifts top out at
# 1.46 m (stretched, but reading as a person) while an occluded person's lift jumps
# straight past 3 m -- 20 of 321 on Kaohsiung-cam04, one every few frames, a figure of
# metre-long tubes sprawled over the desk. The two populations do not touch, so 1.5 is
# a gap, not a tuning knob. Keypoint confidence CANNOT stand in for this check: the
# exploded lifts' minimum limb confidence (p50 0.649) is HIGHER than the clean ones'
# (0.481) -- the pose head is confidently wrong about a body a counter is hiding.
LIFT_EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (0, 5), (0, 6),
)  # fmt: skip
MAX_BONE_M = 1.5


def box_iou(box, boxes):
    """IoU of one xyxy box against many. The tracker returns a box, not the index of the
    detection it came from, so the keypoints are found by overlap."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not len(boxes):
        return np.zeros(0)
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def lift_fronto_parallel(kp_src, x_m, z_m, cf: CameraFile):
    """2D keypoints -> (17, 3) joints in scene metres, or None if the frame cannot carry it.

    Every joint is put on the vertical plane through the track's floor point that is
    perpendicular to the camera's horizontal view direction there. That is the cheapest
    lift there is: it fixes the depth of all seventeen at the feet's range, so the heights
    and the limb directions come out right and the depth content of the pose is discarded.

    **It is chosen here over the solvers that fix the bone lengths because those fold.**
    Both a per-bone depth solve and a constrained least-squares over the whole skeleton
    satisfied every bone length and put a 1.95 m person's head at 0.73 m -- the figure
    sinks to the floor and hinges at the hips while the internal distances stay correct
    (PLAN 7c.30). This one is wrong in a way that stays readable: limbs come out 0.67-1.15x
    their true length and a foot that is forward reads as a foot that is raised.
    """
    kp = np.asarray(kp_src, dtype=float)
    if kp.shape != (17, 3) or not np.isfinite(kp).all():
        return None
    if float(np.median(kp[:, 2])) < KP_MIN_CONF:
        return None
    half = kp[:, :2] / 2.0  # camera.json is calibrated at half res
    if cf.lens is not None:
        half = undistort_points(half, cf.lens.k1, cf.lens.centre_px, cf.lens.radius_px)
    cam = cf.camera
    rays = np.stack(
        [(half[:, 0] - cam.cx) / cam.fx, (half[:, 1] - cam.cy) / cam.fy, np.ones(len(half))],
        axis=-1,
    )
    d = rays @ cf.plane.rotation
    rng = float(np.hypot(x_m, z_m))
    denom = x_m * d[:, 0] + z_m * d[:, 2]
    if rng < 1e-6 or np.any(np.abs(denom) < 1e-9):
        return None
    t = rng * rng / denom
    j = np.stack([d[:, 0] * t, cf.plane.height - d[:, 1] * t, d[:, 2] * t], axis=-1)
    if not np.isfinite(j).all():
        return None
    # The feet are on the floor whatever the lift says: pin the lower ankle and carry the
    # rest with it, so a figure never floats or sinks even when the ankle keypoint is poor.
    j[:, 1] -= min(j[15, 1], j[16, 1])
    if not (0.5 < float(j[:, 1].max()) < 2.6):
        return None  # a figure this tall or this short is a fault
    if any(float(np.linalg.norm(j[a] - j[b])) > MAX_BONE_M for a, b in LIFT_EDGES):
        return None  # an exploded skeleton, not a long-limbed person; see MAX_BONE_M
    return j


def content_crop(view, meshes, img_size, aspect, pad=26.0):
    """Fixed crop onto the projected scene, so the panel is room and not dead space.

    Computed once from the static scene plus the walkable zone -- the region a figure
    can legally occupy -- and then reused for every frame, because a crop recomputed
    per frame would make the room swim behind the people.
    """
    us, vs = [], []
    for verts in meshes:
        uv, depth = view.project_points(np.asarray(verts, float))
        keep = depth > 0
        if keep.any():
            us.append(uv[keep, 0])
            vs.append(uv[keep, 1])
    if not us:
        return (0, 0, img_size[0], img_size[1])
    u, v = np.concatenate(us), np.concatenate(vs)
    x0, x1 = float(u.min()) - pad, float(u.max()) + pad
    y0, y1 = float(v.min()) - pad, float(v.max()) + pad
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:  # too tall: widen
        need = h * aspect
        cx = (x0 + x1) / 2
        x0, x1 = cx - need / 2, cx + need / 2
    else:  # too wide: heighten
        need = w / aspect
        cy = (y0 + y1) / 2
        y0, y1 = cy - need / 2, cy + need / 2
    w_px, h_px = img_size
    return (max(0, int(x0)), max(0, int(y0)), min(w_px, int(x1)), min(h_px, int(y1)))
