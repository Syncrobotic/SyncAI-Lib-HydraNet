"""Procedural meshes for the scene renderer: a human, a fixture, and a ground disc.

Generated rather than downloaded. A mesh file in the repository is a binary blob with a
licence to track, a CDN or an LFS pointer to serve it from, and no way to ask it for a
1.62 m person instead of a 1.70 m one. Everything here is a function of its dimensions,
which is what makes it usable for a scene where the dimensions are the output.

**On what a human-shaped mesh does and does not claim.** `bev_page.py` (removed in
`500cdd2`; readable at `git show 500cdd2^:scripts/bev_page.py`) drew objects as
wireframe cuboids and argued, correctly, that a solid asset asserts a shape nobody
measured. That argument is about *shape*. It is not an argument for a crude mesh, because
the thing actually uncertain in a retail scene is **position** -- `analytics/dwell.py`
records the mechanism: a shopper behind a counter has their feet occluded, the box bottom
lands on the counter edge, and the projected position is metres out.

Crudeness is a bad way to express that. It degrades the picture everywhere in order to
hint at an error that lives in one place, and a viewer cannot read metres off a capsule
any better than off a person. So the shape here is a person, and the uncertainty gets its
own visual channel: `ground_disc` draws the position's error radius on the floor, where
it is in the same units as the thing it qualifies. Honest *and* legible, rather than one
bought with the other.

What the human mesh still does not claim: it is one build at one height, standing. It is
a **stand-in for a person**, in the way Tesla's car models are stand-ins for cars -- the
network said "person, here, this tall", and the mesh says exactly that back. Nothing about
its pose, width, or which way it is facing is measured unless a caller passes it in.

Coordinates are the scene payload's: **x right, y up, z forward, metres**, origin at the
feet. `bev.scene` emits `x_m`/`z_m` on the floor, so placing one is a translation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

Mesh = tuple[np.ndarray, np.ndarray]  # (vertices [N,3] float, faces [M,3] int)


def _tube(p0, p1, r0: float, r1: float, sides: int = 12) -> Mesh:
    """A tapered tube between two points. The one primitive a skeleton needs.

    Radii at each end rather than one: a thigh is not a cylinder, and interpolating the
    radius costs nothing and removes the sausage look that makes low-poly figures read as
    toys rather than as people.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        raise ValueError("tube endpoints coincide")
    axis = axis / length
    # Any vector not parallel to the axis gives a usable basis; picking the smallest
    # component to cross against is what keeps it well conditioned near the poles.
    tmp = np.zeros(3)
    tmp[int(np.argmin(np.abs(axis)))] = 1.0
    u = np.cross(axis, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    verts = np.vstack([p0 + r0 * ring, p1 + r1 * ring])

    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, sides + j])
        faces.append([i, sides + j, sides + i])
    return verts, np.asarray(faces, int)


def _sphere(centre, radius: float, rings: int = 8, sides: int = 12) -> Mesh:
    centre = np.asarray(centre, float)
    verts = [centre + [0, radius, 0]]  # noqa: RUF005 - ndarray add, not concat
    for i in range(1, rings):
        phi = math.pi * i / rings
        y, r = math.cos(phi) * radius, math.sin(phi) * radius
        for k in range(sides):
            th = 2 * math.pi * k / sides
            verts.append(centre + [r * math.cos(th), y, r * math.sin(th)])  # noqa: RUF005
    verts.append(centre + [0, -radius, 0])  # noqa: RUF005 - ndarray add
    verts = np.asarray(verts, float)

    faces = []
    for k in range(sides):
        faces.append([0, 1 + k, 1 + (k + 1) % sides])
    for i in range(rings - 2):
        a, b = 1 + i * sides, 1 + (i + 1) * sides
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append([a + k, b + k, b + k2])
            faces.append([a + k, b + k2, a + k2])
    last = len(verts) - 1
    base = 1 + (rings - 2) * sides
    for k in range(sides):
        faces.append([last, base + (k + 1) % sides, base + k])
    return verts, np.asarray(faces, int)


def _merge(*meshes: Mesh) -> Mesh:
    verts, faces, offset = [], [], 0
    for v, f in meshes:
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.vstack(verts), np.vstack(faces)


# Proportions as fractions of standing height, from the standard 7.5-head canon. They are
# constants rather than parameters because a caller who wants a different *build* wants a
# different mesh, not eleven keyword arguments they have to keep consistent with each other.
_P = {
    "head_r": 0.0665,  # 1/7.5 of height is head length; radius is a bit under half of it
    "neck_y": 0.870,
    "shoulder_y": 0.818,
    "shoulder_x": 0.130,  # 0.44 m across at 1.70 m, which is the anthropometric figure
    "hip_y": 0.530,
    "hip_x": 0.052,
    "knee_y": 0.285,
    "ankle_y": 0.039,
    "elbow_y": 0.630,
    "wrist_y": 0.455,
    "arm_x": 0.150,
    "torso_r": 0.088,
    "waist_r": 0.070,
    "thigh_r": 0.050,
    "calf_r": 0.035,
    "upper_arm_r": 0.032,
    "forearm_r": 0.026,
    "foot_z": 0.090,  # toe reach ahead of the ankle; a footless figure hovers
    "foot_r": 0.032,
}


@dataclass(frozen=True)
class Placement:
    """Where to put a mesh on the floor, in the scene payload's own units."""

    x_m: float = 0.0
    z_m: float = 0.0
    heading_rad: float | None = None  # None when nothing measured a facing


def human(height_m: float = 1.70, sides: int = 10) -> Mesh:
    """A standing human figure of the given height, feet at y = 0.

    1.70 m is the same assumption `scripts/fit_camera_from_people.py` fitted camera pose
    against -- the script went in `500cdd2`, readable at
    `git show 500cdd2^:scripts/fit_camera_from_people.py` -- and it was an assumption
    there too. Passing a height through rather than
    hardcoding one means a scene that later measures stature can render it.
    """
    if height_m <= 0:
        raise ValueError(f"height_m must be positive, got {height_m}")
    h = float(height_m)
    p = {k: v * h for k, v in _P.items()}

    parts = [
        _sphere([0, p["neck_y"] + p["head_r"] * 1.15, 0], p["head_r"], sides=sides),
        _tube(
            [0, p["neck_y"], 0],
            [0, p["shoulder_y"], 0],
            p["waist_r"] * 0.55,
            p["torso_r"] * 0.7,
            sides,
        ),
        _tube([0, p["shoulder_y"], 0], [0, p["hip_y"], 0], p["torso_r"], p["waist_r"], sides),
    ]
    for side in (-1, 1):
        parts += [
            _tube(
                [side * p["shoulder_x"], p["shoulder_y"], 0],
                [side * p["arm_x"], p["elbow_y"], 0],
                p["upper_arm_r"],
                p["upper_arm_r"] * 0.82,
                sides,
            ),
            _tube(
                [side * p["arm_x"], p["elbow_y"], 0],
                [side * p["arm_x"], p["wrist_y"], 0],
                p["forearm_r"],
                p["forearm_r"] * 0.8,
                sides,
            ),
            _tube(
                [side * p["hip_x"], p["hip_y"], 0],
                [side * p["hip_x"], p["knee_y"], 0],
                p["thigh_r"],
                p["calf_r"] * 1.15,
                sides,
            ),
            _tube(
                [side * p["hip_x"], p["knee_y"], 0],
                [side * p["hip_x"], p["ankle_y"], 0],
                p["calf_r"],
                p["calf_r"] * 0.62,
                sides,
            ),
            # The foot is why the figure stands on the floor rather than hovering over it.
            # It also gives the silhouette a facing, which matters once `heading_rad` is
            # passed: a legs-only figure looks identical from every angle, so a measured
            # heading would render as no change at all.
            _tube(
                [side * p["hip_x"], p["foot_r"], 0],
                [side * p["hip_x"], p["foot_r"] * 0.8, p["foot_z"]],
                p["foot_r"],
                p["foot_r"] * 0.75,
                sides,
            ),
        ]
    verts, faces = _merge(*parts)
    # The canon's landmarks put the crown of the head at 1.013 of standing height, so a
    # figure built straight from them comes out 1.3% too tall -- `human(1.70)` returned
    # 1.722 m. Small, plausible, and wrong in the one quantity this module exists to be
    # trusted on. Normalising the built mesh is better than hand-tuning `head_r` until the
    # number looks right, because it stays correct if any landmark is ever revised.
    #
    # Both ends, not just the top. The foot is a tube swept along a slanted axis, so its
    # lowest ring sits a fraction of a millimetre off zero -- scaling by the crown alone
    # left the figure 0.14 mm into the floor and the *height* still wrong by that much.
    # A tenth of a millimetre does not matter; a height that is not the height does,
    # because the next thing to use this module will be a renderer printing metres.
    lo = float(verts[:, 1].min())
    verts[:, 1] = (verts[:, 1] - lo) * (h / (float(verts[:, 1].max()) - lo))
    return verts, faces


# COCO's 17 keypoints, in the order every pose head in this project emits them.
COCO_NOSE, COCO_LSH, COCO_RSH, COCO_LEL, COCO_REL, COCO_LWR, COCO_RWR = 0, 5, 6, 7, 8, 9, 10
COCO_LHI, COCO_RHI, COCO_LKN, COCO_RKN, COCO_LAN, COCO_RAN = 11, 12, 13, 14, 15, 16
#: A limb shorter than this is drawn as nothing rather than as a degenerate sliver.
DEGENERATE_LIMB_M = 1e-4
#: Shoulder line to head centre, as a fraction of stature: `_P`'s neck_y plus the
#: head radius, less shoulder_y. `human` puts the head there and so must a posed one.
HEAD_CENTRE_FRAC = 0.870 + 0.0665 * 1.15 - 0.818


def human_posed(joints: np.ndarray, height_m: float = 1.70, sides: int = 10) -> Mesh:
    """:func:`human`'s limbs, between measured joints instead of the standing constants.

    ``joints`` is ``(17, 3)`` in the scene's own metres -- x right, y up from the floor,
    z forward -- in COCO keypoint order. Twelve of those seventeen land on a tube endpoint
    :func:`human` already has, which is why this needs no rig and no skinning: the figure
    was always a set of tubes between named joints, and this passes different names in.

    **What this is not.** The joints it is given come from lifting 2D keypoints, and on a
    single frame that lift is not solved (PLAN 7c.30). The cheapest lift puts every joint
    on one vertical plane at the feet's range, which gets the heights and the limb
    directions right and the bone lengths wrong by 0.67-1.15x; the solvers that fix the
    bone lengths fold the skeleton instead. So a figure drawn from this shows *what the
    person is doing* and must not be measured: a limb length or a joint separation read
    off it is a property of the lift, not of the person.

    Unlike :func:`human` the mesh is NOT normalised to ``height_m``. The height is the
    joints' own, because scaling a posed figure to a standing person's stature would
    stretch a crouch into something taller than the crouch was.
    """
    j = np.asarray(joints, dtype=float)
    if j.shape != (17, 3):
        raise ValueError(f"joints must be (17, 3) in COCO order, got {j.shape}")
    if not np.isfinite(j).all():
        raise ValueError("joints contains non-finite values; drop the figure instead")
    h = float(height_m)
    if h <= 0:
        raise ValueError(f"height_m must be positive, got {h}")
    p = {k: v * h for k, v in _P.items()}
    mid_sh = (j[COCO_LSH] + j[COCO_RSH]) / 2
    mid_hip = (j[COCO_LHI] + j[COCO_RHI]) / 2
    # **The head sits on the spine, and the nose is on the front of the head.** Two wrong
    # versions of this, and the second is the instructive one. Deriving the head centre
    # from the nose alone put it a median 12 cm above the shoulders where a person's is
    # about 20 -- the lift flattens depth and from a ceiling camera the neck points most
    # directly at the lens, so it foreshortens worst -- and with an 11 cm head radius the
    # sphere sat inside the shoulders. Placing it along the shoulders-to-nose direction at
    # the canon's distance fixed the distance and kept the direction, which was the error:
    # that direction points forward and up, because the nose is on the FRONT of the head.
    # Every figure then craned its neck forward and still read as hunched.
    #
    # The head centre is above the neck, so the direction that places it is the spine's --
    # hips to shoulders. A person leaning over a counter leans their head with their torso
    # and the figure shows it. What this does NOT show is the head's own tilt: someone
    # standing straight while looking down at a phone renders looking ahead. The nose is
    # not used to place the head at all, because the one thing it reliably says is which
    # way the face points, and a sphere has no face.
    spine = mid_sh - mid_hip
    spine_len = float(np.linalg.norm(spine))
    up = spine / spine_len if spine_len > 1e-6 else np.array([0.0, 1.0, 0.0])
    head_c = mid_sh + up * (HEAD_CENTRE_FRAC * h)

    # A limb of zero length is what a *lift* produces when two keypoints land on the same
    # ray at the same depth -- a fully foreshortened forearm pointing at the camera, or two
    # low-confidence joints collapsing onto each other. `_tube` refuses coincident
    # endpoints, which is right for geometry somebody authored and wrong for geometry
    # somebody measured: it killed a 900-frame render partway through. Drawing nothing is
    # the correct picture of a limb with no extent; raising is not.
    segments = [
        (mid_sh, head_c, p["waist_r"] * 0.55, p["torso_r"] * 0.7),
        (mid_sh, mid_hip, p["torso_r"], p["waist_r"]),
    ]
    for sh, el, wr, hi, kn, an in (
        (COCO_LSH, COCO_LEL, COCO_LWR, COCO_LHI, COCO_LKN, COCO_LAN),
        (COCO_RSH, COCO_REL, COCO_RWR, COCO_RHI, COCO_RKN, COCO_RAN),
    ):
        segments += [
            (j[sh], j[el], p["upper_arm_r"], p["upper_arm_r"] * 0.82),
            (j[el], j[wr], p["forearm_r"], p["forearm_r"] * 0.8),
            (j[hi], j[kn], p["thigh_r"], p["calf_r"] * 1.15),
            (j[kn], j[an], p["calf_r"], p["calf_r"] * 0.62),
        ]
    parts = [_sphere(head_c, p["head_r"], sides=sides)]
    parts += [
        _tube(a, b, r0, r1, sides)
        for a, b, r0, r1 in segments
        if float(np.linalg.norm(np.asarray(b) - np.asarray(a))) >= DEGENERATE_LIMB_M
    ]
    return _merge(*parts)


def box(width_m: float, height_m: float, depth_m: float) -> Mesh:
    """An axis-aligned box standing on y = 0, centred on x and z.

    What a fixture gets: its BEV footprint is measured, its height usually is not, so the
    caller passes what it knows and the renderer is expected to say which is which.
    """
    for name, value in (("width_m", width_m), ("height_m", height_m), ("depth_m", depth_m)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    hx, hz = width_m / 2, depth_m / 2
    v = np.array(
        [[-hx, 0, -hz], [hx, 0, -hz], [hx, 0, hz], [-hx, 0, hz],
         [-hx, height_m, -hz], [hx, height_m, -hz], [hx, height_m, hz], [-hx, height_m, hz]],
        float,
    )  # fmt: skip
    f = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
        int,
    )  # fmt: skip
    return v, f


def _ear_clip(poly: np.ndarray) -> np.ndarray:
    """Triangulate a simple polygon given counter-clockwise in the xz plane.

    Ear clipping rather than a fan, because a real footprint is not convex: a gondola run
    with an end cap is L-shaped and a shelf bay is U-shaped, and a fan over either fills
    in the notch -- silently, as a solid where the aisle is.
    """
    idx = list(range(len(poly)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10 * len(poly):
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0:  # reflex corner, not an ear
                continue
            others = [poly[j] for j in idx if j not in (i0, i1, i2)]
            if any(_in_triangle(p, a, b, c) for p in others):
                continue
            tris.append([i0, i1, i2])
            idx.pop(k)
            break
        else:
            break  # nothing clippable: degenerate or self-intersecting input
    if len(idx) == 3:
        tris.append(idx)
    return np.asarray(tris, int)


def _in_triangle(p, a, b, c) -> bool:
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def extrude(footprint, height_m: float) -> Mesh:
    """A prism from an (x, z) footprint polygon. The primitive everything below is built on.

    This is the shape the perception actually produces: `bev.scene` puts a class on every
    floor cell, and connected components over the `fixture` or `column` cells give a
    polygon per object. So an extrusion is not a modelling convenience -- **the footprint
    is measured and the height is not**, and keeping them as separate arguments is what
    stops the two being confused by whoever renders the result.
    """
    poly = np.asarray(footprint, float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError(f"footprint must be (N>=3, 2) of (x, z), got shape {poly.shape}")
    if height_m <= 0:
        raise ValueError(f"height_m must be positive, got {height_m}")
    # Signed area fixes the winding, so a caller need not know which way round it traced
    # its contour -- OpenCV and a hand-written walk disagree about that often enough.
    area = 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                              - np.roll(poly[:, 0], -1) * poly[:, 1]))  # fmt: skip
    if area < 0:
        poly = poly[::-1]
    n = len(poly)

    bottom = np.stack([poly[:, 0], np.zeros(n), poly[:, 1]], 1)
    verts = np.vstack([bottom, bottom + [0, height_m, 0]])  # noqa: RUF005
    caps = _ear_clip(poly)
    faces = [[c, b, a] for a, b, c in caps]  # floor, wound downward
    faces += [[a + n, b + n, c + n] for a, b, c in caps]
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])
    return verts, np.asarray(faces, int)


def column(width_m: float, depth_m: float, height_m: float, *, round_: bool = False) -> Mesh:
    """A structural column. `round_` for a circular section.

    The class this taxonomy was created to split out of `wall`, so it gets its own builder
    rather than being a box: a square pillar and a round one read completely differently
    in a shop, and the footprint mask says which.
    """
    if round_:
        ang = np.linspace(0, 2 * math.pi, 20, endpoint=False)
        foot = np.stack([width_m / 2 * np.cos(ang), depth_m / 2 * np.sin(ang)], 1)
        return extrude(foot, height_m)
    return box(width_m, height_m, depth_m)


def wall(points, height_m: float, thickness_m: float = 0.12) -> Mesh:
    """A wall along a polyline, given as (x, z) points in metres.

    Takes the BEV boundary directly. Height is the caller's assumption -- `bev3d.py` uses
    2.4 m and says so in its docstring: "a wall is drawn 2.4 m tall because walls are
    about that; the camera never said so". Nothing here changes that, it just makes the
    assumption an argument instead of a constant.
    """
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError(f"points must be (N>=2, 2) of (x, z), got shape {pts.shape}")
    segments = []
    for a, b in itertools.pairwise(pts):
        d = b - a
        length = float(np.hypot(*d))
        if length < 1e-9:
            continue
        nrm = np.array([-d[1], d[0]]) / length * (thickness_m / 2)
        segments.append(extrude([a + nrm, b + nrm, b - nrm, a - nrm], height_m))
    if not segments:
        raise ValueError("every wall segment had zero length")
    return _merge(*segments)


def table(
    width_m: float, depth_m: float, height_m: float = 0.75, *, leg_r: float = 0.03
) -> Mesh:
    """A display table: a top slab on four legs.

    0.75 m is desk height and a reasonable default for a display table. An Apple-store
    podium is nearer 0.9 m, which is why it is an argument.
    """
    for name, value in (("width_m", width_m), ("depth_m", depth_m), ("height_m", height_m)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    top_t = min(0.04, height_m * 0.12)
    parts = [(lambda m: (m[0] + [0, height_m - top_t, 0], m[1]))(box(width_m, top_t, depth_m))]
    inset = leg_r * 2.5
    for sx in (-1, 1):
        for sz in (-1, 1):
            x = sx * (width_m / 2 - inset)
            z = sz * (depth_m / 2 - inset)
            parts.append(_tube([x, 0, z], [x, height_m - top_t, z], leg_r, leg_r * 0.9, 8))
    return _merge(*parts)


def chair(width_m: float, depth_m: float, height_m: float) -> Mesh:
    """A seat slab on four legs with a back. ``height_m`` is the top of the back.

    The seat sits at 0.45 of the total height, which is the ratio a chair holds across the
    range of chairs -- an office chair and a dining chair differ in overall height far more
    than they differ in that proportion. So the one number a detection actually supplies
    places the seat, and the seat is the feature that makes a chair read as a chair from
    above rather than as a small box.
    """
    for name, value in (("width_m", width_m), ("depth_m", depth_m), ("height_m", height_m)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    seat_y = height_m * 0.45
    slab_t = min(0.05, seat_y * 0.14)
    back_t = min(0.05, depth_m * 0.12)
    leg_r = min(0.022, width_m * 0.06)
    parts = [(lambda m: (m[0] + [0, seat_y - slab_t, 0], m[1]))(box(width_m, slab_t, depth_m))]
    back = box(width_m, height_m - seat_y, back_t)
    parts.append((back[0] + [0, seat_y, depth_m / 2 - back_t / 2], back[1]))
    inset = leg_r * 2.5
    for sx in (-1, 1):
        for sz in (-1, 1):
            x = sx * (width_m / 2 - inset)
            z = sz * (depth_m / 2 - inset)
            parts.append(_tube([x, 0, z], [x, seat_y - slab_t, z], leg_r, leg_r * 0.85, 8))
    return _merge(*parts)


def cabinet(width_m: float, depth_m: float, height_m: float, *, shelves: int = 3) -> Mesh:
    """A shelf unit or gondola: a carcass with visible shelf slabs.

    The shelves are the reason this is not `box()`. A gondola rendered as a solid block
    loses the one feature that makes a shop floor legible from above -- that merchandise
    sits on horizontal planes at known heights, which is also where `product` lives.
    """
    if shelves < 0:
        raise ValueError(f"shelves must be non-negative, got {shelves}")
    back_t = min(0.05, depth_m * 0.15)
    parts = [
        (lambda m: (m[0] + [0, 0, depth_m / 2 - back_t / 2], m[1]))(
            box(width_m, height_m, back_t)
        )
    ]
    for sx in (-1, 1):
        side = box(min(0.04, width_m * 0.06), height_m, depth_m)
        parts.append((side[0] + [sx * (width_m / 2), 0, 0], side[1]))
    slab_t = min(0.03, height_m * 0.05)
    for i in range(shelves + 1):
        y = height_m * i / max(shelves, 1) if shelves else 0.0
        y = min(y, height_m - slab_t)
        slab = box(width_m, slab_t, depth_m)
        parts.append((slab[0] + [0, y, 0], slab[1]))
    return _merge(*parts)


KICK_PLATE_M = 0.10  # the recess a shop fixture stands on, whatever its height


def shelf_levels(height_m: float, *, pitch_m: float = 0.32) -> list[float]:
    """The heights `shelving` puts its shelf surfaces at, feet at y = 0.

    Exported so a caller placing merchandise can put it *on a shelf* rather than at
    whatever height the depth model measured for the region. The two must not drift, so
    the mesh reads its levels from here rather than the other way round: a product
    snapped to a level this function invented would sit in mid-air the moment the mesh
    changed its spacing.
    """
    if height_m <= 0 or pitch_m <= 0:
        raise ValueError(f"shelf_levels needs positive extents, got {height_m}, {pitch_m}")
    slab_t = min(0.025, height_m * 0.03)
    out, y = [], min(KICK_PLATE_M, height_m * 0.5)
    while y <= height_m - slab_t:
        out.append(float(y))
        y += pitch_m
    return out


# the recess a shop fixture stands on, whatever its height


def shelving(
    width_m: float,
    depth_m: float,
    height_m: float,
    *,
    pitch_m: float = 0.32,
    lip_m: float = 0.045,
) -> Mesh:
    """Open retail shelving -- a merchandise wall, not a bookcase.

    `cabinet()` renders a carcass: solid end panels the full depth, shelf slabs the full
    depth, evenly spaced from the floor to the top. That is a bookcase, and on a shop
    floor it is the wrong object -- the accessory walls in these stores are open-fronted
    units where the back panel shows above every shelf and the merchandise faces out.

    Four things make the difference, and each is a feature of the real thing rather than
    a styling choice:

    * **the shelves are shallower than the unit**, so the back panel is visible in the
      gap above each one -- that stripe is most of what tells a shelf run from a cabinet;
    * **a front lip** on each shelf, which is what stops stock sliding off and is the
      most recognisable edge of retail shelving from any angle;
    * **a recessed kick plate** instead of a slab lying on the floor, so the unit reads as
      standing rather than as a box resting on the ground;
    * **end posts, not end panels** -- thin uprights at the corners, so the run is open
      from the side the way a gondola is.

    `pitch_m` is the shelf spacing rather than a shelf count: a 2.4 m wall and a 1.2 m
    low unit should have shelves the same distance apart, and asking for "5 shelves"
    stretches them apart on the tall one. 0.32 m is what these stores' accessory walls
    are built on.
    """
    if width_m <= 0 or depth_m <= 0 or height_m <= 0:
        raise ValueError(f"shelving needs positive extents, got {width_m}x{depth_m}x{height_m}")
    if pitch_m <= 0:
        raise ValueError(f"pitch_m must be positive, got {pitch_m}")
    back_t = min(0.03, depth_m * 0.12)
    post_w = min(0.05, width_m * 0.05)
    # A constant, not a fraction of the height: a kick plate is a kick plate on a 1.2 m
    # unit and on a 2.4 m wall, and scaling it moved the first shelf, so two units built
    # on the same pitch came back with their shelves at different heights.
    kick_h = min(KICK_PLATE_M, height_m * 0.5)
    shelf_d = depth_m * 0.82  # the gap is what shows the back panel
    slab_t = min(0.025, height_m * 0.03)
    z_back = depth_m / 2 - back_t / 2
    z_shelf = -depth_m / 2 + shelf_d / 2  # front-aligned, like a real run

    def at(mesh, dx=0.0, dy=0.0, dz=0.0):
        return (mesh[0] + [dx, dy, dz], mesh[1])

    parts = [at(box(width_m, height_m, back_t), dz=z_back)]
    for sx in (-1, 1):
        parts.append(at(box(post_w, height_m, depth_m * 0.9), sx * (width_m - post_w) / 2))
    parts.append(at(box(width_m * 0.96, kick_h, shelf_d * 0.7), dz=z_shelf))
    inner = max(width_m - 2 * post_w, width_m * 0.5)
    for y in shelf_levels(height_m, pitch_m=pitch_m):
        parts.append(at(box(inner, slab_t, shelf_d), dy=y, dz=z_shelf))
        parts.append(
            at(box(inner, lip_m, slab_t), dy=y + slab_t, dz=z_shelf - shelf_d / 2 + slab_t / 2)
        )
    # A top panel. These cameras look down, so the most prominent face of every unit is
    # its top: left open it reads as an empty tray with a lip on the far side, which is a
    # warehouse pallet rack rather than a shop fixture.
    parts.append(at(box(width_m, slab_t, depth_m), dy=height_m - slab_t))
    return _merge(*parts)


def ground_disc(radius_m: float, sides: int = 32) -> Mesh:
    """A flat disc at y = 0 -- the channel the position uncertainty goes in.

    Drawn under a figure at the standard error of its floor position, so the viewer reads
    "somewhere in here" off the floor, in metres, next to everything else in metres.
    Putting it here rather than in the figure's silhouette is the whole argument of this
    module: the mesh answers "what", the disc answers "how well do we know where".
    """
    if radius_m <= 0:
        raise ValueError(f"radius_m must be positive, got {radius_m}")
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    rim = np.stack([radius_m * np.cos(ang), np.zeros(sides), radius_m * np.sin(ang)], 1)
    verts = np.vstack([[[0.0, 0.0, 0.0]], rim])
    faces = np.array([[0, 1 + i, 1 + (i + 1) % sides] for i in range(sides)], int)
    return verts, faces


def place(mesh: Mesh, at: Placement) -> Mesh:
    """Rotate about y by ``heading_rad`` (if any) and translate onto the floor."""
    verts, faces = mesh
    verts = verts.copy()
    if at.heading_rad is not None:
        c, s = math.cos(at.heading_rad), math.sin(at.heading_rad)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        verts = verts @ rot.T
    verts[:, 0] += at.x_m
    verts[:, 2] += at.z_m
    return verts, faces


# **The crease angle: smooth across a curve, never across a hard edge.**
#
# 40 degrees, and the margin either side is what makes it a threshold rather than a tuned
# number. The two shapes this has to separate are measured, not guessed:
#
#   * a round column is 20 side facets, so neighbouring facets meet at **18 degrees** and
#     must keep sharing normals or the cylinder reads as a prism;
#   * a box's faces meet at **90 degrees** and must not, or the two triangles of one flat
#     top get different normals and the quad is painted in two brightnesses.
#
# An earlier attempt thresholded the *smoothed* normal against the face's own, which put
# the box at 21.6 degrees against the tube's 18 -- 3.6 degrees apart, and no threshold on
# that statistic is safe. Comparing neighbours instead puts them 72 degrees apart.
CREASE_COS = math.cos(math.radians(40.0))


#: Shading normals by mesh content. Cleared wholesale rather than evicted one at a
#: time: the working set is a scene's worth of furniture plus a few figures, so a
#: cache that has grown past this is a render whose geometry changes every frame,
#: and for that one nothing is worth keeping.
_NORMALS_CACHE: dict[tuple, np.ndarray] = {}
NORMALS_CACHE_MAX = 512


def smooth_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """One shading normal per face, averaged over its neighbours across smooth joins only.

    Geometry rather than rendering, which is why it lives here: it is a property of the
    mesh and any renderer wants the same answer. It is also **most of the difference
    between a figure and a faceted toy** -- with per-face geometric normals a 10-sided
    tube reads as a prism, and with these it reads as a cylinder. PIL cannot interpolate
    across a polygon, so this is the cheap stand-in for Gouraud: adjacent faces differ by
    a little instead of a lot.

    **A face only borrows from neighbours it agrees with.** Averaging through every shared
    vertex was right for the tube and wrong for everything boxy: a box corner belongs to
    three perpendicular faces, so the two triangles of one flat top averaged to different
    normals and every cabinet, counter and slab in the commissioning renders carried a
    diagonal seam down it. A reviewer reading those images on 2026-08-27 called it a tent
    roof and, on a thin slab, a "45-degree tilted panel, physically impossible" -- a
    shading artefact read as a geometry failure, which is the expensive kind of wrong.

    **Memoised, because a video render asks the same question 900 times.** The normals are
    a pure function of the mesh and a commissioning render draws the same shop on every
    frame: profiling `heads_video` over 20 frames put 10.4 s of a 31 s render inside this
    function, 784 calls for 20 frames, and all but a handful were the furniture -- built
    once, outside the frame loop, and re-normalled for each of them. Only the *figures*
    change frame to frame, so the cache is bounded and the misses are the ones that should
    miss. Keyed on the content rather than on `id()`, so a mesh rebuilt into fresh arrays
    still hits and a recycled address cannot collide.

    The returned array is read-only: it is shared between callers now, so a caller that
    wrote through it would corrupt another frame's shading rather than its own.
    """
    if len(faces) == 0:
        return np.zeros((0, 3))
    verts = np.ascontiguousarray(verts)
    faces = np.ascontiguousarray(faces)
    key = (
        verts.shape,
        faces.shape,
        verts.dtype.str,
        faces.dtype.str,
        hash(verts.tobytes()),
        hash(faces.tobytes()),
    )
    cached = _NORMALS_CACHE.get(key)
    if cached is not None:
        return cached
    fn = np.cross(
        verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]]
    )
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    incident: dict[int, list[int]] = {}
    for fi, tri in enumerate(faces):
        for v in tri:
            incident.setdefault(int(v), []).append(fi)
    out = np.empty_like(fn)
    for fi, tri in enumerate(faces):
        acc = np.zeros(3)
        for v in tri:
            for fj in incident[int(v)]:
                if float(fn[fi] @ fn[fj]) >= CREASE_COS:
                    acc += fn[fj]
        norm = float(np.linalg.norm(acc))
        out[fi] = acc / norm if norm > 1e-12 else fn[fi]
    out.flags.writeable = False
    if len(_NORMALS_CACHE) >= NORMALS_CACHE_MAX:
        _NORMALS_CACHE.clear()
    _NORMALS_CACHE[key] = out
    return out


# Nominal depth in metres, per class that gets a shape. **Depth is the one dimension the
# payload never carries**: `bev.scene` measures the footprint width and the height of the
# box against the ground plane, and nothing measures how far the thing extends away from
# the camera. The fallback in this module has always invented one -- `box(w, h, w)` says
# "as deep as it is wide", which is a claim too, just an unlabelled one. These are the same
# claim written down where it can be argued with.
_NOMINAL_DEPTH_M = {"chair": 0.50, "table": 0.75, "cabinet": 0.55}

# Which builder a **detection** class gets, keyed on the name `cli/scene.detection_class_
# names` established -- COCO's 80 when the head has 80 channels, the head's own `classes:`
# otherwise. Not terrain names: those describe pixels, and nothing here draws a pixel.
#
# The shape claim is the *detector's*, not the renderer's: a box labelled `chair` already
# asserts "this is a chair", and drawing a chair adds nothing to it. What this map must not
# do is put a shape on a class whose name does not determine one -- `potted plant` and
# `backpack` are any shape at all, and a viewer cannot tell a modelled silhouette from a
# measured one, so those keep the extrusion.
#
# **`display_fixture` is deliberately absent**, and it is the entry most likely to be
# proposed again. Two reasons, either sufficient. It is a terrain id (12 in
# `RETAIL_TERRAIN`), so putting it here quietly claims that a segmentation class can arrive
# through the detection payload. And `label_maps_retail_objects.py` records its test IoU at
# **0.336, the lowest of any class with real data**, with the failure being *which* object
# it is looking at: in one audited frame the round podium is `obstacle_furniture` while wall
# shelving three metres away is `display_fixture`. A distinctive silhouette on the class
# least able to tell two objects apart is the shape being more confident than the label
# under it.
_SHAPE = {
    "chair": "chair",
    "bench": "chair",
    "dining table": "table",
    "desk": "table",
    "table": "table",
    "shelf": "cabinet",
    "bookshelf": "cabinet",
    "cabinet": "cabinet",
    "refrigerator": "cabinet",
}


def detected_class(name) -> str:
    """The class out of a scene payload's ``name``, which may carry a grouping in front.

    `hydranet-scene --vocab retail` labels a box ``fixture/oven``: the group is the reading
    and the COCO word is the evidence for it, kept deliberately so a wrong grouping stays
    falsifiable from the rendered frame (`data/coco_subsets.retail_box_label`). Everything
    that keys off the class -- the shape table here, the colours in `bev3d` -- wants the
    evidence half, or a renamed run silently loses every shape and every colour it had.

    A name with no group is returned unchanged, which is every other caller.
    """
    return str(name).rsplit("/", 1)[-1]


def for_object(obj: dict) -> Mesh | None:
    """The mesh a scene-payload object should be drawn as, or None to leave it alone.

    The mapping from a detected class to a shape, in one place, so the renderer does not
    grow a second opinion about it. Everything not named in `_SHAPE` stays the extruded
    footprint the flat map already asserted.

    Heights follow the payload where it has one. `height_m` is None when the box could not
    be placed against the ground plane (`bev.scene` says so explicitly rather than
    substituting a number), and a person then falls back to 1.70 m -- the same assumption
    `fit_camera_from_people.py` fitted camera pose against, and an assumption there too.

    **A shape carries no confidence, so whoever draws it has to.** `score` is in the
    payload and is not read here: a chair at 0.31 gets exactly the mesh a chair at 0.95
    gets, because the shape follows from the *name* and the name is what the detector
    committed to. That is the correct split, but it leaves the panel able to look more
    certain than the run behind it -- `cli/scene.py` labels its 2D boxes `chair 0.31`, and
    a viewer discounts that in a way they cannot discount a silhouette. `bev3d.py` puts the
    score on the label beside the mesh for that reason; a renderer that drops it is
    laundering the detector's doubt.
    """
    name = detected_class(obj.get("name", ""))
    height = obj.get("height_m")
    width = obj.get("width_m")
    if name == "person":
        return human(float(height) if height and height > 0.6 else 1.70)
    if not (width and height):
        return None
    w, h = float(width), float(height)
    kind = _SHAPE.get(name)
    if kind is None:
        return box(w, h, w)
    # Never deeper than it is wide: the width is measured and the depth is not, and a
    # fixture drawn deeper than its own measured footprint contradicts the flat map
    # underneath it -- which is the one place a viewer could catch this being invented.
    d = min(_NOMINAL_DEPTH_M[kind], w)
    if kind == "chair":
        return chair(w, d, h)
    if kind == "table":
        return table(w, d, h)
    # One shelf per 0.45 m of carcass, which is roughly the pitch a gondola is built on.
    # `shelving`, the same mesh the commissioning renderer uses. Two paths drawing two
    # different objects for one physical fixture is worse than either being wrong.
    return shelving(w, d, h)


def to_obj(mesh: Mesh, name: str = "mesh") -> str:
    """Wavefront OBJ. Text, no dependency, and read by every 3D tool including Blender,
    three.js, Open3D and rerun -- which keeps this module free of a renderer choice."""
    verts, faces = mesh
    lines = [f"o {name}"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return "\n".join(lines) + "\n"
