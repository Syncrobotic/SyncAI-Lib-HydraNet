"""Procedural meshes for the scene renderer: a human, a fixture, and a ground disc.

Generated rather than downloaded. A mesh file in the repository is a binary blob with a
licence to track, a CDN or an LFS pointer to serve it from, and no way to ask it for a
1.62 m person instead of a 1.70 m one. Everything here is a function of its dimensions,
which is what makes it usable for a scene where the dimensions are the output.

**On what a human-shaped mesh does and does not claim.** `bev_page.py` draws objects as
wireframe cuboids and argues, correctly, that a solid asset asserts a shape nobody
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

    1.70 m is the same assumption `scripts/fit_camera_from_people.py` fits camera pose
    against, and it is an assumption there too. Passing a height through rather than
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


def to_obj(mesh: Mesh, name: str = "mesh") -> str:
    """Wavefront OBJ. Text, no dependency, and read by every 3D tool including Blender,
    three.js, Open3D and rerun -- which keeps this module free of a renderer choice."""
    verts, faces = mesh
    lines = [f"o {name}"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return "\n".join(lines) + "\n"
