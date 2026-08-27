"""One camera and one light rig, shared by everything in this package that draws a mesh.

`bev3d.py` renders the panel a person actually looks at; `scripts/mesh_preview.py`
rendered the mesh library on a hand-authored scene until `500cdd2` removed it
(`git show 500cdd2^:scripts/mesh_preview.py`; `tools/commissioning/scene_mesh.py`
succeeded it). Both had grown their own projection and their
own shading, and the two had drifted apart: the preview had a three-light rig, a depth
fade and grouped contact shadows while the panel still lit its meshes with two terms and
no shadow at all, so the same `meshes.human()` came out looking like two different objects
depending on which file you ran. `meshes.smooth_normals` already made this argument once
and only got half way -- it was promoted out of the preview script because "bev3d.py needs
the same answer", which is exactly as true of the projection and the lights.

**A mirror this consolidation fixes.** The preview built its camera basis as
``right = fwd x up_world``, which under this package's stated axes -- **x right, y up, z
forward** -- puts world +x on the *left* of the picture. Nothing caught it because the
scene it drew was hand-authored, so there was no ground truth for it to disagree with.
`bev3d` had it the other way round and a test pinning it, `test_lateral_sign_is_not_
mirrored`, because a mirrored panel is a map that sends a robot the wrong way. This module
keeps `bev3d`'s convention, so the preview renders mirrored against the image committed in
`f0558ef`. That is the fix, not a regression.

**What the shading is for, given it invents everything.** None of it is measurement: there
is no light in the store this rig corresponds to. It exists because a viewer reads shape
off shading, and a scene lit by one lamp with no fill reads as flat cut-outs -- which is a
worse lie than an invented light, because the viewer blames the *geometry* for it and
concludes the perception is cruder than it is.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .meshes import Mesh, smooth_normals

UP_WORLD = np.array([0.0, 1.0, 0.0])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


# The rig, in world directions. Three terms, and each one is doing a specific job:
#
# * KEY, from above and in front, is the only term that describes the form.
# * FILL, from the opposite side and much weaker, keeps the shadowed side off pure black.
#   Without it every face turned away from the key is the same value as every other, and a
#   figure loses its silhouette against a dark floor.
# * RIM brightens faces turned away from the camera. It is what separates a dark object
#   from a dark background *without* drawing an outline -- and an outline is the thing to
#   avoid here, because this panel already spends outlines on the boundary cap, where they
#   mean "an edge you could walk into".
KEY = _unit([0.42, 0.80, -0.44])
FILL = _unit([-0.55, 0.35, 0.30])

AMBIENT = 0.38
KEY_GAIN = 0.62
FILL_GAIN = 0.22
RIM_GAIN = 0.26
RIM_POWER = 2.5

# Distance fade, in metres of camera depth. The far edge of a mapped window is an artefact
# of the window, not of the room, and a scene that ends on a hard bright line reads as a
# room that stops there.
FOG_START_M = 5.0
FOG_SPAN_M = 22.0
FOG_MAX = 0.42


class View:
    """A camera as an eye, a point it looks at, and a focal length in output pixels.

    Unrelated to the camera that took the frame -- this is a drawing device. Apparent size
    in anything drawn through it means nothing; the distance rules on the floor are the
    only scale.

    ``right`` is ``up_world x fwd`` and not the other way round. See the module docstring:
    the other order mirrors the picture, and a mirrored map is wrong rather than ugly.
    """

    __slots__ = ("cx", "cy", "eye", "focal_px", "fwd", "right", "up")

    def __init__(self, eye, target, focal_px: float, cx: float = 0.0, cy: float = 0.0):
        self.eye = np.asarray(eye, float)
        fwd = np.asarray(target, float) - self.eye
        if float(np.linalg.norm(fwd)) < 1e-9:
            raise ValueError("eye and target coincide, so the view has no direction")
        self.fwd = _unit(fwd)
        right = np.cross(UP_WORLD, self.fwd)
        if float(np.linalg.norm(right)) < 1e-6:
            raise ValueError("a straight-down view has no lateral axis; tilt it off vertical")
        self.right = _unit(right)
        self.up = np.cross(self.fwd, self.right)
        self.focal_px = float(focal_px)
        self.cx = float(cx)
        self.cy = float(cy)

    def with_intrinsics(self, focal_px: float, cx: float, cy: float) -> View:
        """The same eye and direction at a different focal length and principal point.

        Which is what fitting a scene to a panel changes, and only that: refitting must not
        be able to move the camera, or the fit and the geometry stop describing one view.
        """
        return View(self.eye, self.eye + self.fwd, focal_px, cx, cy)

    def project_points(self, points):
        """``(..., 3)`` world metres -> ``(..., 2)`` panel pixels and ``(...)`` camera depth.

        Depth is returned **unclamped**, so a caller can tell "behind the camera" from
        "very close". Only the division is guarded; a point behind the eye still projects
        to a number, and it is the caller's job to cull on the sign.
        """
        d = np.asarray(points, float) - self.eye
        xc = d @ self.right
        yc = d @ self.up
        zc = d @ self.fwd
        safe = np.where(np.abs(zc) < 1e-3, 1e-3, zc)
        return np.stack([self.cx + self.focal_px * xc / safe,
                         self.cy - self.focal_px * yc / safe], -1), zc  # fmt: skip

    def project(self, x, y, z):
        """Scalar/array world coordinates -> ``(u, v, depth)``, broadcast together."""
        x, y, z = (np.asarray(v, float) for v in (x, y, z))
        shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
        pts = np.stack(np.broadcast_arrays(x, y, z), -1).reshape(-1, 3)
        uv, depth = self.project_points(pts)
        return (
            uv[:, 0].reshape(shape),
            uv[:, 1].reshape(shape),
            depth.reshape(shape),
        )


def shade(normals, rgb, depth, view: View, *, bg, fog: bool = True) -> np.ndarray:
    """Key + fill + rim per normal, then faded toward ``bg`` with camera depth.

    ``normals`` is one shading normal per face -- `meshes.smooth_normals`, not the raw
    geometric ones. That substitution is most of the difference between a figure and a
    faceted toy, and it is the reason this rig looks like anything at all: PIL cannot
    interpolate a colour across a polygon, so smooth normals are the only Gouraud
    available.
    """
    n = np.asarray(normals, float)
    key = np.maximum(n @ KEY, 0.0)
    fill = np.maximum(n @ FILL, 0.0)
    rim = np.power(np.clip(1.0 - np.abs(n @ view.fwd), 0.0, 1.0), RIM_POWER)
    lit = AMBIENT + KEY_GAIN * key + FILL_GAIN * fill + RIM_GAIN * rim
    colour = np.asarray(rgb, float)[None, :3] * lit[:, None]
    if not fog:
        return np.clip(colour, 0, 255)
    haze = np.clip((np.asarray(depth, float) - FOG_START_M) / FOG_SPAN_M, 0.0, FOG_MAX)[:, None]
    return np.clip(colour * (1 - haze) + np.asarray(bg, float)[None, :3] * haze, 0, 255)


def draw_scene(
    draw: ImageDraw.ImageDraw,
    view: View,
    items: list[tuple[Mesh, tuple, int]],
    *,
    bg,
    fog: bool = True,
) -> None:
    """Paint ``(mesh, rgb, alpha)`` items back to front through ``view``, in one sort.

    **One sort across every item, not one per mesh.** Sorting each mesh separately makes
    the draw order between two meshes an accident of the order the caller listed them in,
    which is how a figure ends up painted over a fixture standing in front of it. Merging
    the faces first costs a concatenate and removes the whole class of error.

    **A painter's algorithm, not a z-buffer**, and the limit belongs next to the code
    rather than only in a script's docstring: faces are sorted by centroid depth, so two
    long surfaces that interpenetrate can sort wrongly -- visible where a shelf passes
    behind a column. Correct for convex objects standing apart, which is what a scene of
    detections is. Anything that has to be *correct* rather than illustrative wants
    per-pixel depth, which means a real renderer: `meshes.to_obj` exists so that rerun,
    Open3D or Blender can be that renderer.
    """
    tris: list[tuple[float, list, tuple]] = []
    for (verts, faces), rgb, alpha in items:
        if len(faces) == 0:
            continue
        uv, vert_depth = view.project_points(verts)
        depth = vert_depth[faces].mean(axis=1)
        colours = shade(smooth_normals(verts, faces), rgb, depth, view, bg=bg, fog=fog)
        for face, dep, colour in zip(faces, depth, colours, strict=True):
            if dep <= 0.0:  # behind the eye
                continue
            tris.append(
                (
                    float(dep),
                    [(float(uv[i, 0]), float(uv[i, 1])) for i in face],
                    (*(int(c) for c in colour), int(alpha)),
                )
            )
    for _, points, colour in sorted(tris, key=lambda t: -t[0]):
        draw.polygon(points, fill=colour)


def draw_mesh(
    draw: ImageDraw.ImageDraw,
    view: View,
    mesh: Mesh,
    rgb,
    *,
    bg,
    alpha: int = 255,
    fog: bool = True,
) -> None:
    """One mesh, for a caller that has only one. `draw_scene` for anything else."""
    draw_scene(draw, view, [(mesh, rgb, alpha)], bg=bg, fog=fog)


def contact_shadows(
    size: tuple[int, int],
    view: View,
    meshes: list[Mesh],
    *,
    blur_px: float,
    alpha: int = 105,
    margin_m: float = 0.10,
) -> Image.Image:
    """An RGBA layer of soft ellipses under everything that touches the floor.

    Without one, objects hover. Hovering is the artefact a viewer blames on the *geometry*
    -- it reads as "the system does not know these things are on the ground", when what it
    actually is is a missing shadow.

    Blurred as a group, in one pass, rather than per object: a per-object blur costs a full
    filter each and makes overlapping shadows stack to black where they meet.
    """
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if not meshes:
        return layer
    d = ImageDraw.Draw(layer)
    for verts, _ in meshes:
        if len(verts) == 0:
            continue
        cx = float(verts[:, 0].min() + verts[:, 0].max()) / 2
        cz = float(verts[:, 2].min() + verts[:, 2].max()) / 2
        rx = float(np.ptp(verts[:, 0])) / 2 + margin_m
        rz = float(np.ptp(verts[:, 2])) / 2 + margin_m
        ring = np.stack(
            [
                cx + rx * np.cos(np.linspace(0, 2 * np.pi, 24)),
                np.zeros(24),
                cz + rz * np.sin(np.linspace(0, 2 * np.pi, 24)),
            ],
            -1,
        )
        uv, depth = view.project_points(ring)
        if (depth > 0).all():
            d.polygon([(float(u), float(v)) for u, v in uv], fill=(0, 0, 0, alpha))
    return layer.filter(ImageFilter.GaussianBlur(blur_px)) if blur_px > 0 else layer
