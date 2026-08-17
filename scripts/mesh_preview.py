#!/usr/bin/env python3
"""Render a shop scene from `geometry/meshes.py`, in the style of a car's 3D map.

    python3 scripts/mesh_preview.py

Writes `assets/mesh_shop_preview.png`. The point is not the picture -- it is that the
picture is reproducible, because an asset in `assets/` with no generator is the thing this
repository keeps finding and cannot re-derive.

**What makes the look, and none of it is the meshes.** The geometry here is a few hundred
triangles of tubes and prisms; the same meshes rendered with flat per-face shading read as
a faceted low-poly toy. Four things do the work, in order of how much:

1. **Smooth normals.** Each face is shaded by the average of its vertices' normals, where
   a vertex normal is the mean of the faces touching it. Adjacent faces then differ by a
   little instead of a lot, and a 20-sided tube reads as a cylinder rather than a prism.
   PIL cannot interpolate across a polygon, so this is the cheap stand-in for Gouraud and
   it is most of the difference.
2. **Three lights, not one.** A key light, a dim fill from the opposite side so nothing
   goes to pure black, and a rim term that brightens faces turned away from the camera --
   which is what separates a dark object from a dark background without an outline.
3. **Depth fade.** Colours lerp toward the background with distance. `bev3d.py` makes the
   same argument for its far edge: a scene that ends on a hard line reads as a room that
   stops there.
4. **Contact shadow.** A soft ellipse under everything that touches the floor. Without it
   objects hover, and hovering is exactly the artefact a viewer would blame on the
   geometry rather than on the shading.

**A deliberate limit: this is a painter's algorithm, not a z-buffer.** Faces are sorted by
centroid depth, so two long surfaces that interpenetrate can sort wrongly -- visible where
a shelf passes behind a column. Fixing it means per-pixel depth, which means a real
renderer. Anything that has to be *correct* rather than illustrative should go to one:
`rerun-sdk`, Open3D or a glTF export via `to_obj`, all of which read this module's output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.geometry.meshes import (  # noqa: E402
    Placement,
    cabinet,
    column,
    extrude,
    ground_disc,
    human,
    place,
    table,
    wall,
)

W, H, SS = 1180, 700, 2  # SS: supersample, resolved down at the end. Same trick as bev3d.
BG = np.array([17, 21, 28], float)

KEY = np.array([0.42, 0.80, -0.44])
FILL = np.array([-0.55, 0.35, 0.30])
KEY /= np.linalg.norm(KEY)
FILL /= np.linalg.norm(FILL)

# Desaturated and close together on purpose: in a car's map almost everything is one
# neutral and only what the driver must react to carries colour. Here that is people.
PALETTE = {
    "wall": (118, 128, 145),
    "column": (150, 160, 178),
    "fixture": (138, 149, 168),
    "table": (158, 166, 182),
    "product": (86, 214, 188),
    "person": (168, 208, 250),
    "disc": (242, 180, 78),
}


def smooth_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face shading normals, averaged through the vertices they share."""
    fn = np.cross(
        verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]]
    )
    lengths = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(lengths, 1e-12)
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    out = vn[faces].mean(axis=1)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


class View:
    def __init__(self, eye, target, focal):
        self.eye = np.asarray(eye, float)
        fwd = np.asarray(target, float) - self.eye
        self.fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(self.fwd, [0.0, 1.0, 0.0])
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.fwd)
        self.focal = focal

    def project(self, v):
        d = v - self.eye
        cam = np.stack([d @ self.right, d @ self.up, d @ self.fwd], -1)
        z = np.maximum(cam[:, 2], 1e-6)
        xy = np.stack(
            [W * SS / 2 + self.focal * SS * cam[:, 0] / z,
             H * SS / 2 - self.focal * SS * cam[:, 1] / z], -1
        )  # fmt: skip
        return xy, cam[:, 2]


def shade(normals, base, depth, view):
    """Key + fill + rim, then fade into the background with distance."""
    key = np.maximum(normals @ KEY, 0.0)
    fill = np.maximum(normals @ FILL, 0.0)
    rim = np.power(np.clip(1.0 - np.abs(normals @ view.fwd), 0.0, 1.0), 2.5)
    lit = 0.38 + 0.62 * key + 0.22 * fill + 0.26 * rim
    colour = np.asarray(base, float)[None, :] * lit[:, None]
    fog = np.clip((depth - 5.0) / 22.0, 0.0, 0.42)[:, None]
    return np.clip(colour * (1 - fog) + BG[None, :] * fog, 0, 255)


def main() -> None:
    # An elevated three-quarter view, which is the framing a car's map uses and is not a
    # taste call: from low down the fixtures occlude each other and the floor -- the one
    # surface every measurement here is expressed on -- is edge-on and unreadable.
    view = View([6.6, 5.0, -4.6], [0.35, 0.7, 2.5], 620.0)
    img = Image.new("RGB", (W * SS, H * SS), tuple(int(c) for c in BG))
    shadows = Image.new("RGBA", img.size, (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(shadows)

    items = []  # (mesh, colour_key, alpha, casts_shadow)
    # Walls translucent, and the reason is not composition. Only their *footprint* is
    # measured -- `bev3d.py`: "a wall is drawn 2.4 m tall because walls are about that;
    # the camera never said so". A solid slab asserts that height and occludes the
    # fixtures behind it; a translucent one shows where the room ends without claiming
    # how high it goes. The look and the honesty want the same thing here.
    items += [(wall([[-4.0, 5.4], [5.4, 5.4]], 2.7), "wall", 105, False)]
    items += [(wall([[5.4, 5.4], [5.4, -1.4]], 2.7), "wall", 105, False)]
    items += [(place(column(0.46, 0.46, 2.7), Placement(1.5, 3.3)), "column", 255, True)]
    items += [
        (place(column(0.52, 0.52, 2.7, round_=True), Placement(-2.4, 3.3)), "column", 255, True)
    ]
    for x in (-3.0, -0.4, 2.9):
        items.append((place(cabinet(2.1, 0.55, 2.05, shelves=4), Placement(x, 5.0)),
                      "fixture", 255, True))  # fmt: skip
        for s in range(4):
            slab = extrude([[-0.95, -0.2], [0.95, -0.2], [0.95, 0.2], [-0.95, 0.2]], 0.17)
            items.append(((slab[0] + [x, 0.07 + s * 2.05 / 4, 5.0], slab[1]),
                          "product", 255, False))  # fmt: skip
    for tx in (-1.7, 2.3):
        items.append((place(table(1.45, 0.9, 0.9), Placement(tx, 1.5)), "table", 255, True))
        top = extrude([[-0.5, -0.28], [0.5, -0.28], [0.5, 0.28], [-0.5, 0.28]], 0.06)
        items.append(((top[0] + [tx, 0.9, 1.5], top[1]), "product", 255, False))
    people = (
        (-0.8, 0.6, 1.72, 0.28, 20),
        (1.3, 2.6, 1.63, 0.55, -120),
        (3.7, 1.1, 1.78, 1.0, 150),
    )
    for x, z, hm, sigma, heading in people:
        items.append((place(ground_disc(sigma), Placement(x, z)), "disc", 120, False))
        items.append((place(human(hm), Placement(x, z, heading_rad=np.radians(heading))),
                      "person", 255, True))  # fmt: skip

    # Contact shadows go on the floor layer, under everything, and are blurred as a group
    # so one blur pass serves them all.
    for mesh, _, _, casts in items:
        if not casts:
            continue
        v = mesh[0]
        cx, cz = (v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 2].min() + v[:, 2].max()) / 2
        rx, rz = (np.ptp(v[:, 0]) / 2 + 0.10), (np.ptp(v[:, 2]) / 2 + 0.10)
        ring = np.array(
            [
                [cx + rx * np.cos(t), 0.0, cz + rz * np.sin(t)]
                for t in np.linspace(0, 6.2832, 24)
            ]
        )
        p, z = view.project(ring)
        if (z > 0).all():
            fdraw.polygon([tuple(q) for q in p], fill=(0, 0, 0, 105))
    shadows = shadows.filter(ImageFilter.GaussianBlur(5 * SS))
    img = Image.alpha_composite(img.convert("RGBA"), shadows).convert("RGB")

    # Grid after the shadows, so the blur does not erase it. It is the only cue for scale
    # on an empty floor, and it was invisible when the two shared a layer.
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(-14, 25):
        for a, b in (([i * 0.5, 0, -5], [i * 0.5, 0, 9]), ([-6, 0, i * 0.5], [8, 0, i * 0.5])):
            p, z = view.project(np.array([a, b], float))
            if (z > 0).all():
                draw.line([tuple(p[0]), tuple(p[1])], fill=(58, 68, 84, 190), width=SS)

    tris = []
    for mesh, key, alpha, _ in items:
        verts, faces = mesh
        p, z = view.project(verts)
        depth = z[faces].mean(axis=1)
        visible = depth > 0
        colours = shade(smooth_normals(verts, faces), PALETTE[key], depth, view)
        for face, d, colour in zip(
            faces[visible], depth[visible], colours[visible], strict=True
        ):
            tris.append((d, [tuple(p[i]) for i in face], (*(int(c) for c in colour), alpha)))

    for _, pts, colour in sorted(tris, key=lambda t: -t[0]):
        draw.polygon(pts, fill=colour)

    img = img.resize((W, H), Image.LANCZOS)
    text = ImageDraw.Draw(img)
    text.text((18, 16), "geometry/meshes.py  ·  wall / column / cabinet / table / human",
              fill=(216, 224, 236))  # fmt: skip
    text.text((18, 36), "amber disc = position uncertainty, metres on the floor",
              fill=PALETTE["disc"])  # fmt: skip
    text.text((18, 54), "teal = product extent on a surface, never counted items",
              fill=PALETTE["product"])  # fmt: skip
    out = HERE.parent / "assets" / "mesh_shop_preview.png"
    img.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
