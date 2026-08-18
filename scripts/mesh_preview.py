#!/usr/bin/env python3
"""Render a shop scene from `geometry/meshes.py`, in the style of a car's 3D map.

    python3 scripts/mesh_preview.py

Writes `assets/mesh_shop_preview.png`. The point is not the picture -- it is that the
picture is reproducible, because an asset in `assets/` with no generator is the thing this
repository keeps finding and cannot re-derive.

**This scene is hand-authored and nothing here is a model output.** The coordinates below
are typed in. It exists to show what the mesh library and the shading rig do with a scene
that has fixtures in it, which is not something the site clips currently give the panel --
`bev3d.py` renders the same meshes through the same camera and the same lights, and what
it can draw is limited by what the perception actually returns, not by this file.

**The camera and the lights are no longer here.** They were, and that was the bug: this
script grew a three-light rig, a depth fade and grouped contact shadows while
`geometry/bev3d.py` -- the renderer anybody actually looks at -- still lit its meshes with
two terms and no shadow, so the same `human()` came out as two different objects depending
on which file you ran. Both now go through `geometry/shading.py`. The move also fixed a
mirror this file had and could not have noticed: it built its camera basis as
``right = fwd x up``, which puts world +x on the *left* under this package's axes, and a
hand-authored scene has no ground truth to catch that with. So this render is mirrored
against the one committed in `f0558ef`, and the new one is the correct hand.

What still makes the look, in order of how much, and none of it is the meshes -- the
geometry here is a few hundred triangles of tubes and prisms:

1. **Smooth normals**, `geometry.meshes.smooth_normals`. Each face is shaded by the average
   of its vertices' normals, so adjacent faces differ by a little instead of a lot and a
   20-sided tube reads as a cylinder rather than a prism.
2. **Three lights, not one** -- key, fill, and a rim term that separates a dark object from
   a dark background without an outline. `geometry/shading.py` argues each one.
3. **Depth fade**, colours lerping toward the background with distance, so the scene does
   not end on a hard line.
4. **Contact shadows**, blurred as a group. Without them objects hover, and hovering is
   the artefact a viewer blames on the geometry rather than on the shading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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
from syncai_hydranet.geometry.shading import (  # noqa: E402
    View,
    contact_shadows,
    draw_scene,
)

W, H, SS = 1180, 700, 2  # SS: supersample, resolved down at the end. Same trick as bev3d.
BG = (17, 21, 28)

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


def scene() -> list[tuple]:
    """The shop, as ``(mesh, colour_key, alpha, casts_shadow)``."""
    items = []
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
    for x, z, hm, sigma, heading in (
        (-0.8, 0.6, 1.72, 0.28, 20),
        (1.3, 2.6, 1.63, 0.55, -120),
        (3.7, 1.1, 1.78, 1.0, 150),
    ):
        items.append((place(ground_disc(sigma), Placement(x, z)), "disc", 120, False))
        items.append((place(human(hm), Placement(x, z, heading_rad=np.radians(heading))),
                      "person", 255, True))  # fmt: skip
    return items


def main() -> None:
    # An elevated three-quarter view, which is the framing a car's map uses and is not a
    # taste call: from low down the fixtures occlude each other and the floor -- the one
    # surface every measurement here is expressed on -- is edge-on and unreadable.
    view = View([6.6, 5.0, -4.6], [0.35, 0.7, 2.5], 620.0 * SS, W * SS / 2, H * SS / 2)
    img = Image.new("RGB", (W * SS, H * SS), BG)
    items = scene()

    # Contact shadows go on the floor layer, under everything, and are blurred as a group
    # so one blur pass serves them all.
    img = Image.alpha_composite(
        img.convert("RGBA"),
        contact_shadows(
            img.size, view, [m for m, _, _, casts in items if casts], blur_px=5 * SS
        ),
    ).convert("RGB")

    # Grid after the shadows, so the blur does not erase it. It is the only cue for scale
    # on an empty floor, and it was invisible when the two shared a layer.
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(-14, 25):
        for a, b in (([i * 0.5, 0, -5], [i * 0.5, 0, 9]), ([-6, 0, i * 0.5], [8, 0, i * 0.5])):
            uv, depth = view.project_points(np.array([a, b], float))
            if (depth > 0).all():
                draw.line([tuple(uv[0]), tuple(uv[1])], fill=(58, 68, 84, 190), width=SS)

    draw_scene(draw, view, [(m, PALETTE[k], a) for m, k, a, _ in items], bg=BG)

    img = img.resize((W, H), Image.Resampling.LANCZOS)
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
