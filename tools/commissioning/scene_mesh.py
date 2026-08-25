"""Solid-mesh 3D scene of a commissioned camera -- the presentation-grade view.

`scene3d.py` draws the flat free-space panel (a diagnostic). This renders the same
commissioning data as *solid geometry*: fixture footprints extruded to their measured
heights, translucent walls (only their footprint is measured -- a solid slab would
assert a height the camera never saw), the walkable floor as a tinted carpet, contact
shadows, and the metre grid. Also writes a `scene.obj` per camera so a real renderer
(Blender, Open3D, macOS preview) can take over -- `draw_scene` is a painter's
algorithm and says so in its own docstring.

Usage: uv run python tools/commissioning/scene_mesh.py <camera> [...] [--gif]
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from syncai_bev3d.meshes import Placement, _merge, extrude, ground_disc, place
from syncai_bev3d.shading import View, contact_shadows, draw_scene
from syncai_hydranet.geometry.camera_json import CameraFile

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
W, H, SS = 1180, 700, 2
BG = (17, 21, 28)
PALETTE = {
    "wall": (118, 128, 145),
    "column": (150, 160, 178),
    "display_shelf": (126, 143, 176),
    "display_table": (172, 162, 148),
    "floor": (52, 96, 66),
    "disc": (242, 180, 78),
}
CLASS_NAMES = {2: "wall", 3: "column", 4: "display_table", 5: "display_shelf"}
CELL = 0.06
DRAWN_H = {"wall": 2.4, "column": 2.4, "display_table": 0.75, "display_shelf": 2.0}


def cell_grids(camera):
    """Per-class occupancy in floor metres, and per-class measured heights (p85)."""
    cf = CameraFile.load(ROOT / f"runs/commission01/{camera}.camera.json")
    z = np.load(ROOT / f"runs/site30k_qa/geometry_cache/{camera}.npz")
    cache = np.load(ROOT / f"runs/commission01/{camera}/structure_cache.npz")
    static = cache["static"]
    fh, fw = z["gx"].shape
    walk = (
        np.asarray(
            Image.open(ROOT / "runs/commission01" / cf.mask_files["walkable"]).resize(
                (fw, fh), Image.Resampling.NEAREST
            )
        )
        > 127
    )
    xs = {1: z["gx"][walk]}
    zs = {1: z["gz"][walk]}
    heights = {}
    for cid in CLASS_NAMES:
        sel = (static == cid) & z["geom_ok"]
        xs[cid], zs[cid] = z["lx"][sel], z["lz"][sel]
        hs = z["height"][sel]
        hs = hs[np.isfinite(hs) & (hs > 0.05)]
        if len(hs) > 200:
            heights[cid] = float(np.clip(np.percentile(hs, 85), 0.3, 3.0))
    grids = {}
    for cid in xs:
        x, zz = xs[cid], zs[cid]
        ok = np.isfinite(x) & np.isfinite(zz) & (np.abs(x) < 12) & (zz > 0) & (zz < 14)
        x, zz = x[ok], zz[ok]
        g = np.zeros((int(14 / CELL), int(24 / CELL)), np.int32)
        np.add.at(g, ((zz / CELL).astype(int), ((x + 12) / CELL).astype(int)), 1)
        grids[cid] = g >= (2 if cid == 1 else 3)
    return cf, grids, heights


def rect_decompose(grid, min_cells):
    """Greedy rectangle decomposition of a cell grid -- store fixtures are boxy, and a
    set of boxes keeps the concavities a convex hull would swallow (the first render's
    shelf wall bridged the whole aisle)."""
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    for k in range(1, n + 1):
        sel = lab == k
        if sel.sum() < min_cells:
            continue
        work = ndimage.binary_closing(sel, np.ones((3, 3)))
        # A fixture that fills most of its bounding box IS a box -- render it as one.
        r, c = np.nonzero(sel)
        fill = sel.sum() / ((r.max() - r.min() + 1) * (c.max() - c.min() + 1))
        if fill > 0.55:
            yield [(r.min(), r.max(), c.min(), c.max())]
            continue
        # Coarsen 5x before decomposing: ragged edges otherwise shred the shape into
        # one-row slivers and the render turns into a picket fence.
        f = 5
        rr, cc = work.shape
        coarse = work[: rr - rr % f, : cc - cc % f].reshape(rr // f, f, cc // f, f).max((1, 3))
        rects = []
        remaining = coarse.copy()
        while remaining.any():
            r0, c0 = np.argwhere(remaining)[0]
            c1 = c0
            while c1 + 1 < remaining.shape[1] and remaining[r0, c1 + 1]:
                c1 += 1
            r1 = r0
            while r1 + 1 < remaining.shape[0] and remaining[r1 + 1, c0 : c1 + 1].all():
                r1 += 1
            remaining[r0 : r1 + 1, c0 : c1 + 1] = False
            if (r1 - r0 + 1) * (c1 - c0 + 1) >= 2:
                rects.append((r0 * f, (r1 + 1) * f - 1, c0 * f, (c1 + 1) * f - 1))
        yield rects


def floor_mesh(grid):
    """One quad per walkable cell, merged. y = 0 so the grid lines stay visible."""
    r, c = np.nonzero(grid)
    x0, z0 = c * CELL - 12, r * CELL
    verts, faces = [], []
    for i, (x, zz) in enumerate(zip(x0, z0, strict=True)):
        b = 4 * i
        verts += [
            [x, 0.004, zz],
            [x + CELL, 0.004, zz],
            [x + CELL, 0.004, zz + CELL],
            [x, 0.004, zz + CELL],
        ]
        faces += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    return np.asarray(verts, float), np.asarray(faces, int)


def build_scene(camera):
    cf, grids, heights = cell_grids(camera)
    items = []  # (mesh, colour_key, alpha, casts_shadow)
    items.append((floor_mesh(grids[1]), "floor", 150, False))
    for cid, name in CLASS_NAMES.items():
        h = heights.get(cid, DRAWN_H[name])
        for rects in rect_decompose(grids[cid], min_cells=60):
            solid = name != "wall"
            for r0, r1, c0, c1 in rects:
                x0, x1 = c0 * CELL - 12, (c1 + 1) * CELL - 12
                z0, z1 = r0 * CELL, (r1 + 1) * CELL
                poly = np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]])
                mesh = extrude(poly, h if solid else 2.4)
                items.append((mesh, name, 255 if solid else 105, solid))
    items.append((place(ground_disc(0.35), Placement(0.0, 0.0)), "disc", 130, False))
    return cf, items, heights


def render(camera, items, heights, out_path, *, eye=None, target=None):
    xs = np.concatenate([m[0][:, 0] for m, *_ in items])
    zs = np.concatenate([m[0][:, 2] for m, *_ in items])
    cx_m, cz_m = float(np.median(xs)), float(np.median(zs))
    target = target or [cx_m, 0.6, cz_m]
    eye = eye or [cx_m + 7.5, 5.6, cz_m - 6.5]
    view = View(eye, target, 620.0 * SS, W * SS / 2, H * SS / 2)
    img = Image.new("RGB", (W * SS, H * SS), BG)
    img = Image.alpha_composite(
        img.convert("RGBA"),
        contact_shadows(img.size, view, [m for m, _, _, s in items if s], blur_px=5 * SS),
    ).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(-24, 25):
        for a, b in (
            ([i * 0.5, 0, 0], [i * 0.5, 0, 14]),
            ([-12, 0, i * 0.5 + 7], [12, 0, i * 0.5 + 7]),
        ):
            uv, depth = view.project_points(np.array([a, b], float))
            if (depth > 0).all():
                draw.line([tuple(uv[0]), tuple(uv[1])], fill=(58, 68, 84, 150), width=SS)
    draw_scene(draw, view, [(m, PALETTE[k], a) for m, k, a, _ in items], bg=BG)
    img = img.resize((W, H), Image.Resampling.LANCZOS)
    text = ImageDraw.Draw(img)
    hm = "  ".join(
        f"{CLASS_NAMES[k].replace('display_', '')} {v:.2f}m" for k, v in sorted(heights.items())
    )
    text.text(
        (18, 16),
        f"{camera}  ·  commissioning mesh: footprints extruded to measured heights",
        fill=(216, 224, 236),
    )
    text.text(
        (18, 36),
        f"measured p85: {hm}  |  walls translucent (drawn 2.4 m), amber disc = camera point",
        fill=(170, 182, 200),
    )
    img.save(out_path)
    return view


def export_obj(camera, items):
    solids = [m for m, k, a, _ in items if a == 255]
    walls = [m for m, k, a, _ in items if a == 105]
    from syncai_bev3d.meshes import to_obj

    obj = to_obj(_merge(*solids, *walls), name=camera)
    path = ROOT / "runs/commission01" / camera / "scene.obj"
    path.write_text(obj)
    return path


def main():
    argv = [a for a in sys.argv[1:] if a != "--gif"]
    gif = "--gif" in sys.argv[1:]
    for camera in argv:
        _cf, items, heights = build_scene(camera)
        out = ROOT / f"assets/commission_mesh_{camera}.png"
        render(camera, items, heights, out)
        obj = export_obj(camera, items)
        print(f"{camera}: {out.name}, {obj}")
        if gif:
            xs = np.concatenate([m[0][:, 0] for m, *_ in items])
            zs = np.concatenate([m[0][:, 2] for m, *_ in items])
            cx_m, cz_m = float(np.median(xs)), float(np.median(zs))
            frames = []
            for ang in np.linspace(0, 2 * np.pi, 28, endpoint=False):
                eye = [cx_m + 9.2 * np.cos(ang), 5.8, cz_m + 9.2 * np.sin(ang)]
                tmp = ROOT / "assets" / "_orbit_tmp.png"
                render(camera, items, heights, tmp, eye=eye, target=[cx_m, 0.5, cz_m])
                frames.append(Image.open(tmp).convert("P", palette=Image.ADAPTIVE))
            tmp.unlink()
            gif_path = ROOT / f"assets/commission_mesh_{camera}_orbit.gif"
            frames[0].save(
                gif_path, save_all=True, append_images=frames[1:], duration=140, loop=0
            )
            print(f"{camera}: {gif_path.name}")


if __name__ == "__main__":
    main()
