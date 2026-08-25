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

from syncai_bev3d.meshes import (
    Placement,
    _merge,
    cabinet,
    column,
    extrude,
    ground_disc,
    place,
    table,
    wall,
)
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
    "door": (176, 126, 88),
    "product": (86, 214, 188),
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
    # extras: door footprints, and product cells with the height merchandise sits at
    hts = {}
    for name, cid in (("door", 6), ("product", 7)):
        f = ROOT / "runs/commission01" / camera / "masks" / f"{name}.png"
        if not f.exists():
            continue
        m = np.asarray(Image.open(f).resize((fw, fh), Image.Resampling.NEAREST)) > 127
        sel = m & z["geom_ok"]
        xs[cid], zs[cid] = z["lx"][sel], z["lz"][sel]
        if name == "product":
            hts[cid] = z["height"][sel]
    grids = {}
    grid_h = {}
    for cid in xs:
        x, zz = xs[cid], zs[cid]
        ok = (
            np.isfinite(x)
            & np.isfinite(zz)
            & (np.abs(x) < 12 - CELL)
            & (zz > 0)
            & (zz < 14 - CELL)
        )
        x, zz = x[ok], zz[ok]
        g = np.zeros((int(14 / CELL), int(24 / CELL)), np.int32)
        rows, cols = (zz / CELL).astype(int), ((x + 12) / CELL).astype(int)
        np.add.at(g, (rows, cols), 1)
        grids[cid] = g >= (2 if cid == 1 else 3)
        if cid in hts:
            hh = hts[cid][ok]  # heights follow the exact same filter as the coordinates
            hsum = np.zeros_like(g, float)
            np.add.at(hsum, (rows, cols), np.nan_to_num(hh))
            grid_h[cid] = hsum / np.maximum(g, 1)
    return cf, grids, heights, grid_h


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
    cf, grids, heights, _grid_h = cell_grids(camera)
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


def store_yaw(grids):
    """Dominant fixture orientation, folded mod 90 deg -- the store's own axis."""
    angles, weights = [], []
    for cid in (2, 5, 4):
        lab, n = ndimage.label(grids[cid], structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            r, c = np.nonzero(lab == k)
            if len(r) < 120:
                continue
            pts = np.stack([c.astype(float), r.astype(float)], 1)
            pts -= pts.mean(0)
            _, s_, vt = np.linalg.svd(pts, full_matrices=False)
            if s_[0] < 2 * s_[1]:
                continue  # not elongated: no orientation vote
            angles.append(np.arctan2(vt[0, 1], vt[0, 0]) % (np.pi / 2))
            weights.append(len(r))
    if not angles:
        return 0.0
    # circular mean on the quadrant-folded angle
    a = np.array(angles) * 4
    m = (
        np.arctan2(
            np.average(np.sin(a), weights=weights), np.average(np.cos(a), weights=weights)
        )
        / 4
    )
    return float(m % (np.pi / 2))


def build_scene_regular(camera):
    """B-path: every fixture becomes a store-axis-aligned parametric mesh.

    The depth-derived footprints are ragged; the furniture is not. Each component is
    fitted with a robust axis-aligned box in the STORE frame (p3-p97 extents, snapped to
    5 cm) and rendered as the parametric mesh its class names: cabinets with shelf
    slabs, tables with legs, thin walls, columns.
    """
    cf, grids, heights, grid_h = cell_grids(camera)
    yaw = store_yaw(grids)
    cy, sy = np.cos(yaw), np.sin(yaw)
    items = [(floor_mesh(grids[1]), "floor", 150, False)]
    for cid, name in CLASS_NAMES.items():
        h = heights.get(cid, DRAWN_H[name])
        lab, n = ndimage.label(grids[cid], structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            r, c = np.nonzero(lab == k)
            if len(r) < 60:
                continue
            x = c * CELL - 12 + CELL / 2
            z = r * CELL + CELL / 2
            # into the store frame, robust extents, snap to 5 cm
            u = x * cy + z * sy
            v = -x * sy + z * cy
            u0, u1 = np.percentile(u, [3, 97])
            v0, v1 = np.percentile(v, [3, 97])
            w = max(round((u1 - u0) / 0.05) * 0.05, 0.3)
            d = max(round((v1 - v0) / 0.05) * 0.05, 0.3)
            um, vm = (u0 + u1) / 2, (v0 + v1) / 2
            px, pz = um * cy - vm * sy, um * sy + vm * cy
            at = Placement(px, pz, heading_rad=-yaw)
            if name == "wall":
                short = min(w, d)
                if short > 0.4:  # a fat "wall" blob is a wall corner: thin it
                    w, d = (w, 0.15) if w >= d else (0.15, d)
                half = (w if w >= d else d) / 2
                pts = [[-half, 0.0], [half, 0.0]] if w >= d else [[0.0, -half], [0.0, half]]
                mesh = wall(pts, 2.4, thickness_m=max(min(w, d), 0.12))
                items.append((place(mesh, at), name, 105, False))
            elif name == "column":
                mesh = column(min(w, 0.8), min(d, 0.8), max(h, 2.2))
                items.append((place(mesh, at), name, 255, True))
            elif name == "display_shelf":
                mesh = cabinet(w, min(d, 0.8), h, shelves=max(2, int(h / 0.45)))
                items.append((place(mesh, at), name, 255, True))
            else:  # display_table
                mesh = (
                    table(w, d, h)
                    if max(w, d) < 2.2
                    else extrude(
                        np.array(
                            [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]]
                        ),
                        h,
                    )
                )
                items.append((place(mesh, at), name, 255, True))
    # extras: doors as solid tall slabs, products as slabs at their measured height
    cy2, sy2 = np.cos(yaw), np.sin(yaw)
    for cid, name, hgt in ((6, "door", 2.05), (7, "product", 0.15)):
        if cid not in grids:
            continue
        lab, n = ndimage.label(grids[cid], structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            r, c = np.nonzero(lab == k)
            if len(r) < (20 if name == "product" else 40):
                continue
            x = c * CELL - 12 + CELL / 2
            zz2 = r * CELL + CELL / 2
            u = x * cy2 + zz2 * sy2
            v = -x * sy2 + zz2 * cy2
            u0, u1 = np.percentile(u, [3, 97])
            v0, v1 = np.percentile(v, [3, 97])
            w = max(round((u1 - u0) / 0.05) * 0.05, 0.2)
            d = max(round((v1 - v0) / 0.05) * 0.05, 0.2)
            um, vm = (u0 + u1) / 2, (v0 + v1) / 2
            at = Placement(um * cy2 - vm * sy2, um * sy2 + vm * cy2, heading_rad=-yaw)
            if name == "door":
                short = min(w, d)
                if short > 0.3:
                    w, d = (w, 0.12) if w >= d else (0.12, d)
                mesh = extrude(
                    np.array(
                        [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]]
                    ),
                    hgt,
                )
                items.append((place(mesh, at), name, 255, True))
            else:
                base = float(np.nanmedian(grid_h[cid][lab == k])) if cid in grid_h else 0.9
                base = float(np.clip(base, 0.1, 2.0))
                slab = extrude(
                    np.array(
                        [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]]
                    ),
                    hgt,
                )
                verts = slab[0].copy()
                verts[:, 1] += base
                items.append((place((verts, slab[1]), at), name, 255, False))
    items.append((place(ground_disc(0.35), Placement(0.0, 0.0)), "disc", 130, False))
    return cf, items, heights


def export_glb(camera, items):
    """A-path: the same scene as a GLB any glTF viewer can orbit."""
    import trimesh

    scene = trimesh.Scene()
    for i, (mesh, key, alpha, _s) in enumerate(items):
        verts, faces = mesh
        if len(faces) == 0:
            continue
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        rgba = [*PALETTE[key], alpha]
        tm.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[c / 255 for c in rgba],
                metallicFactor=0.05,
                roughnessFactor=0.85,
                alphaMode="BLEND" if alpha < 255 else "OPAQUE",
                doubleSided=True,
            )
        )
        scene.add_geometry(tm, node_name=f"{key}_{i}")
    path = ROOT / "runs/commission01" / camera / "scene.glb"
    scene.export(path)
    return path


def export_obj(camera, items):
    solids = [m for m, k, a, _ in items if a == 255]
    walls = [m for m, k, a, _ in items if a == 105]
    from syncai_bev3d.meshes import to_obj

    obj = to_obj(_merge(*solids, *walls), name=camera)
    path = ROOT / "runs/commission01" / camera / "scene.obj"
    path.write_text(obj)
    return path


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    gif = "--gif" in sys.argv[1:]
    regular = "--regular" in sys.argv[1:]
    for camera in argv:
        _cf, items, heights = (build_scene_regular if regular else build_scene)(camera)
        out = ROOT / f"assets/commission_mesh_{camera}.png"
        render(camera, items, heights, out)
        obj = export_obj(camera, items)
        glb = export_glb(camera, items)
        print(f"{camera}: {out.name}, {obj.name}, {glb.name}")
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
