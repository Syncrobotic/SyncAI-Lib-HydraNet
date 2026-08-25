"""3D scene v2: fixture footprints and measured heights, from the geometry cache.

v1 projected every mask pixel through the ground plane, so an elevated counter top
smeared metres behind its true position and the boundary walls took their colour from
the smear. v2 uses what the commissioning cache already knows per pixel:

* ``lx``/``lz`` -- level-frame horizontal coordinates from the scaled DA-V2 depth, so a
  fixture pixel lands at its *footprint*, not at its ground-plane shadow.
* ``height``  -- height above the floor plane, so each structure class gets a measured
  height (its p85 over that camera's pixels) instead of a drawn constant.

Floor pixels stay on the ground-ray projection (``gx``/``gz``): they are on the plane,
where that projection is exact.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from syncai_bev3d import bev3d
from syncai_bev3d.bev import IGNORE, BevGrid, free_space_map
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.utils.visualize import TRAV_COLORS

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
CLASSES = {1: "floor", 2: "wall", 3: "column", 4: "display_table", 5: "display_shelf"}
PALETTE = np.array(
    [
        [30, 34, 44],
        [60, 200, 90],
        [150, 150, 160],
        [240, 200, 60],
        [200, 90, 200],
        [90, 140, 240],
    ],
    dtype=np.uint8,
)
GRID = BevGrid(x_min=-6.0, x_max=6.0, z_min=0.4, z_max=9.0, cell=0.03)


def bin_cells(x, z, grid):
    rows, _cols = grid.shape
    ok = np.isfinite(x) & np.isfinite(z)
    ok &= (x >= grid.x_min) & (x < grid.x_max) & (z >= grid.z_min) & (z < grid.z_max)
    r = rows - 1 - ((z[ok] - grid.z_min) / grid.cell).astype(int)
    c = ((x[ok] - grid.x_min) / grid.cell).astype(int)
    return r, c


def run(camera: str):
    cf = CameraFile.load(ROOT / f"runs/commission01/{camera}.camera.json")
    w, h = cf.image_size_px
    z = np.load(ROOT / f"runs/site30k_qa/geometry_cache/{camera}.npz")
    gx, gz, lx, lz = z["gx"], z["gz"], z["lx"], z["lz"]
    height, geom_ok = z["height"], z["geom_ok"]
    fh, fw = gx.shape

    cache = np.load(ROOT / f"runs/commission01/{camera}/structure_cache.npz")
    static = cache["static"]  # 1080p, background IGNORE
    walk = (
        np.asarray(
            Image.open(ROOT / "runs/commission01" / cf.mask_files["walkable"]).resize(
                (fw, fh), Image.Resampling.NEAREST
            )
        )
        > 127
    )

    rows, cols = GRID.shape
    votes = np.zeros((6, rows, cols), np.int32)
    r, c = bin_cells(gx[walk], gz[walk], GRID)
    np.add.at(votes[1], (r, c), 1)
    heights_m = {}
    for cid in (2, 3, 4, 5):
        sel = (static == cid) & geom_ok
        if not sel.any():
            continue
        r, c = bin_cells(lx[sel], lz[sel], GRID)
        np.add.at(votes[cid], (r, c), 1)
        hs = height[sel]
        hs = hs[np.isfinite(hs) & (hs > 0.05)]
        if len(hs) > 200:
            heights_m[cid] = float(np.clip(np.percentile(hs, 85), 0.3, 3.0))
    # a cell is the class with the most pixel votes; unseen cells stay IGNORE
    total = votes.sum(0)
    terrain = np.where(total > 0, votes.argmax(0), IGNORE).astype(np.uint8)
    # fixture footprints win over floor smear at their own cells
    trav = np.full(terrain.shape, IGNORE, np.uint8)
    trav[total > 0] = 0
    trav[terrain == 1] = 2
    fs = free_space_map(trav, GRID)

    drawn = {1: 0.0, 2: 2.4, 3: 2.4, 4: 0.75, 5: 2.0}
    bev3d.CLASS_HEIGHT_M = {**drawn, **heights_m}
    bev3d.DEFAULT_HEIGHT_M = 1.0
    hm = ", ".join(f"{CLASSES[k]} {v:.2f}m" for k, v in sorted(heights_m.items()))

    panel_h = h
    panel_w = max(int(panel_h * 0.78) // 2 * 2, 2)
    panel = bev3d.render(
        fs,
        terrain,
        GRID,
        [],
        (panel_w, panel_h),
        trav_colors=TRAV_COLORS,
        terrain_colors=PALETTE,
        class_names=("void", *CLASSES.values()),
    )

    cleanest = str(cache["cleanest"])
    plate_path = ROOT / "datasets/studioa_static" / camera / f"plate_{cleanest}.png"
    if not plate_path.exists():
        plate_path = ROOT / str(cf.plate_file)
    plate = Image.open(plate_path).convert("RGB").resize((w, h))
    base = np.asarray(plate, dtype=np.float64)
    walk_small = (
        np.asarray(Image.open(ROOT / "runs/commission01" / cf.mask_files["walkable"])) > 127
    )
    over = base.copy()
    over[walk_small] = 0.55 * base[walk_small] + 0.45 * np.array([60, 200, 90])
    view = Image.fromarray(over.astype(np.uint8))

    out = Image.new("RGB", (w + 8 + panel_w, panel_h), (7, 9, 13))
    out.paste(view, (0, 0))
    out.paste(panel, (w + 8, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([0, h - 16, w, h], fill=(0, 0, 0))
    d.text(
        (6, h - 14),
        f"{camera}  3D v2: fixture FOOTPRINTS from depth (lx/lz), measured heights p85 "
        f"[{hm}]; floor on the exact plane projection",
        fill=(230, 240, 255),
    )
    path = ROOT / f"assets/commission_scene3d_{camera}.png"
    out.save(path)
    print("saved", path, "heights:", {CLASSES[k]: round(v, 2) for k, v in heights_m.items()})


if __name__ == "__main__":
    for camera in sys.argv[1:]:
        run(camera)
