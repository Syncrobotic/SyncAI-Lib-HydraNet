"""Depth completion of the structure masks -- the teachers verify, the depth decides.

The structure vote is deliberately conservative: `wall` needs SAM3 to fire AND the b03
student to agree, and on a large plain white surface neither does -- so the STUDIO A
wall, the corridor floor and the top of a column end up unclaimed. But the depth cache
already measured every one of those pixels. This pass spends that knowledge:

* unclaimed + measured height >= TALL_M  ->  structure. The pixel inherits the nearest
  already-accepted structural class within REACH_PX (a column grows back to full height,
  a shelf's missed edge rejoins the shelf); anything further from every accepted class
  is a bare wall.
* unclaimed + |height| <= FLAT_M  ->  floor, joining the walkable mask.
* mid-height unclaimed stays unclaimed: that band is merchandise, smears and people,
  and a wrong guess there pollutes the classes that are already right.

Zero GPU: everything reads the geometry cache and the structure cache.

Usage: uv run python tools/commissioning/depth_complete.py <camera> [...]
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from syncai_hydranet.geometry.camera_json import CameraFile

# The repo root, derived rather than written out: every one of these 26 tools had it
# as an absolute path, so a second checkout ran against the first one's `runs/` and
# any machine but this one failed at import with a path and no reason. Two levels up
# from `tools/<group>/<tool>.py`, and `tests/test_no_absolute_sys_path.py` keeps it so.
ROOT = Path(__file__).resolve().parents[2]
IGNORE = 255
TALL_M = 1.35
FLAT_M = 0.08
REACH_PX = 90  # at 1080p: how far a completion pixel may look for a class to inherit
CLASS_NAMES = {1: "floor", 2: "wall", 3: "column", 4: "display_table", 5: "display_shelf"}


def run(camera: str):
    z = np.load(ROOT / f"runs/site30k_qa/geometry_cache/{camera}.npz")
    height, geom_ok = z["height"], z["geom_ok"]
    rng = np.hypot(z["gx"], z["gz"])
    tol = np.clip(0.06 + 0.035 * rng, None, 0.30)  # the recipe's own on-plane tolerance
    fh, fw = height.shape
    cf = CameraFile.load(ROOT / f"runs/commission01/{camera}.camera.json")
    w, h = cf.image_size_px
    mask_dir = ROOT / "runs/commission01"

    def load(name):
        return (
            np.asarray(
                Image.open(mask_dir / cf.mask_files[name]).resize(
                    (fw, fh), Image.Resampling.NEAREST
                )
            )
            > 127
        )

    # Baseline rebuilt from the pristine structure cache every run -- completing on top
    # of a previous completion is not idempotent (run 1 eats the wall tops, run 2 then
    # sees only short leftovers and completes nothing).
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recipe", str(ROOT / "tools/site30k/recipe.py")
    )
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    cache = np.load(ROOT / f"runs/commission01/{camera}/structure_cache.npz")
    static, floor_raw = cache["static"], cache["floor"]
    plate = (
        Image.open(ROOT / "datasets/studioa_static" / camera / f"plate_{cache['cleanest']}.png")
        .convert("RGB")
        .resize((fw, fh), Image.Resampling.LANCZOS)
    )
    floor_clip = (
        recipe.guided(
            floor_raw.astype(np.float32),
            np.asarray(plate).astype(np.float32) / 255.0,
            recipe.GUIDE_RADIUS,
            recipe.GUIDE_EPS,
        )
        >= 0.5
    )
    floor_clip = ndimage.binary_fill_holes(floor_clip)
    holes = np.zeros((fh, fw), bool)
    lbo, no = ndimage.label(static != IGNORE)
    for k in range(1, no + 1):
        comp = lbo == k
        if comp.sum() < 2000:
            continue
        holes |= ndimage.binary_fill_holes(comp) & ~comp
    floor_clip &= ~holes
    combined = np.zeros((fh, fw), np.uint8)
    for cid in (2, 3, 4, 5):
        combined[static == cid] = cid
    combined[floor_clip & (combined == 0)] = 1
    extras = np.zeros((fh, fw), bool)  # door/product keep their own layer, never overpainted
    for name in ("door", "product"):
        if name in cf.mask_files:
            extras |= load(name)

    unclaimed = (combined == 0) & ~extras & geom_ok
    # ORDER MATTERS: carve the on-plane pixels out FIRST. On Tao-Hsin-cam03 the white
    # counters, the plank floor and the wall behind them form one unclaimed component
    # whose p90 clears any height bar -- decided as a block, the floor drowns with it.
    # Floor first splits the block, and the leftover pieces get judged on their own.
    flat = unclaimed & (np.abs(height) <= tol)
    flat = ndimage.binary_opening(flat, np.ones((5, 5)))  # depth speckle is not floor
    if "floor_fill" in cf.mask_files:
        ff = load("floor_fill")
        # the SAM3 floor answer fills unclaimed pixels, and it also beats a WALL vote:
        # on Tao-Hsin-cam03 the structure pass mis-voted the entire right aisle as wall
        # (static == 2 over 100% of the plank floor). Floor is what the event layer
        # consumes; wall is cosmetic; fixtures, door and product are never overridden.
        combined[(combined == 2) & ff] = 1
        flat |= unclaimed & ff
    unclaimed &= ~flat
    # Component-level, not pixel-level: a wall's lower half measures under any pixel
    # threshold, but the component it belongs to reaches the ceiling. A merchandise
    # pile does not (p90 ~1.2 m), so the mid-band exclusion survives the change.
    tall = np.zeros_like(unclaimed)
    lab_u, n_u = ndimage.label(unclaimed, structure=np.ones((3, 3)))
    for k in range(1, n_u + 1):
        sel = lab_u == k
        if sel.sum() < 800:
            continue
        hs = height[sel]
        hs = hs[np.isfinite(hs)]
        if len(hs) > 200 and np.percentile(hs, 90) >= 1.5:
            tall |= sel

    # tall pixels inherit the nearest accepted structural class within reach
    structural = np.isin(combined, (2, 3, 4, 5))
    added = dict.fromkeys(CLASS_NAMES, 0)
    if structural.any() and tall.any():
        dist, (ir, ic) = ndimage.distance_transform_edt(~structural, return_indices=True)
        inherit = tall & (dist <= REACH_PX)
        combined[inherit] = combined[ir[inherit], ic[inherit]]
        bare = tall & (dist > REACH_PX)
        combined[bare] = 2
        for cid in (2, 3, 4, 5):
            added[cid] = int((combined[tall] == cid).sum())
    elif tall.any():
        combined[tall] = 2
        added[2] = int(tall.sum())
    combined[flat & (combined == 0)] = 1
    added[1] = int(flat.sum())

    # The wall is where the floor ends. DA-V2's depth collapses on distant white walls
    # (STUDIO A wall: true 2.4 m, measured p50 1.12 m; the back-left wall reads 0.12 m),
    # so height cannot certify them -- but the floor mask can: everything ABOVE the
    # floor's far boundary in a column, still unclaimed, is the room shell.
    floor_m = combined == 1
    col_has = floor_m.any(axis=0)
    top = np.where(col_has, floor_m.argmax(axis=0), fh)
    top = ndimage.median_filter(top, size=31)
    rows_idx = np.arange(fh)[:, None]
    above = (rows_idx < top[None, :]) & col_has[None, :]
    shell = above & (combined == 0) & ~extras
    # floor the depth still recognises must not become wall: the shell rule exists
    # for pixels the depth cannot certify, not for on-plane ground past the boundary
    shell &= ~(np.abs(height) <= tol)
    # only keep substantial components -- gaps between fixtures stay unclaimed
    lab_s, n_s = ndimage.label(shell, structure=np.ones((3, 3)))
    for k in range(1, n_s + 1):
        sel = lab_s == k
        if sel.sum() >= 4000:
            combined[sel] = 2
            added[2] += int(sel.sum())

    # drop completion crumbs: a class keeps only components that touch its original mask
    # or exceed a real size
    for cid in (2, 3, 4, 5):
        m = combined == cid
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            sel = lab == k
            if sel.sum() < 1500:
                combined[sel] = 0

    for cid, name in CLASS_NAMES.items():
        m = (combined == cid).astype(np.uint8) * 255
        small = np.asarray(Image.fromarray(m).resize((w, h), Image.Resampling.NEAREST))
        Image.fromarray(small).save(mask_dir / cf.mask_files[name])
    walk = (combined == 1).astype(np.uint8) * 255
    Image.fromarray(
        np.asarray(Image.fromarray(walk).resize((w, h), Image.Resampling.NEAREST))
    ).save(mask_dir / cf.mask_files["walkable"])

    share = {CLASS_NAMES[c]: f"+{100 * a / (fh * fw):.1f}%" for c, a in added.items() if a}
    print(f"  [{camera}] depth completion added: {share}")


def main():
    for camera in sys.argv[1:]:
        run(camera)


if __name__ == "__main__":
    main()
