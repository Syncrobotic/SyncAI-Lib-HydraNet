"""Solid-mesh 3D scene of a commissioned camera -- the presentation-grade view.

`tools/commissioning/scene3d.py` draws the flat free-space panel (a diagnostic). This
builds the same commissioning data as *solid geometry*: fixture footprints extruded to
their measured heights, translucent walls (only their footprint is measured -- a solid
slab would assert a height the camera never saw), the walkable floor as a tinted carpet,
contact shadows, and the metre grid.

**This lived in `tools/commissioning/scene_mesh.py` until 2026-08-29, and moved because
four tools import it.** `demo_video`, `heads_video`, `scene_overlay` and `social_card`
all reach for `build_scene_regular`, `render` and `PALETTE`; the fourth of them turned
`tests/test_scripts_are_not_libraries.py` red, which is that test working. The cost it
names is the one that applies here word for word: under `tools/` this file sat outside
the wheel, outside the type ratchet and outside the coverage floor, so the module every
renderer in the project depends on was the one nothing checked -- 13% of its statements,
and its whole scene build had never been executed by a test. The CLI stays behind in
`tools/commissioning/scene_mesh.py`, which is what a script is for.

The commissioning inputs are read from `root`, defaulting to the checkout this package
sits in. They are `runs/` artefacts, which is deliberately outside the wheel: an
installed copy has no commissioning data, and a test passes a `tmp_path` instead.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from syncai_bev3d.floorplan import (
    FLOOR_BOTH_SIDES,
    floor_both_sides,
    resolve_overlaps,
    snap_to_walls,
    wall_runs,
)
from syncai_bev3d.meshes import (
    Placement,
    _merge,
    box,
    column,
    extrude,
    ground_disc,
    place,
    shelf_levels,
    shelving,
    table,
    wall,
)
from syncai_bev3d.shading import View, contact_shadows, draw_scene, occlusion_alpha
from syncai_hydranet.geometry.camera_json import CameraFile

# The checkout this package sits in -- `src/syncai_bev3d/scene.py` -> the repo root --
# rather than the absolute path this file carried while it lived under `tools/`. Every
# reader of it is commissioning data under `runs/`, which no installed wheel has, and
# `cell_grids` takes a `root` argument so a test can hand it a `tmp_path` store.
ROOT = Path(os.environ.get("SYNCAI_ROOT", Path(__file__).resolve().parents[2]))
# Annotated `int` rather than left to inference, because callers **rewrite them**:
# `social_card` renders at the panel's own aspect, and every video tool drops the
# supersample to 1. Left inferred they are `Literal[1180] / Literal[700] / Literal[2]`
# and every one of those assignments is a type error -- which is what the type ratchet
# said the moment this module moved somewhere a checker could resolve it.
W: int = 1180
H: int = 700
SS: int = 2
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
    "product_boxed_stock": (190, 90, 235),
    "product_macbook": (60, 150, 255),
    "product_ipad": (255, 150, 0),
    "product_iphone": (255, 70, 70),
}
CLASS_NAMES = {2: "wall", 3: "column", 4: "display_table", 5: "display_shelf"}
WALL_CID = 2  # named because the wall runs read the class mask directly, not by name
# **Measured and rejected 2026-08-29: cutting a fixture mask at its class's plausible
# ceiling.** The diagnosis was right and the remedy was not. Single-view metrology on the
# component tops does identify the welding PLAN 7.21 measured -- the largest
# `display_table` component reads 2.44 m on Taichung-cam01, 1.90 m on cam11, 2.09 m on
# Kaohsiung-cam04, and it is the *largest* component every time, which is what welding
# produces. But clearing the mask above the row 1.30 m projects to deleted real furniture
# with it: display tables went 5 -> 1 on Taichung-cam01 and to **zero** on Kaohsiung-cam04
# and Tao-Hsin-cam03. `geometry.pixel_row_at_height` stays, because the solve is right and
# tested; what is missing is a base position per component that a welded mask does not
# corrupt, and the component's lowest rows are not it.
CELL = 0.06
DRAWN_H = {"wall": 2.4, "column": 2.4, "display_table": 0.75, "display_shelf": 2.0}
# A column runs floor to ceiling. The depth model reads 1.07-1.65 m for one (see the
# block below), so the drawn height is floored here rather than taken from the depth.
COLUMN_MIN_H = 2.2
# The SAM3 vote puts real furniture in the wall mask -- on Taichung-cam10 the back
# service counter and a stock trolley are both `wall`. Height looked like the way to
# separate them and IS NOT: measured per component, all six wall components on this
# camera come out 1.07-1.65 m p85, so any height rule reclassifies every wall in the
# store. That is not a mask defect, it is the known one in the depth model -- DA-V2
# collapses on white walls, a 2.4 m wall reading 1.12 m -- which is exactly why the
# walls here are drawn at a constant DRAWN_H and only their footprint is claimed as
# measured. The fix belongs in `masks_pass.py`, where the class is decided.


# --- counters and merchandise ---------------------------------------------------------
# A footprint wider than `table()`'s four-leg model suits was previously extruded into a
# solid prism: a measured outline with no top, no plinth and no way to read it as a
# surface things sit on. A retail counter is a slab on a recessed body, and the recess is
# the feature that makes it read as a counter from any angle.
COUNTER_TOP_T = 0.05
COUNTER_INSET = 0.06

# Merchandise footprints are measured per *region*; the individual items tiled into a
# region are SCHEMATIC and the panel says so. Sizes are the real products' own.
PRODUCT_UNIT_M = {
    "product_macbook": (0.33, 0.23, 0.02),
    "product_ipad": (0.25, 0.18, 0.012),
    "product_iphone": (0.075, 0.15, 0.009),
    "product_boxed_stock": (0.30, 0.20, 0.16),
    "product": (0.20, 0.15, 0.05),
}
PRODUCT_GAP_M = 0.06
PRODUCT_MAX_PER_REGION = 28


def counter(width_m: float, depth_m: float, height_m: float):
    """A top slab on a recessed body -- the shape a long display counter actually is."""
    top = box(width_m, COUNTER_TOP_T, depth_m)
    top = (top[0] + [0, height_m - COUNTER_TOP_T, 0], top[1])
    body_w = max(width_m - 2 * COUNTER_INSET, width_m * 0.5)
    body_d = max(depth_m - 2 * COUNTER_INSET, depth_m * 0.5)
    body = box(body_w, height_m - COUNTER_TOP_T, body_d)
    return _merge(top, body)


def _laptop(w: float, d: float, t: float):
    """Base slab plus a lid standing at the back: the silhouette that reads as a laptop."""
    base = box(w, t * 2.5, d)
    lid = box(w, d * 0.92, t * 2.0)
    lid = (lid[0] + [0, t * 2.5, -d / 2 + t], lid[1])
    return _merge(base, lid)


def _slab(w: float, d: float, t: float):
    return box(w, max(t, 0.008), d)


def product_units(name: str, w_m: float, d_m: float):
    """Tile a measured merchandise region with unit-sized items, centred.

    The region is measured; the tiling is a schematic of what sits in it. Returning the
    placements rather than one merged blob keeps that distinction available to the caller
    -- and to the label it has to write.
    """
    uw, ud, ut = PRODUCT_UNIT_M.get(name, PRODUCT_UNIT_M["product"])
    step_x, step_z = uw + PRODUCT_GAP_M, ud + PRODUCT_GAP_M
    nx = max(1, int(w_m // step_x))
    nz = max(1, int(d_m // step_z))
    while nx * nz > PRODUCT_MAX_PER_REGION and max(nx, nz) > 1:
        if nx >= nz:
            nx -= 1
        else:
            nz -= 1
    make = (
        _laptop
        if name == "product_macbook"
        else ((lambda a, b, c: box(a, c, b)) if name == "product_boxed_stock" else _slab)
    )
    out = []
    for i in range(nx):
        for j in range(nz):
            ox = (i - (nx - 1) / 2) * step_x
            oz = (j - (nz - 1) / 2) * step_z
            mesh = make(uw, ud, ut)
            out.append(((mesh[0] + [ox, 0.0, oz]), mesh[1]))
    return out


# **The estimator is chosen per class, and p85 for everything was the bug.**
#
# p85 answers "85% of what I saw of this thing is below here". For a fixture with a flat
# top -- a display table -- that IS its height, and the fleet agrees: 0.78-0.97 m across
# eight cameras, tight and right. For a surface that runs from the floor to its top --
# a wall, a column, a merchandise shelf -- it answers how much of it the camera saw, and
# under-reads by however much is above the 85th percentile of visible pixels.
#
# Measured 2026-08-27, `display_shelf` p85 against p99 on the same pixels:
#
#     Taichung-cam01   1.69 -> 2.08     Taichung-cam10   1.82 -> 2.38
#     Taichung-cam11   1.73 -> 2.04     Tao-Hsin-cam04   1.17 -> 1.32
#     Kaohsiung-cam04  0.93 -> 1.21
#
# The first three land on believable merchandise-wall heights, so those cameras were
# never a depth failure -- they were this estimator. Kaohsiung-cam04 is low at p99 too,
# and that one **is** the white-surface collapse the block above `CLASS_NAMES` describes.
# Separating the two is the whole point of changing the estimator: with p85 everywhere,
# a wrong estimator and a broken depth model produced the same symptom and nothing could
# tell them apart.
#
# p99 rather than max: a mask's top row picks up the ceiling or a smear, and one bad
# pixel would set the height of the fixture.
HEIGHT_PCT = {2: 99, 3: 99, 4: 85, 5: 99}


def cell_grids(camera, root: Path | None = None):
    """Per-class occupancy in floor metres, and per-class measured heights (p85).

    `root` is the checkout holding `runs/`; the default is this package's own. It is an
    argument rather than a module global because it is the only thing between this
    function and a test -- everything else it needs is a numpy array on disk.
    """
    root = Path(root) if root is not None else ROOT
    cf = CameraFile.load(root / f"runs/commission01/{camera}.camera.json")
    z = np.load(root / f"runs/site30k_qa/geometry_cache/{camera}.npz")
    fh, fw = z["gx"].shape
    # The mask PNGs are the artefact of record -- depth completion and the human zone
    # stamps land there, not in the intermediate cache. Rebuild the class map from them.
    static = np.full((fh, fw), 255, np.uint8)
    cf_early = CameraFile.load(root / f"runs/commission01/{camera}.camera.json")
    for cid, cname in ((2, "wall"), (3, "column"), (4, "display_table"), (5, "display_shelf")):
        f = cf_early.mask_files.get(cname)
        if f and (root / "runs/commission01" / f).exists():
            m = (
                np.asarray(
                    Image.open(root / "runs/commission01" / f).resize(
                        (fw, fh), Image.Resampling.NEAREST
                    )
                )
                > 127
            )
            static[m] = cid
    walk = (
        np.asarray(
            Image.open(root / "runs/commission01" / cf.mask_files["walkable"]).resize(
                (fw, fh), Image.Resampling.NEAREST
            )
        )
        > 127
    )
    xs = {1: z["gx"][walk]}
    zs = {1: z["gz"][walk]}
    heights = {}
    hts = {}
    for cid in CLASS_NAMES:
        sel = (static == cid) & z["geom_ok"]
        xs[cid], zs[cid] = z["lx"][sel], z["lz"][sel]
        hts[cid] = z["height"][sel]  # per cell, so a component can be measured too
        hs = z["height"][sel]
        hs = hs[np.isfinite(hs) & (hs > 0.05)]
        if len(hs) > 200:
            heights[cid] = float(np.clip(np.percentile(hs, HEIGHT_PCT[cid]), 0.3, 3.0))
    # extras: door footprints, and product cells with the height merchandise sits at
    for name, cid in (
        ("door", 6),
        ("product", 7),
        ("product_boxed_stock", 8),
        ("product_macbook", 9),
        ("product_ipad", 10),
        ("product_iphone", 11),
    ):
        f = root / "runs/commission01" / camera / "masks" / f"{name}.png"
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


def build_scene(camera, root: Path | None = None):
    cf, grids, heights, _grid_h = cell_grids(camera, root)
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


def height_caption(heights: dict[int, float]) -> str:
    """The second caption line: what was measured, and what was drawn to a convention.

    **`wall` and `column` must never appear as measured heights.** They are drawn at
    `DRAWN_H["wall"]` and `COLUMN_MIN_H`, for the reason stated above `CLASS_NAMES`:
    DA-V2 collapses on white surfaces, so every wall component on Taichung-cam10
    measures 1.07-1.65 m against a real 2.4 and Taichung-cam01's column reads 1.16 m for
    something that runs floor to ceiling. This line used to print those numbers as
    "measured p85" beside a picture drawn at the constants -- a caption asserting a
    height the render did not use and the code already knew was wrong.

    They still appear, because deleting a measurement is not the same as labelling it:
    the collapse is a real property of the depth model on this fleet and a reader who
    sees it named is a reader who will not re-derive it. It is labelled as what it is.
    """
    drawn = {k for k, v in CLASS_NAMES.items() if v in ("wall", "column")}

    def part(keys):
        return "  ".join(
            f"{CLASS_NAMES[k].replace('display_', '')} {v:.2f}m(p{HEIGHT_PCT[k]})"
            for k, v in sorted(heights.items())
            if k in keys
        )

    measured = part(set(CLASS_NAMES) - drawn)
    seen = part(drawn)
    line = (
        f"measured: {measured}  |  wall drawn at {DRAWN_H['wall']:.1f} m and column "
        f"at {COLUMN_MIN_H:.1f} m min -- footprint only"
    )
    return line + (f"; depth saw {seen}, which is the white-surface collapse" if seen else "")


def render(camera, items, heights, out_path, *, eye=None, target=None, shapes=()):
    # The *furniture* sets the frame, not the floor. The walkable carpet is drawn out to
    # the edge of the projectable area, so including it made the span the room's reach
    # rather than the room's contents and every scene came out small and far away.
    solid = [m for m, name, _a, _s in items if name != "floor"] or [m for m, *_ in items]
    xs = np.concatenate([m[0][:, 0] for m in solid])
    zs = np.concatenate([m[0][:, 2] for m in solid])
    cx_m, cz_m = float(np.median(xs)), float(np.median(zs))
    # **Frame the scene, do not frame a fixed distance.** The eye used to sit at a
    # constant offset whatever the room measured, so a 6 m shop filled the canvas and a
    # 12 m one sat in the middle of it at a third of the width -- and an unreadable render
    # is not a neutral cost. On 2026-08-27 a reviewer reading these images called a
    # foreshortened 1 cm iPad slab a "45-degree tilted panel, physically impossible" and
    # correct product placement "floating"; both readings were of the framing, not of the
    # reconstruction. Distance scales with the scene's own diagonal.
    span = max(float(xs.max() - xs.min()), float(zs.max() - zs.min()), 2.0)
    cx_m = float((xs.min() + xs.max()) / 2)
    cz_m = float((zs.min() + zs.max()) / 2)
    target = target or [cx_m, 0.6, cz_m]
    # A fixed (+x, -z) diagonal, and it has a defect this session found and did not
    # fix: nothing asks whether something tall lies along it. Taichung-cam10's accessory
    # wall is 6.3 m long and 2.4 m high against the +x side, so the eye stands behind it
    # and that render is one blank panel with the shop hidden. Choosing among the four
    # diagonals by how much tall geometry each one has to see through was tried and
    # reverted the same hour: it fixes cam10 and makes cam11 -- the camera this render
    # was composed on -- markedly worse, so picking a corner is a composition decision
    # rather than a scoring one. The defect was invisible until the shelf-depth bug
    # below stopped truncating that wall to 1.35 m.
    eye = eye or [cx_m + 0.62 * span, 0.46 * span, cz_m - 0.54 * span]
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
    # Fade whatever stands between the eye and the room. PLAN 7.22 left this open, and
    # the alternative -- picking a different corner -- was tried and reverted because it
    # fixes one camera and breaks another. `occlusion_alpha` leaves the composition alone.
    painted = [(m, PALETTE[k], a) for m, k, a, _ in items]
    faded = occlusion_alpha(view, painted, keep=(PALETTE["floor"],))
    draw_scene(
        draw, view, [(m, c, a) for (m, c, _), a in zip(painted, faded, strict=True)], bg=BG
    )
    img = img.resize((W, H), Image.Resampling.LANCZOS)
    text = ImageDraw.Draw(img)
    text.text(
        (18, 16),
        f"{camera}  ·  commissioning mesh: footprints measured, heights where the depth holds",
        fill=(216, 224, 236),
    )
    text.text((18, 36), height_caption(heights), fill=(170, 182, 200))
    # The third line only exists when there is something to say. `shapes` defaults to
    # empty so a caller that does not pass it is unchanged, and a clean scene that does
    # pass it renders identically to one that does not.
    said = implausible_caption(shapes)
    if said:
        text.text((18, 56), said, fill=(236, 168, 96))
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


# **Plausibility intervals per class, in metres.** Not a filter and not a fit: a
# component outside its own class's interval is a component the mask welded together or
# the depth got wrong, and the point is to *say so* rather than to draw it as if it were
# a fixture. A display table 3.5 m deep and 0.97 m high is two tables joined by a mask
# bridge; drawn silently it reads as one enormous table, and every number taken off it
# afterwards is wrong in a way the picture does not show.
#
# The intervals are shop facts, and they are wide on purpose -- this is a tripwire for
# "that is not furniture", not a shape prior that pulls a fitted box toward a mean.
# `span` is the longer horizontal side, `short` the other.
PLAUSIBLE_M = {
    "display_table": {"height": (0.55, 1.15), "span": (0.4, 3.2), "short": (0.3, 1.6)},
    "display_shelf": {"height": (0.90, 2.60), "span": (0.4, 9.0), "short": (0.2, 1.4)},
    "column": {"height": (1.60, 3.20), "span": (0.15, 1.30), "short": (0.15, 1.30)},
    "wall": {"height": (1.80, 3.20), "span": (0.3, 20.0), "short": (0.05, 0.6)},
    # A door, including a double one. Tao-Hsin-cam04 builds a 3.7 m "door" and draws it
    # as a solid slab that dominates the render: that is the shopfront glazing, not a
    # door, and it went unreported because this table had no entry for the class. What it
    # should be *drawn* as is a separate decision -- glass is not opaque either -- and
    # this only stops it being drawn silently.
    "door": {"height": (1.90, 2.60), "span": (0.6, 2.4), "short": (0.05, 0.4)},
}

# How far above a fixture's top a product may sit and still count as resting on it. A
# cabinet's merchandise sits on shelves *inside* it, so the test is "inside the footprint
# and not floating above the top", not "level with the top".
SUPPORT_TOL_M = 0.25

# A wall-mounted merchandise run in these shops stands about 0.4-0.45 m off the wall,
# read off the plates. The cap was 0.8 m, inherited from the column code next to it, and
# it made every accessory wall twice as deep as the one in the picture.
SHELF_MAX_DEPTH_M = 0.45


def footprint_of(mesh):
    """World-axis-aligned `(x0, x1, z0, z1, top_y)` of one emitted mesh.

    Axis-aligned rather than the true rotated rectangle, and that is a real
    approximation: at the store yaw a rotated box's AABB is larger than the box. It is
    used only for the support test, where over-covering means a product is *kept* rather
    than dropped -- the direction that does not delete geometry on a technicality.
    """
    v = mesh[0]
    return (
        float(v[:, 0].min()),
        float(v[:, 0].max()),
        float(v[:, 2].min()),
        float(v[:, 2].max()),
        float(v[:, 1].max()),
    )


def implausible(shapes) -> list[str]:
    """Every emitted fixture outside its class's interval, named.

    `shapes` is `(name, width_m, depth_m, height_m)` **as the fixture was built**, in the
    store frame. It is not derived from the mesh, and that is the whole reason this takes
    an argument instead of reading `items`: a first version measured the world
    axis-aligned bound of the placed mesh, and at a 37-degree store yaw the AABB of a
    2.59 x 2.46 m counter is 3.55 x 3.53 -- so every rotated fixture in the shop reported
    as implausible, and a 0.15 m wall came back 1.60 m thick. The same confusion had
    already produced a false "footprints are systematically too large" finding earlier
    the same day. A box's dimensions are its own, not its shadow's.

    Returned rather than raised, and rather than silently skipped: this is a
    commissioning render whose job is to be looked at, and a component that is not
    furniture is exactly what the person looking needs told.
    """
    out = []
    for name, w, d, h in shapes:
        rule = PLAUSIBLE_M.get(name)
        if rule is None:
            continue
        for label, value in (("height", h), ("span", max(w, d)), ("short", min(w, d))):
            lo, hi = rule[label]
            if not (lo <= value <= hi):
                out.append(
                    f"{name}: {label} {value:.2f} m outside {lo:.2f}-{hi:.2f} "
                    f"(built {max(w, d):.2f}x{min(w, d):.2f} m, height {h:.2f} m)"
                )
    return out


def not_furniture(shapes) -> list[tuple]:
    """The subset of `shapes` `implausible` objects to, as data rather than as sentences.

    Filtered through `implausible` itself rather than by re-reading `PLAUSIBLE_M`: a
    second copy of the rule is the failure this repository keeps paying for, and the
    caption below has to name exactly what the printed line names or the picture and the
    scrollback disagree about the same scene.
    """
    return [s for s in shapes if implausible([s])]


def implausible_caption(shapes) -> str:
    """One line for the picture, naming what the render draws and does not believe.

    **Empty when there is nothing to say, and every caller draws nothing then**, so a
    scene with no defect renders to the same pixels it did before this existed -- which
    is what let the two published figures be re-cut byte-identical.

    It exists because `implausible` had no consumer for the whole of its life. It was
    called in one CLI's `main()` and printed to a terminal; the four tools that produce
    every published figure -- `demo_video`, `heads_video`, `scene_overlay`,
    `social_card` -- discarded the shapes and drew the fixture anyway. Measured across
    the fleet 2026-08-29 (PLAN 7.27): 7 fixtures on 4 of the 8 commissioned cameras, all
    of them shipped in pictures that said nothing about them.

    The dimensions are the informative part and the interval is not: a reader who sees
    `display_table 2.60x2.45 m` knows it is two counters welded by a mask without being
    told what a table may measure. Three at most, because this goes on an image.
    """
    bad = not_furniture(shapes)
    if not bad:
        return ""
    shown = ", ".join(f"{n} {max(w, d):.2f}x{min(w, d):.2f} m" for n, w, d, _h in bad[:3])
    more = f", +{len(bad) - 3} more" if len(bad) > 3 else ""
    return f"drawn and not believed -- outside its class interval: {shown}{more}"


# **Walls: three attempts, none of them shippable, and the record is the value.**
#
# Fitting walls to the *floor's* boundary instead of to their own mask components is
# still the right idea -- floor pixels are on the ground plane where the projection is
# exact, and a wall's own pixels come through a depth model this fleet has measured
# collapsing on white surfaces. Measured 2026-08-28: 15 of 17 `wall` components across
# three cameras already sat within 0.37 m of that boundary, so the components are in the
# right *place*; there are simply 8 to 14 of them per wall.
#
# What failed was every way tried of turning boundary cells into lines:
#
#   free RANSAC        a line only has to pass *near* cells, so one diagonal collects
#                      cells from three walls with metres of floor between them and
#                      outvotes every real wall. Taichung-cam04 and cam07 each produced
#                      an 11.9 m and 11.6 m "wall", longer than the room; Tao-Hsin-cam04
#                      totalled 46.5 m for a shop whose perimeter is about 20.
#   + contiguous       killed the diagonals and the real walls with them: a true wall's
#                      boundary is broken wherever a fixture stands against it, and
#                      Taichung-cam01 fell to three short panes with the room left open.
#   + store-axis bins  a wall a few degrees off the fitted store axis drifts across bins
#                      and comes back as a row of parallel 3 m pieces: 12 to 19 segments
#                      per camera, none longer than 3.4.
#
# The next attempt should probably fit the *room* rather than each wall -- a rectangle in
# the store frame, sized to the walkable extent -- because a shop has four walls and the
# thing being recovered has four degrees of freedom, not one per boundary cell. That is a
# different program from anything here, which is why this is a comment and not a
# half-finished function.


def build_scene_regular(camera, root: Path | None = None):
    """B-path: every fixture becomes a store-axis-aligned parametric mesh.

    The depth-derived footprints are ragged; the furniture is not. Each component is
    fitted with a robust axis-aligned box in the STORE frame (p3-p97 extents, snapped to
    5 cm) and rendered as the parametric mesh its class names: cabinets with shelf
    slabs, tables with legs, thin walls, columns.
    """
    cf, grids, heights, grid_h = cell_grids(camera, root)
    yaw = store_yaw(grids)
    cy, sy = np.cos(yaw), np.sin(yaw)
    items = [(floor_mesh(grids[1]), "floor", 150, False)]
    # The room's own centre, for deciding which way a fixture faces. Taken from the floor
    # rather than from the furniture: the furniture is what is being oriented.
    _fv = items[0][0][0]
    room_cx = float((_fv[:, 0].min() + _fv[:, 0].max()) / 2)
    room_cz = float((_fv[:, 2].min() + _fv[:, 2].max()) / 2)
    shapes: list[tuple] = []  # (name, w, d, h) as built, for `implausible`
    # ---- 1. FIT. Every fixture as an axis-aligned box in the store frame. Nothing is
    # meshed here: a box's neighbours decide as much about it as its own evidence does,
    # and until 2026-08-28 this loop drew each one the moment it was fitted, which is why
    # nothing in the scene had ever compared two fixtures with each other.
    fixtures: list[list] = []  # [name, u0, u1, v0, v1, h]
    for cid, name in CLASS_NAMES.items():
        if name == "wall":
            continue  # walls are fitted to the whole point set below, not per blob
        h = heights.get(cid, DRAWN_H[name])
        # Open first: a mask bridge a few cells wide welds neighbouring fixtures into one
        # component and the p3-p97 box then spans both. 3 cells is 0.18 m at CELL=0.06 and
        # it is the largest element that helps everything -- measured on this camera, it
        # trims the front counter's fit from 2.89x1.61 m to 2.52x1.39 while leaving the
        # 6.3 m shelf run whole. A 5-cell element fits the counter better still (1.20 m
        # deep) and shatters that shelf run into five pieces, which is worse than the
        # error it fixes: one box too big reads as one fixture, five read as five.
        lab, n = ndimage.label(
            ndimage.binary_opening(grids[cid], np.ones((3, 3))), structure=np.ones((3, 3))
        )
        for k in range(1, n + 1):
            r, c = np.nonzero(lab == k)
            if len(r) < 60:
                continue
            x = c * CELL - 12 + CELL / 2
            z = r * CELL + CELL / 2
            u = x * cy + z * sy
            v = -x * sy + z * cy
            u0, u1 = np.percentile(u, [3, 97])
            v0, v1 = np.percentile(v, [3, 97])
            um, vm = (u0 + u1) / 2, (v0 + v1) / 2
            w = max(round((u1 - u0) / 0.05) * 0.05, 0.3)
            d = max(round((v1 - v0) / 0.05) * 0.05, 0.3)
            if name == "column":
                w, d, h = min(w, 0.8), min(d, 0.8), max(h, COLUMN_MIN_H)
            fixtures.append([name, um - w / 2, um + w / 2, vm - d / 2, vm + d / 2, h])

    # ---- 2. THE WALLS, fitted to the whole `wall` point set rather than to its
    # components, and then only the runs a shopper cannot stand on both sides of.
    _wr, _wc = np.nonzero(grids[WALL_CID])
    _wx = _wc * CELL - 12 + CELL / 2
    _wz = _wr * CELL + CELL / 2
    _fr, _fc = np.nonzero(grids[1])
    _fx = _fc * CELL - 12 + CELL / 2
    _fz = _fr * CELL + CELL / 2
    floor_u, floor_v = _fx * cy + _fz * sy, -_fx * sy + _fz * cy
    walls = []
    for run in wall_runs(_wx * cy + _wz * sy, -_wx * sy + _wz * cy):
        axis, perp, lo, hi, _thick = run
        if floor_both_sides((axis, perp, lo, hi), floor_u, floor_v) > FLOOR_BOTH_SIDES:
            # **Dropped, not re-classified -- and the difference was measured.** A shopper
            # can stand on both sides of this, so it is not a room boundary: that much the
            # relation establishes, and removing it is the whole gain.
            #
            # Converting it into a merchandise run instead was tried on 2026-08-29 and put
            # the fixture in the wrong place, which is the one error the user's standard
            # does not allow. The line is fitted to the **`wall` point set**, and on a
            # white-fixture store that set is the counter *and the wall behind it*, welded
            # (PLAN 7.21). So the fitted line sits between them -- at the wall, a metre
            # behind the counter that made it fail the relation. Projected back through
            # the camera, a box on the wall's line appears floating above the counter in
            # front of it, and `scene_overlay` shows exactly that: Tao-Hsin-cam03 went from
            # four well-placed meshes to six, the two extra ones hovering over the counters.
            #
            # A fixture the mask cannot place is better absent than present and wrong.
            continue
        walls.append(run)
    for axis, perp, lo, hi, _thick in walls:
        a = (lo, perp) if axis == "u" else (perp, lo)
        b = (hi, perp) if axis == "u" else (perp, hi)
        pts = [
            [a[0] * cy - a[1] * sy, a[0] * sy + a[1] * cy],
            [b[0] * cy - b[1] * sy, b[0] * sy + b[1] * cy],
        ]
        shapes.append(("wall", hi - lo, 0.15, DRAWN_H["wall"]))
        items.append((wall(pts, DRAWN_H["wall"], thickness_m=0.15), "wall", 105, False))

    # ---- 3. REGULARISE. The step every scan-to-BIM and structured-modelling pipeline
    # has between fitting and meshing, and the one this file did not: two fixtures cannot
    # occupy the same floor, and a fixture a hand's width from a wall is against it.
    resolved = resolve_overlaps([(f[1], f[2], f[3], f[4]) for f in fixtures])
    kept = []
    for spec, resolved_box in zip(fixtures, resolved, strict=True):
        if resolved_box is None:
            continue
        spec[1], spec[2], spec[3], spec[4] = snap_to_walls(resolved_box, walls)
        kept.append(spec)

    # ---- 4. MESH.
    for name, u0, u1, v0, v1, h in kept:
        w, d = u1 - u0, v1 - v0
        if min(w, d) < 0.15:
            continue  # shrunk to nothing by a neighbour: it was that neighbour
        um, vm = (u0 + u1) / 2, (v0 + v1) / 2
        px, pz = um * cy - vm * sy, um * sy + vm * cy
        at = Placement(px, pz, heading_rad=-yaw)
        if name == "column":
            shapes.append((name, w, d, h))
            items.append((place(column(w, d, h), at), name, 255, True))
        elif name == "display_shelf":
            # **The cap is on the depth, and the depth is the SHORTER side -- which is
            # not always `d`.** `min(d, SHELF_MAX_DEPTH_M)` assumed a merchandise wall
            # runs along `u`, which is true of the camera the cap was measured on and
            # false elsewhere: Taichung-cam10's three shelf components measure
            # u1.33 x v6.29, u0.97 x v2.70 and u0.55 x v1.08 in the store frame, so the
            # cap took a **6.29 m run down to 0.45 m** and left its 1.33 m depth alone.
            # `PLAUSIBLE_M` could not catch it because a truncated run is still a
            # plausible small unit.
            if w >= d:
                run_m, depth_m, head = w, min(d, SHELF_MAX_DEPTH_M), -yaw
            else:
                run_m, depth_m, head = d, min(w, SHELF_MAX_DEPTH_M), -(yaw + np.pi / 2)
            # **And the back panel faces away from the room, not at it.** `shelving` puts
            # its back at local +Z; local +Z lands on (sin head, cos head), and if that
            # points at the room's centre the unit is inside out. Half a turn is the fix.
            if (np.sin(head) * (room_cx - px) + np.cos(head) * (room_cz - pz)) > 0:
                head += np.pi
            shapes.append((name, run_m, depth_m, h))
            mesh = shelving(run_m, depth_m, h)
            items.append((place(mesh, Placement(px, pz, heading_rad=head)), name, 255, True))
        else:  # display_table
            # a footprint too long for four legs is a counter, not a solid prism: a slab
            # on a recessed body, so the surface merchandise sits on exists
            shapes.append((name, w, d, h))
            mesh = table(w, d, h) if max(w, d) < 2.2 else counter(w, d, h)
            items.append((place(mesh, at), name, 255, True))

    # Every fixture that can hold merchandise, as a world AABB plus its top.
    # Each support carries the heights merchandise may actually rest at: a table's top,
    # or a shelving unit's shelf levels. Without them a product sits at whatever height
    # the depth model measured for its region, which is a diagonal staircase drifting
    # through the shelves it is supposed to be standing on -- and "merchandise is on a
    # shelf" is the one sentence this scene exists to say.
    supports = []
    for m, nm, _a, _s in items:
        if nm not in ("display_table", "display_shelf"):
            continue
        x0, x1, z0, z1, top = footprint_of(m)
        levels = shelf_levels(top) if nm == "display_shelf" else [top]
        supports.append((x0, x1, z0, z1, top, levels))
    unsupported: list[tuple] = []
    # extras: doors as solid tall slabs, products as slabs at their measured height
    cy2, sy2 = np.cos(yaw), np.sin(yaw)
    for cid, name, hgt in (
        (6, "door", 2.05),
        (8, "product_boxed_stock", 0.15),
        (9, "product_macbook", 0.12),
        (10, "product_ipad", 0.1),
        (11, "product_iphone", 0.08),
    ):
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
                shapes.append((name, w, d, hgt))
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
                # the region is measured; the items tiled into it are schematic, which is
                # why they are unit-sized real products rather than one slab the size of
                # the region -- a slab asserts a single object that was never detected
                # **Support: merchandise rests on a fixture or it is not drawn.**
                # `base` is the region's measured height, so where the fixture holding
                # it was never reconstructed the products hang in the air at a metre --
                # one of Taichung-cam07's 24 product groups did exactly that, over a
                # shelf whose class that camera never detected. A floating product is
                # not a small cosmetic error: it is the render asserting merchandise at
                # a position with nothing under it, which is the one thing a scene of
                # fixtures exists to deny.
                # **The group's extent, not its centre.** A first version tested only
                # `at.x_m, at.z_m`, so a group whose centre sat on a cabinet was drawn at
                # full size and overhung it -- on Taichung-cam10 a 4x4 array of boxed
                # stock hung half a metre past the cabinet into the aisle, in every frame
                # of the demo. That is the exact state this rule exists to forbid, waved
                # through by the one point that was checked. The corners have to be held
                # too, and a group only partly over a fixture is not resting on it.
                px, pz = at.x_m, at.z_m
                hw, hd = w / 2, d / 2
                corners = [
                    (
                        px + ux * cy2 - vz * sy2,
                        pz + ux * sy2 + vz * cy2,
                    )
                    for ux, vz in ((hw, hd), (hw, -hd), (-hw, hd), (-hw, -hd))
                ]
                held = [
                    (top, levels)
                    for x0, x1, z0, z1, top, levels in supports
                    if all(x0 <= qx <= x1 and z0 <= qz <= z1 for qx, qz in corners)
                    and base <= top + SUPPORT_TOL_M
                ]
                if not held:
                    unsupported.append((name, float(base), px, pz))
                    continue
                # snap to the nearest surface of the fixture holding it. The measured
                # height decides *which* shelf; the fixture decides where that shelf is.
                levels = min(held, key=lambda h: abs(h[0] - base))[1]
                base = min(levels, key=lambda y: abs(y - base))
                for unit in product_units(name, w, d):
                    verts = unit[0].copy()
                    verts[:, 1] += base
                    items.append((place((verts, unit[1]), at), name, 255, False))
    items.append((place(ground_disc(0.35), Placement(0.0, 0.0)), "disc", 130, False))
    for name, base, px, pz in unsupported:
        print(
            f"  {camera}: dropped {name} at {base:.2f} m, "
            f"({px:.2f}, {pz:.2f}) -- nothing under it"
        )
    return cf, items, heights, shapes
