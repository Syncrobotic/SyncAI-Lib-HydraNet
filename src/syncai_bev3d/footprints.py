"""A fixture's footprint from the image and one number, never from per-pixel depth.

Gate D2 (PLAN 10.4b). Until 2026-09-10 every fixture's footprint was the cloud of its
mask pixels lowered to the floor by DA-V2's height at each pixel, rasterised and fitted
at p3-p97 -- and DA-V2's height collapses on exactly the white and wood surfaces a shop
is made of, so the cloud was a smear along the camera's rays: widths inflated, positions
pulled toward the camera, orientations turned toward the ray. Every rule added on top
(the contact band, the object body, the stray cells) was a filter on bad data.

Here the depth model contributes two things per object and nothing per pixel: a height
scalar (p85 of its heights) and which pixels face up (`horiz`). Position and extent come
from geometry the image gives outright:

* a **table** is its top: the up-facing pixels of the object (plus the merchandise on
  it, which hides the top it stands on), each cast onto the plane at the table's height
  -- a ray meets that plane at (H - h) / H of its way to the floor, so the floor hit the
  calibration already gives, scaled, is the point on the top;
* a **shelf** is its foot: the lowest mask pixel of each column, on the floor, fitted as
  a run along the store's axis, with the class's depth behind it;
* a **column** is its foot's width, square.

Each footprint carries its own orientation (the minimum-area rectangle of its points) so
a fixture standing off the store's axis says so, and a reprojection score -- the box
projected back through the camera against the object's own mask -- so a box that does
not sit on what it was built from is flagged rather than drawn as if it did.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import distort_points, pixel_to_ground, undistort_points

# Up-facing threshold on DA-V2's surface normal (|n_y|), the same one `recipe` reads.
TOP_HORIZ_MIN = 0.8
# Fewer top-face pixels than this and the top is not measured; the foot is used instead.
TOP_MIN_PX = 300
# Fewer contact points than this and the object places nothing.
FOOT_MIN_PTS = 20
# Class height intervals the scalar is clipped to (PLAN 7c.27's plausibility, by class).
HEIGHT_CLIP = {
    "display_table": (0.55, 1.15),
    "display_shelf": (0.90, 2.60),
    "column": (1.60, 3.20),
    "wall": (1.80, 3.20),
}
DEFAULT_HEIGHT = {"display_table": 0.85, "display_shelf": 1.9, "column": 2.4, "wall": 2.4}
# Depth behind the foot for a shelf whose top is not seen: the wall-mounted runs in these
# shops (scene_mesh.SHELF_MAX_DEPTH_M).
SHELF_DEPTH_M = 0.45
# A counter whose top is not seen: the depth its foot is given (the fleet's counters
# measure 0.8-1.3 m deep where a top is seen).
TABLE_DEPTH_M = 0.9
COLUMN_SIDE_M = (0.3, 0.8)
# A footprint within this of the store's axis is snapped to it; beyond it keeps its own.
SNAP_DEG = 5.0
# Heights a tall class's top edge is tried at besides its measured one: a wall-mounted
# accessory run tops out near 2.0 m, a shop pillar at the ceiling.
TOP_EDGE_HEIGHTS = {"display_shelf": (1.6, 1.8, 2.0, 2.2), "column": (2.4, 2.7, 3.0)}
# A table's top is taken over its foot from this reprojection IoU up.
TOP_PREFER_MIN = 0.50
# Below this reprojection IoU the footprint is reported and not built.
REPROJECTION_MIN = 0.30


@dataclass
class Footprint:
    name: str
    oid: int
    u0: float
    u1: float
    v0: float
    v1: float
    h: float
    own_deg: float  # the footprint's own orientation relative to the store axis
    n_px: int
    source: str  # "top" or "foot"
    iou: float = 0.0  # reprojection score, filled by `object_footprints`
    members: tuple[int, ...] = ()  # the objects a merged footprint is the union of


def _ground(px_cache, cf: CameraFile, cache_shape) -> np.ndarray:
    """Cache-resolution pixels -> floor metres through the calibration; NaN off the floor."""
    fh, fw = cache_shape
    w, h = cf.image_size_px
    pts = px_cache * np.array([w / fw, h / fh])
    if cf.lens is not None:
        pts = undistort_points(pts, cf.lens.k1, cf.lens.centre_px, cf.lens.radius_px)
    x, z = pixel_to_ground(pts[:, 0], pts[:, 1], cf.camera, cf.plane)
    return np.stack([x, z], axis=1)


def _height(name: str, heights: np.ndarray) -> float:
    hs = heights[np.isfinite(heights) & (heights > 0.05)]
    lo, hi = HEIGHT_CLIP[name]
    if len(hs) < 100:
        return DEFAULT_HEIGHT[name]
    return float(np.clip(np.percentile(hs, 85), lo, hi))


# Steeper than this, in image rows per column, the mask's lower edge is a face's side
# (a vertical edge, nearly vertical in the image) and not its foot on the floor.
FOOT_MAX_SLOPE = 1.0


def _body(mask: np.ndarray) -> np.ndarray:
    """An object's largest connected run of pixels: a mask's strays -- Tao-Hsin-cam15's
    pillar carries a strip of ceiling -- would otherwise cast a foot of their own."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n <= 1:
        return mask
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def _foot_pixels(mask: np.ndarray):
    """The lowest mask pixel per column, on the columns whose lower edge is a foot.

    A tall face is wider at its top in the image than at its foot, so the columns beyond
    the foot hold only upper pixels and their lowest pixel runs up the face's slanted
    side. That side is steep -- rows change fast per column -- where a floor line, even
    an oblique one, is shallow. The steep columns are dropped.
    """
    cols = np.nonzero(mask.any(axis=0))[0]
    rows = np.array([np.nonzero(mask[:, c])[0].max() for c in cols], dtype=int)
    if len(cols) < 5:
        return rows, cols
    slope = np.abs(np.gradient(rows.astype(float), cols.astype(float)))
    keep = slope <= FOOT_MAX_SLOPE
    return rows[keep], cols[keep]


def _near_edge(pts: np.ndarray, *, band_m: float = 0.25) -> np.ndarray:
    """The foot is the near edge of what a mask's bottom casts on the floor.

    A tall face is wider at its top in the image than at its foot -- the top is nearer
    the camera -- so the columns beyond the foot hold only upper pixels, and their lowest
    pixel casts onto the floor BEHIND the fixture, along a line parallel to the foot. On
    the synthetic shelf of `tests/test_footprints.py` that put the foot 0.8 m deep. The
    true foot is the nearest of those lines: points within `band_m` of the near side,
    measured across the points' own dominant direction.
    """
    if len(pts) < 3:
        return pts
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    across = (pts - centre) @ vt[1]
    # the camera stands at the origin; "near" is the side of the line facing it
    if (-centre) @ vt[1] < 0:
        across = -across
    near = np.percentile(across, 90)  # the side facing the camera, robust to a few strays
    return pts[across >= near - band_m]


def _min_area_rect(pts: np.ndarray) -> tuple[float, tuple[float, float]]:
    """(angle deg of the long side, (long, short)) of the minimum-area rectangle."""
    if len(pts) < 3:
        return 0.0, (0.0, 0.0)
    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:  # degenerate (collinear) input
        return 0.0, (0.0, 0.0)
    best_area = float("inf")
    best_angle, best_ext = 0.0, (0.0, 0.0)
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        a = np.arctan2(e[1], e[0])
        c, s = np.cos(-a), np.sin(-a)
        q = np.stack([pts[:, 0] * c - pts[:, 1] * s, pts[:, 0] * s + pts[:, 1] * c], axis=1)
        lo, hi = np.percentile(q, 3, axis=0), np.percentile(q, 97, axis=0)
        ext = hi - lo
        area = float(np.prod(ext))
        if area < best_area:
            long_first = ext[0] >= ext[1]
            best_area = area
            best_angle = float(np.degrees(a) + (0 if long_first else 90))
            best_ext = (float(max(ext)), float(min(ext)))
    return best_angle, best_ext


def _box(pts, h, source, name, oid, n_px, cy, sy) -> Footprint:
    """The store-frame box of a point set, by the class's rules for its source."""
    u = pts[:, 0] * cy + pts[:, 1] * sy
    v = -pts[:, 0] * sy + pts[:, 1] * cy
    own, _ext = _min_area_rect(np.stack([u, v], axis=1))
    own = (own + 45) % 90 - 45
    u0, u1 = np.percentile(u, [3, 97])
    v0, v1 = np.percentile(v, [3, 97])
    if source != "top":
        along_u = (u1 - u0) >= (v1 - v0)
        if name == "column":
            c_, s_ = np.cos(np.radians(own)), np.sin(np.radians(own))
            along = u * c_ + v * s_
            length = float(np.percentile(along, 97) - np.percentile(along, 3))
            side = float(np.clip(length, *COLUMN_SIDE_M))
            um, vm = (u0 + u1) / 2, (v0 + v1) / 2
            u0, u1, v0, v1 = um - side / 2, um + side / 2, vm - side / 2, vm + side / 2
        else:
            depth = TABLE_DEPTH_M if name == "display_table" else SHELF_DEPTH_M
            # the box extends from the foot away from the camera, which stands at the
            # origin of the ground frame: the foot is the face it sees
            if along_u:
                vm = (v0 + v1) / 2
                v0, v1 = (vm, vm + depth) if vm >= 0 else (vm - depth, vm)
            else:
                um = (u0 + u1) / 2
                u0, u1 = (um, um + depth) if um >= 0 else (um - depth, um)
    return Footprint(
        name=name, oid=int(oid), u0=float(u0), u1=float(u1), v0=float(v0), v1=float(v1),
        h=float(h), own_deg=float(own), n_px=n_px, source=source,
    )  # fmt: skip


def object_footprints(ev, yaw: float, *, products=None) -> list[Footprint]:
    """One footprint per `masks_pass` object, in the store frame at `yaw`.

    Each object offers a few candidates -- its top cast at each height in the class's
    interval, its foot on the floor, its top edge cast at its measured height -- and the
    one that reprojects best onto the object's own mask is the footprint. The score is
    kept on it; below REPROJECTION_MIN the caller does not build it.
    """
    from syncai_bev3d.scene_mesh import CLASS_NAMES  # class ids; import here to avoid a cycle

    cf, z = ev.cf, ev.z
    fh, fw = z["gx"].shape
    cy, sy = np.cos(yaw), np.sin(yaw)
    h_cam = cf.plane.height
    out: list[Footprint] = []
    for oid in np.unique(ev.objects[ev.objects > 0]):
        obj = _body(ev.objects == oid)
        cids, counts = np.unique(ev.static[obj], return_counts=True)
        cid = int(cids[np.argmax(counts)])
        if cid not in CLASS_NAMES or CLASS_NAMES[cid] == "wall":
            continue
        name = CLASS_NAMES[cid]
        good = obj & z["geom_ok"]
        h_meas = _height(name, z["height"][good])
        n_px = int(obj.sum())
        candidates: list[Footprint] = []

        if name == "display_table":
            top = good & (z["horiz"] >= TOP_HORIZ_MIN) if "horiz" in z else good
            if products is not None:
                # merchandise hides the top it stands on; inside the object's outline it
                # is the top, at the object's height
                r, c = np.nonzero(obj)
                box = np.zeros_like(obj)
                box[r.min() : r.max() + 1, c.min() : c.max() + 1] = True
                top = top | (products & box & ndimage.binary_dilation(obj, iterations=4))
            if top.sum() >= TOP_MIN_PX:
                r, c = np.nonzero(top)
                g = _ground(np.stack([c + 0.5, r + 0.5], axis=1).astype(float), cf, (fh, fw))
                g = g[np.isfinite(g).all(axis=1)]
                # cast at the measured height. Letting the reprojection choose the height
                # was tried (2026-09-10): the score climbs with a taller box whatever the
                # counter is, and it picked 1.10-1.15 m for 0.9 m counters and 0.55 for
                # others. The score chooses the SOURCE of a footprint, not its height.
                candidates.append(
                    _box(g * (h_cam - h_meas) / h_cam, h_meas, "top", name, oid, n_px, cy, sy)
                )
        # the foot on the floor
        r, c = _foot_pixels(obj)
        g = _ground(np.stack([c + 0.5, r + 0.5], axis=1).astype(float), cf, (fh, fw))
        foot = _near_edge(g[np.isfinite(g).all(axis=1)])
        if len(foot) >= FOOT_MIN_PTS:
            candidates.append(_box(foot, h_meas, "foot", name, oid, n_px, cy, sy))
        # the top edge cast at the measured height: a shelf standing on a cabinet, a
        # pillar whose foot is behind a counter, have no foot on the floor to read
        if name != "display_table":
            r, c = _top_pixels(obj)
            g = _ground(np.stack([c + 0.5, r + 0.5], axis=1).astype(float), cf, (fh, fw))
            g = g[np.isfinite(g).all(axis=1)]
            if len(g) >= FOOT_MIN_PTS:
                for h in (h_meas, *TOP_EDGE_HEIGHTS.get(name, ())):
                    if h < h_cam - 0.3:
                        candidates.append(
                            _box(
                                g * (h_cam - h) / h_cam, h, "top edge", name, oid, n_px, cy, sy
                            )
                        )
        if not candidates:
            continue
        for fp in candidates:
            fp.iou = reprojection_iou(fp, ev, yaw)
        # A table's top says where the table is; its foot says only where its front is.
        # The score cannot choose between them: a shallow box from the foot still covers
        # the counter's whole front face, which is most of its mask from a low camera,
        # and on Taichung-cam10 it beat the top (0.60 to 0.55) with a 0.45 m deep counter.
        # A top that reprojects poorly is not the top (Tao-Hsin-cam15's bar reads its
        # stools and the wall behind as top, 0.31); then the best candidate stands.
        tops = [fp for fp in candidates if fp.source == "top" and fp.iou >= TOP_PREFER_MIN]
        out.append(
            max(tops, key=lambda f: f.iou) if tops else max(candidates, key=lambda f: f.iou)
        )
    return out


def _top_pixels(mask: np.ndarray):
    """The highest mask pixel per column, on the columns whose upper edge is shallow."""
    cols = np.nonzero(mask.any(axis=0))[0]
    rows = np.array([np.nonzero(mask[:, c])[0].min() for c in cols], dtype=int)
    if len(cols) < 5:
        return rows, cols
    slope = np.abs(np.gradient(rows.astype(float), cols.astype(float)))
    keep = slope <= FOOT_MAX_SLOPE
    return rows[keep], cols[keep]


def _silhouette(fp: Footprint, ev, yaw: float) -> np.ndarray | None:
    """The box projected through the camera, as a boolean image at the cache's
    resolution; None when a corner is behind the camera."""
    cf, z = ev.cf, ev.z
    fh, fw = z["gx"].shape
    w, h = cf.image_size_px
    cy, sy = np.cos(yaw), np.sin(yaw)
    corners = []
    for u in (fp.u0, fp.u1):
        for v in (fp.v0, fp.v1):
            x, zz = u * cy - v * sy, u * sy + v * cy
            for y in (0.0, fp.h):
                corners.append((x, y, zz))
    verts = np.asarray(corners, float)
    level = np.stack([verts[:, 0], cf.plane.height - verts[:, 1], verts[:, 2]], axis=-1)
    cam = level @ cf.plane.rotation.T
    depth = cam[:, 2]
    if (depth <= 0.05).any():
        return None
    px = np.stack(
        [
            cf.camera.fx * cam[:, 0] / depth + cf.camera.cx,
            cf.camera.fy * cam[:, 1] / depth + cf.camera.cy,
        ],
        axis=1,
    )
    if cf.lens is not None:
        px = distort_points(px, cf.lens.k1, cf.lens.centre_px, cf.lens.radius_px)
    px = px * np.array([fw / w, fh / h])
    try:
        hull = px[ConvexHull(px).vertices]
    except Exception:
        return None
    sil = Image.new("1", (fw, fh), 0)
    ImageDraw.Draw(sil).polygon([tuple(p) for p in hull], fill=1)
    return np.asarray(sil, bool)


def reprojection_iou(fp: Footprint, ev, yaw: float, *, oids=None) -> float:
    """The box projected through the camera against the object's own mask: IoU of the
    box's silhouette with the mask, at the cache's resolution. `oids` names the objects
    the mask is the union of (default: the footprint's own)."""
    return reprojection_iou_union([fp], ev, yaw, oids=oids if oids is not None else (fp.oid,))


def reprojection_iou_union(fps: list[Footprint], ev, yaw: float, *, oids) -> float:
    """Several boxes together against the union of their objects' masks -- the score a
    split is judged by, since each part alone covers only its share of the mask."""
    sil = None
    for fp in fps:
        one = _silhouette(fp, ev, yaw)
        if one is None:
            return 0.0
        sil = one if sil is None else (sil | one)
    if sil is None:
        return 0.0
    mask = np.isin(ev.objects, list(oids))
    inter = (sil & mask).sum()
    union = (sil | mask).sum()
    return float(inter / union) if union else 0.0


# --- Gate D3: the same object, by geometry ------------------------------------------
# Two footprints are one fixture when they are one surface: the same class, tops at one
# height, axes aligned, and touching end to end. A footprint is two fixtures when a box
# cannot hold it: an L-shaped counter row fills little of its own rectangle, and its
# points split into two rectangles that each fill theirs.
MERGE_HEIGHT_M = 0.15
MERGE_AXIS_DEG = 10.0
MERGE_GAP_M = 0.30
MERGE_OVERLAP_MIN = 0.5  # share of the shorter side the two boxes overlap across
SPLIT_FILL_MAX = 0.65  # store-frame fill below which a footprint may be more than one box
SPLIT_FILL_OWN_MAX = 0.70  # and its own-frame fill too (an L's is ~0.67, a rectangle's ~1)


def _fill(pts_uv: np.ndarray) -> float:
    """Share of the points' own minimum rectangle that they cover, on a 5 cm grid."""
    if len(pts_uv) < 10:
        return 1.0
    ang, (long, short) = _min_area_rect(pts_uv)
    if long <= 0 or short <= 0:
        return 1.0
    a = np.radians(ang)
    c, s = np.cos(-a), np.sin(-a)
    q = np.stack([pts_uv[:, 0] * c - pts_uv[:, 1] * s, pts_uv[:, 0] * s + pts_uv[:, 1] * c], 1)
    lo = np.percentile(q, 3, axis=0)
    cells = set(map(tuple, np.floor((q - lo) / 0.05).astype(int)))
    return min(1.0, len(cells) / max((long / 0.05) * (short / 0.05), 1.0))


def _fill_axis(pts_uv: np.ndarray) -> float:
    """Share of the points' axis-aligned (store-frame) rectangle they cover, 5 cm grid."""
    if len(pts_uv) < 10:
        return 1.0
    lo, hi = np.percentile(pts_uv, 3, axis=0), np.percentile(pts_uv, 97, axis=0)
    ext = hi - lo
    if (ext <= 0).any():
        return 1.0
    cells = set(map(tuple, np.floor((pts_uv - lo) / 0.05).astype(int)))
    return min(1.0, len(cells) / max((ext[0] / 0.05) * (ext[1] / 0.05), 1.0))


def _touching(a: Footprint, b: Footprint) -> bool:
    """End to end along the shared long axis, overlapping across it."""
    au, av = a.u1 - a.u0, a.v1 - a.v0
    bu, bv = b.u1 - b.u0, b.v1 - b.v0
    along_u = (au >= av) and (bu >= bv)
    along_v = (av > au) and (bv > bu)
    if not (along_u or along_v):
        return False
    if along_u:
        gap = max(a.u0, b.u0) - min(a.u1, b.u1)
        over = min(a.v1, b.v1) - max(a.v0, b.v0)
        short = min(av, bv)
    else:
        gap = max(a.v0, b.v0) - min(a.v1, b.v1)
        over = min(a.u1, b.u1) - max(a.u0, b.u0)
        short = min(au, bu)
    return gap <= MERGE_GAP_M and over >= MERGE_OVERLAP_MIN * short


def _same_fixture(a: Footprint, b: Footprint) -> bool:
    return (
        a.name == b.name
        and abs(a.h - b.h) <= MERGE_HEIGHT_M
        and abs(((a.own_deg - b.own_deg) + 45) % 90 - 45) <= MERGE_AXIS_DEG
        and _touching(a, b)
    )


def merge_footprints(fps: list[Footprint], ev, yaw: float) -> list[Footprint]:
    """Footprints that are one surface become one box, kept only if the box reprojects
    onto the union of their masks no worse than the parts did onto theirs."""
    fps = list(fps)
    merged = True
    while merged:
        merged = False
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                a, b = fps[i], fps[j]
                if not _same_fixture(a, b):
                    continue
                union = Footprint(
                    name=a.name,
                    oid=a.oid,
                    u0=min(a.u0, b.u0),
                    u1=max(a.u1, b.u1),
                    v0=min(a.v0, b.v0),
                    v1=max(a.v1, b.v1),
                    h=(a.h * a.n_px + b.h * b.n_px) / (a.n_px + b.n_px),
                    own_deg=a.own_deg if a.n_px >= b.n_px else b.own_deg,
                    n_px=a.n_px + b.n_px,
                    source=f"{a.source}+{b.source}",
                )
                union.iou = reprojection_iou(
                    union, ev, yaw, oids=(*(a.members or (a.oid,)), *(b.members or (b.oid,)))
                )
                if union.iou >= min(a.iou, b.iou) - 0.05:
                    union.members = (*(a.members or (a.oid,)), *(b.members or (b.oid,)))
                    fps = [f for k, f in enumerate(fps) if k not in (i, j)] + [union]
                    merged = True
                    break
            if merged:
                break
    return fps


def _points_uv(fp: Footprint, ev, yaw: float):
    """The floor-frame points a footprint was built from (its top or its foot), again."""
    cf, z = ev.cf, ev.z
    fh, fw = z["gx"].shape
    cy, sy = np.cos(yaw), np.sin(yaw)
    oids = fp.members or (fp.oid,)
    obj = np.isin(ev.objects, list(oids))
    obj = _body(obj) if len(oids) == 1 else obj
    good = obj & z["geom_ok"]
    if fp.source.startswith("top") and "horiz" in z:
        sel = good & (z["horiz"] >= TOP_HORIZ_MIN)
        r, c = np.nonzero(sel)
        g = _ground(np.stack([c + 0.5, r + 0.5], axis=1).astype(float), cf, (fh, fw))
        g = g[np.isfinite(g).all(axis=1)] * (cf.plane.height - fp.h) / cf.plane.height
    else:
        r, c = _foot_pixels(obj)
        g = _ground(np.stack([c + 0.5, r + 0.5], axis=1).astype(float), cf, (fh, fw))
        g = _near_edge(g[np.isfinite(g).all(axis=1)])
    return np.stack([g[:, 0] * cy + g[:, 1] * sy, -g[:, 0] * sy + g[:, 1] * cy], axis=1)


SPLIT_GRID_M = 0.05  # rect_decompose coarsens 5x: 0.25 m blocks
SPLIT_MIN_CELLS = 12  # a part smaller than this (0.12 m2) is a mask edge, not a fixture
SPLIT_MAX_PARTS = 3
# A part narrower than this is the decomposition's residue along a ragged edge, not a
# counter (Taichung-cam04's L came with a 0.25 x 1.75 m sliver beside its two legs).
SPLIT_PART_SHORT_M = 0.4
# The rectangles must cover this share of the occupied cells, or they are not the shape.
SPLIT_COVER_MIN = 0.85


def split_footprints(fps: list[Footprint], ev, yaw: float) -> list[Footprint]:
    """A footprint that is not one box in the store's frame is more than one: its
    points, rasterised along the store's axes, are decomposed into rectangles
    (`scene_mesh.rect_decompose`, the store's boxy prior), each becomes a box, and the
    set is kept when two or three rectangles cover the points and together reproject
    onto the object no worse than the one box did. An L-shaped counter row is two legs;
    a rectangle is one rectangle; a fixture standing off the store's axis decomposes
    into a staircase of many small ones and is left alone.

    The store frame, not the points' own: the minimum-area rectangle of an L is
    diagonal (-36 deg on the synthetic one), which is why `own_deg` says nothing about
    an L and why the cut is made along the axes the counters were laid on.
    """
    from syncai_bev3d.scene_mesh import rect_decompose

    out: list[Footprint] = []
    for fp in fps:
        if (
            fp.name != "display_table"
            or not fp.source.startswith("top")
            or fp.iou < REPROJECTION_MIN
        ):
            out.append(fp)  # a footprint not on its object is not worth cutting
            continue
        pts = _points_uv(fp, ev, yaw)
        # not a box in any frame: a rotated rectangle fills its own frame and is left
        # alone; an L fills neither (its own minimum rectangle is diagonal)
        if (
            len(pts) < 200
            or _fill(pts) >= SPLIT_FILL_OWN_MAX
            or _fill_axis(pts) >= SPLIT_FILL_MAX
        ):
            out.append(fp)
            continue
        lo = np.percentile(pts, 1, axis=0)
        cells = np.floor((pts - lo) / SPLIT_GRID_M).astype(int)
        cells = cells[(cells >= 0).all(axis=1)]
        grid = np.zeros((cells[:, 1].max() + 2, cells[:, 0].max() + 2), bool)
        grid[cells[:, 1], cells[:, 0]] = True
        grid = ndimage.binary_closing(grid, np.ones((3, 3)))
        rects = [r for group in rect_decompose(grid, SPLIT_MIN_CELLS) for r in group]
        rects = [
            r
            for r in rects
            if (r[1] - r[0] + 1) * (r[3] - r[2] + 1) >= SPLIT_MIN_CELLS
            and min(r[1] - r[0] + 1, r[3] - r[2] + 1) * SPLIT_GRID_M >= SPLIT_PART_SHORT_M
        ]
        if not 2 <= len(rects) <= SPLIT_MAX_PARTS:
            out.append(fp)
            continue
        covered = np.zeros_like(grid)
        for r0, r1, c0, c1 in rects:
            covered[r0 : r1 + 1, c0 : c1 + 1] = True
        if (covered & grid).sum() < SPLIT_COVER_MIN * grid.sum():
            out.append(fp)
            continue
        boxes = []
        for r0, r1, c0, c1 in rects:
            part = Footprint(
                name=fp.name,
                oid=fp.oid,
                u0=float(lo[0] + c0 * SPLIT_GRID_M),
                u1=float(lo[0] + (c1 + 1) * SPLIT_GRID_M),
                v0=float(lo[1] + r0 * SPLIT_GRID_M),
                v1=float(lo[1] + (r1 + 1) * SPLIT_GRID_M),
                h=fp.h,
                own_deg=0.0,
                n_px=fp.n_px // len(rects),
                source=fp.source + " (split)",
                members=fp.members,
            )
            boxes.append(part)
        oids = fp.members or (fp.oid,)
        together = reprojection_iou_union(boxes, ev, yaw, oids=oids)
        if together >= fp.iou - 0.02:
            for b in boxes:
                b.iou = together
            out.extend(boxes)
        else:
            out.append(fp)
    return out


def regularise_footprints(fps: list[Footprint], ev, yaw: float) -> list[Footprint]:
    """Gate D3 in one call: merge what is one surface, split what a box cannot hold."""
    return split_footprints(merge_footprints(fps, ev, yaw), ev, yaw)


# --- Gate D4: the walls, from where they meet the floor ---------------------------------
# A wall object's foot -- the lowest mask pixel per column, on the floor -- is a line in
# the store frame along one of its axes. Runs on one axis within WALL_PERP_M of each
# other and within WALL_GAP_M end to end are one wall. The cell smear these replace put
# walls a metre into the aisle wherever DA-V2 lowered a white face too far.
WALL_PERP_M = 0.35
WALL_GAP_M = 1.0
WALL_MIN_RUN_M = 0.8
WALL_MIN_PTS = 40
# The scene's own extent (scene_mesh's 24 x 14 m grid); a foot beyond it is the horizon.
WALL_RANGE_M = 12.0
# A wall slab reprojecting below this is not on its mask: a foot read off a face's side.
WALL_IOU_MIN = 0.20


@dataclass
class WallRun:
    axis: str  # "u": runs along u at v = perp; "v": runs along v at u = perp
    perp: float
    lo: float
    hi: float
    members: tuple[int, ...]
    iou: float = 0.0


def wall_runs_from_feet(ev, yaw: float) -> list[WallRun]:
    """One run per wall object's foot, merged where collinear and near."""
    from syncai_bev3d.scene_mesh import WALL_CID

    cf, z = ev.cf, ev.z
    fh, fw = z["gx"].shape
    cy, sy = np.cos(yaw), np.sin(yaw)
    runs: list[WallRun] = []
    for oid in np.unique(ev.objects[ev.objects > 0]):
        obj = _body(ev.objects == oid)
        cids, counts = np.unique(ev.static[obj], return_counts=True)
        if int(cids[np.argmax(counts)]) != WALL_CID:
            continue
        r, c = _foot_pixels(obj)
        g = _ground(np.stack([c + 0.5, r + 0.5], axis=1).astype(float), cf, (fh, fw))
        g = g[np.isfinite(g).all(axis=1)]
        # a wall mask reaches the horizon, and a pixel near it casts kilometres away
        g = g[(np.abs(g[:, 0]) < WALL_RANGE_M) & (g[:, 1] > 0) & (g[:, 1] < WALL_RANGE_M)]
        g = _near_edge(g, band_m=WALL_PERP_M)
        if len(g) < WALL_MIN_PTS:
            continue
        u = g[:, 0] * cy + g[:, 1] * sy
        v = -g[:, 0] * sy + g[:, 1] * cy
        u0, u1 = np.percentile(u, [3, 97])
        v0, v1 = np.percentile(v, [3, 97])
        if (u1 - u0) >= (v1 - v0):
            run = WallRun("u", float(np.median(v)), float(u0), float(u1), (int(oid),))
        else:
            run = WallRun("v", float(np.median(u)), float(v0), float(v1), (int(oid),))
        if run.hi - run.lo >= WALL_MIN_RUN_M:
            runs.append(run)
    merged = True
    while merged:
        merged = False
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a, b = runs[i], runs[j]
                if a.axis != b.axis or abs(a.perp - b.perp) > WALL_PERP_M:
                    continue
                if max(a.lo, b.lo) - min(a.hi, b.hi) > WALL_GAP_M:
                    continue
                n_a, n_b = a.hi - a.lo, b.hi - b.lo
                union = WallRun(
                    a.axis,
                    (a.perp * n_a + b.perp * n_b) / (n_a + n_b),
                    min(a.lo, b.lo),
                    max(a.hi, b.hi),
                    (*a.members, *b.members),
                )
                runs = [r for k, r in enumerate(runs) if k not in (i, j)] + [union]
                merged = True
                break
            if merged:
                break
    for run in runs:
        run.iou = _wall_iou(run, ev, yaw)
    return [run for run in runs if run.iou >= WALL_IOU_MIN]


def _wall_iou(run: WallRun, ev, yaw: float, *, height_m: float = 2.4) -> float:
    """The wall slab reprojected against the union of its objects' masks."""
    if run.axis == "u":
        fp = Footprint(
            "wall",
            run.members[0],
            run.lo,
            run.hi,
            run.perp - 0.075,
            run.perp + 0.075,
            height_m,
            0.0,
            0,
            "foot",
        )
    else:
        fp = Footprint(
            "wall",
            run.members[0],
            run.perp - 0.075,
            run.perp + 0.075,
            run.lo,
            run.hi,
            height_m,
            0.0,
            0,
            "foot",
        )
    return reprojection_iou(fp, ev, yaw, oids=run.members)
