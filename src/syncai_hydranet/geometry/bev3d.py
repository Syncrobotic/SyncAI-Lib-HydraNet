"""A perspective view of the floor map, with what is standing on it drawn standing up.

The flat top-down panel in ``bev.py`` is the honest primitive -- metres, class ids, no
colour -- and it stays that way. This renders it for a person: the floor in perspective,
the boundary raised into a wall, and detections as boxes on the ground rather than dots.

**None of this adds information.** Every surface here is drawn from something the flat map
already asserted, and the two disagreements worth naming up front are:

* **Heights are not measured for terrain.** A wall is drawn 2.4 m tall because walls are
  about that; the camera never said so. Only the *footprint* of the boundary is derived
  from the mask. Object heights, by contrast, are recovered from the top of the detection
  box against the same ground plane, so those are as real as the plane assumption.
* **Perspective invites depth reading.** The virtual camera is a drawing device with no
  relation to the real one, so apparent size in this panel means nothing; the distance
  rules on the floor are the only scale.

What survives, and what this view is for: where the free space ends, in which direction,
and what class the thing at the boundary is. That last part is the reason it exists --
the flat map paints a boundary cell `blocked` and stops, while the terrain head knows
whether the robot is looking at a wall, a glass storefront or a display fixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .bev import IGNORE, BevGrid

# Nominal heights, in metres, for the boundary ribbon. Drawing, not measurement -- see
# the module docstring. Keyed by the retail/indoor terrain id, which share 0-11.
CLASS_HEIGHT_M = {
    1: 0.0,  # floor_hard
    2: 0.0,  # floor_soft
    3: 0.0,  # floor_metal
    4: 0.0,  # wet_slippery
    5: 0.6,  # stairs
    6: 0.15,  # threshold_ramp
    7: 2.4,  # wall
    8: 2.2,  # glass
    9: 2.1,  # door
    10: 0.9,  # obstacle_furniture
    11: 1.7,  # person
    12: 1.4,  # display_fixture
}
DEFAULT_HEIGHT_M = 1.0


@dataclass(frozen=True)
class VirtualCam:
    """The camera this panel is *drawn* from. Unrelated to the one that took the frame.

    Sits behind and above the origin looking forward and down, which is the view that
    makes a floor plan read as a floor. ``focal`` is in pixels of the output panel.
    """

    height_m: float = 8.5
    setback_m: float = 2.6
    pitch_deg: float = 60.0
    focal_px: float = 620.0
    cx: float = 0.0
    cy: float = 0.0

    def project(self, x, y, z):
        """World (x right, y up from floor, z forward) -> panel pixels, plus depth."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        t = np.radians(self.pitch_deg)
        # forward is tilted down by `t`; up stays orthogonal to it
        fwd = np.array([0.0, -np.sin(t), np.cos(t)])
        up = np.array([0.0, np.cos(t), np.sin(t)])
        right = np.array([1.0, 0.0, 0.0])
        px_ = x - 0.0
        py_ = y - self.height_m
        pz_ = z + self.setback_m
        xc = px_ * right[0] + py_ * right[1] + pz_ * right[2]
        yc = px_ * up[0] + py_ * up[1] + pz_ * up[2]
        zc = px_ * fwd[0] + py_ * fwd[1] + pz_ * fwd[2]
        zc = np.maximum(zc, 1e-3)
        return self.cx + self.focal_px * xc / zc, self.cy - self.focal_px * yc / zc, zc


def _perspective_coeffs(dst, src):
    """PIL wants output->input. `dst` are panel points, `src` the matching raster points."""
    a = []
    b = []
    for (dx, dy), (sx, sy) in zip(dst, src, strict=True):
        a.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        b.append(sx)
        a.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        b.append(sy)
    return np.linalg.solve(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def _shade(rgb, factor: float):
    return tuple(int(np.clip(c * factor, 0, 255)) for c in rgb[:3])


def boundary_rays(trav_bev: np.ndarray, grid: BevGrid, n_rays: int = 160):
    """Per ray: how far the floor reached, and the grid cell that ended it.

    Same quantity the flat map's free-space filter uses -- the last walkable range along
    each bearing -- returned rather than rasterised, because a ribbon needs the polyline.
    """
    xx, zz = grid.centres()
    xx = xx[::-1]
    zz = zz[::-1]
    ang = np.arctan2(xx, zz)
    rng = np.hypot(xx, zz)
    lo, hi = float(ang.min()), float(ang.max())
    ray = np.clip(
        ((ang - lo) / max(hi - lo, 1e-9) * (n_rays - 1)).astype(np.int32), 0, n_rays - 1
    )
    walkable = (trav_bev == 2) | (trav_bev == 1)
    reach = np.zeros(n_rays)
    np.maximum.at(reach, ray[walkable], rng[walkable])
    angles = lo + (np.arange(n_rays) + 0.5) / n_rays * (hi - lo)
    return angles, reach


def render(
    trav_bev: np.ndarray,
    terrain_bev: np.ndarray | None,
    grid: BevGrid,
    objects,
    size: tuple[int, int],
    *,
    trav_colors,
    terrain_colors=None,
    cam: VirtualCam | None = None,
    class_names=None,
    bg=(11, 14, 19),
) -> Image.Image:
    """Draw the panel. `trav_bev`/`terrain_bev` are the flat maps, far field at row 0."""
    w, h = size
    cam = cam or VirtualCam()
    # Fit the mapped window to the panel instead of trusting a hand-tuned focal length:
    # the window is set by --range at the call site, so a fixed focal leaves the floor
    # as a postage stamp in a black field the moment anyone changes it.
    probe = VirtualCam(cam.height_m, cam.setback_m, cam.pitch_deg, 1.0, 0.0, 0.0)
    cx_m = [grid.x_min, grid.x_max]
    cz_m = [grid.z_min, grid.z_max]
    top_m = max(CLASS_HEIGHT_M.values())
    us, vs = [], []
    for x in cx_m:
        for z in cz_m:
            for y in (0.0, top_m):
                pu, pv, _ = probe.project(x, y, z)
                us.append(float(pu))
                vs.append(float(pv))
    span_u = max(max(us) - min(us), 1e-6)
    span_v = max(max(vs) - min(vs), 1e-6)
    # A room seen at this pitch projects roughly square, so fit it into a square region
    # at the top and leave the remainder for the legend. Fitting to the full panel
    # height on a tall column stretches nothing and wastes everything: the scene stays
    # the same shape and simply floats in black.
    fit_h = min(h, w)
    focal = min(w / span_u, fit_h / span_v) * 0.92
    # `project` adds focal*u to cx and adds focal*v to cy (its own minus is already in
    # the probe's output), so both offsets are subtracted. Getting this sign wrong
    # centres nothing and only shows once the fit is tight enough to clip.
    mid_u = (max(us) + min(us)) / 2 * focal
    mid_v = (max(vs) + min(vs)) / 2 * focal
    cam = VirtualCam(
        cam.height_m, cam.setback_m, cam.pitch_deg, focal, w / 2 - mid_u, fit_h / 2 - mid_v
    )
    panel = Image.new("RGB", (w, h), bg)

    # --- the floor, warped ---------------------------------------------------
    # Colour by terrain where the flat map has it and the cell is walkable; fall back to
    # traversability. The floor a robot drives on is the part terrain calls floor_*, so
    # anything else there is either the boundary or a mistake, and both should look
    # different from clean floor rather than be smoothed into it.
    src = np.full((*trav_bev.shape, 3), bg, dtype=np.uint8)
    walk = (trav_bev == 2) | (trav_bev == 1)
    if terrain_bev is not None and terrain_colors is not None:
        ok = walk & (terrain_bev != IGNORE)
        src[ok] = np.asarray(terrain_colors)[terrain_bev[ok]]
    fallback = walk if terrain_bev is None else (walk & ~(terrain_bev != IGNORE))
    src[fallback] = np.asarray(trav_colors)[trav_bev[fallback]]

    rows, cols = trav_bev.shape
    corners_m = [
        (grid.x_min, grid.z_min),
        (grid.x_max, grid.z_min),
        (grid.x_max, grid.z_max),
        (grid.x_min, grid.z_max),
    ]
    corners_px = [
        (0.0, float(rows)),
        (float(cols), float(rows)),
        (float(cols), 0.0),
        (0.0, 0.0),
    ]
    scr = [cam.project(x, 0.0, z)[:2] for x, z in corners_m]
    scr = [(float(a), float(b)) for a, b in scr]
    coeffs = _perspective_coeffs(scr, corners_px)
    ground = Image.fromarray(src).transform(
        (w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=bg
    )
    # Only paste where the warp actually landed on floor, so the fill does not wipe the bg.
    mask = Image.fromarray(
        ((np.asarray(ground) != np.array(bg)).any(-1) * 255).astype(np.uint8)
    )
    panel.paste(ground, (0, 0), mask)

    d = ImageDraw.Draw(panel, "RGBA")

    # --- distance rules, drawn on the plane so they carry the perspective ----
    for z in range(int(np.ceil(grid.z_min)), int(grid.z_max) + 1):
        xs = np.linspace(grid.x_min, grid.x_max, 40)
        pts = [tuple(map(float, cam.project(x, 0.0, float(z))[:2])) for x in xs]
        d.line(pts, fill=(96, 112, 132, 150), width=1)
        lx, ly = pts[-1]
        lx = min(max(lx + 6, 2), w - 34)  # a rule that runs off the panel loses its label
        d.text((lx, min(max(ly - 7, 2), h - 14)), f"{z} m", fill=(120, 140, 165))
    for x in np.arange(np.ceil(grid.x_min), grid.x_max + 0.01, 2.0):
        zs = np.linspace(grid.z_min, grid.z_max, 40)
        pts = [tuple(map(float, cam.project(float(x), 0.0, z)[:2])) for z in zs]
        d.line(pts, fill=(96, 112, 132, 90), width=1)

    # --- the boundary, raised ------------------------------------------------
    angles, reach = boundary_rays(trav_bev, grid)
    quads = []
    for i in range(len(angles) - 1):
        r0, r1 = reach[i], reach[i + 1]
        if r0 <= 0 or r1 <= 0:
            continue
        a0, a1 = angles[i], angles[i + 1]
        p0 = (r0 * np.sin(a0), r0 * np.cos(a0))
        p1 = (r1 * np.sin(a1), r1 * np.cos(a1))
        cls = None
        if terrain_bev is not None:
            mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            cell = grid.to_cell(*mid)
            if cell is not None:
                r, c = cell
                # look a little past the boundary for the thing that ended the floor
                rr = max(r - 3, 0)
                v = int(terrain_bev[rr, c])
                cls = v if v != IGNORE else None
        ht = CLASS_HEIGHT_M.get(cls, DEFAULT_HEIGHT_M) if cls is not None else DEFAULT_HEIGHT_M
        if ht <= 0.01:
            ht = 0.25  # something ended the floor even if the class says it is flat
        colour = (
            tuple(np.asarray(terrain_colors)[cls])
            if (cls is not None and terrain_colors is not None)
            else (150, 165, 190)
        )
        b0 = cam.project(p0[0], 0.0, p0[1])
        b1 = cam.project(p1[0], 0.0, p1[1])
        t0 = cam.project(p0[0], ht, p0[1])
        t1 = cam.project(p1[0], ht, p1[1])
        quads.append(
            (
                float(b0[2]),
                [
                    (float(b0[0]), float(b0[1])),
                    (float(b1[0]), float(b1[1])),
                    (float(t1[0]), float(t1[1])),
                    (float(t0[0]), float(t0[1])),
                ],
                colour,
            )
        )
    for _, poly, colour in sorted(quads, key=lambda q: -q[0]):  # far first
        d.polygon(poly, fill=(*_shade(colour, 0.72), 205), outline=(*_shade(colour, 1.15), 235))

    # --- objects, as boxes standing on the floor -----------------------------
    for obj in sorted(objects, key=lambda o: -o.get("z_m", 0.0)):
        x, z = obj.get("x_m"), obj.get("z_m")
        if x is None or z is None:
            continue
        if not (grid.x_min <= x <= grid.x_max and grid.z_min <= z <= grid.z_max):
            continue
        half = max(float(obj.get("width_m") or 0.5), 0.15) / 2
        ht = float(obj.get("height_m") or 1.7)
        base = [
            (x - half, z - half),
            (x + half, z - half),
            (x + half, z + half),
            (x - half, z + half),
        ]
        bp = [tuple(map(float, cam.project(px, 0.0, pz)[:2])) for px, pz in base]
        tp = [tuple(map(float, cam.project(px, ht, pz)[:2])) for px, pz in base]
        col = (90, 200, 255)
        for i in range(4):  # vertical faces
            j = (i + 1) % 4
            d.polygon([bp[i], bp[j], tp[j], tp[i]], fill=(*col, 55), outline=(*col, 170))
        d.polygon(tp, fill=(*col, 80), outline=(*col, 230))
        d.polygon(bp, outline=(*col, 255))
        lx, ly = tp[0]
        d.text(
            (lx + 4, ly - 12),
            f"{obj.get('name', '?')} {obj.get('range_m', 0):.1f}m",
            fill=(225, 240, 255),
        )

    # --- legend, only for what this frame actually contains ------------------
    if terrain_bev is not None and terrain_colors is not None and class_names:
        present = [
            int(c) for c in np.unique(terrain_bev) if c != IGNORE and int(c) < len(class_names)
        ]
        y = h - 12 - 15 * len(present)
        for cid in present:
            col = tuple(int(v) for v in np.asarray(terrain_colors)[cid])
            d.rectangle([10, y, 22, y + 10], fill=col, outline=(0, 0, 0))
            d.text((28, y - 2), class_names[cid], fill=(200, 214, 234))
            y += 15

    # --- ego ------------------------------------------------------------------
    ex, ey = cam.project(0.0, 0.0, grid.z_min)[:2]
    d.polygon(
        [
            (float(ex), float(ey)),
            (float(ex) - 13, float(ey) + 20),
            (float(ex) + 13, float(ey) + 20),
        ],
        fill=(235, 245, 255),
    )
    return panel
