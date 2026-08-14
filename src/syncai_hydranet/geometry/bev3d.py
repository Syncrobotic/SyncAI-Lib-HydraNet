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

How it is drawn, and why:

* **Supersampled.** Every edge here is a polygon and PIL does not anti-alias one, so the
  geometry is drawn at 2x and resolved down. Text is drawn after the downsample, where it
  is already sharp and does not pay for it.
* **The floor warp samples NEAREST, not BILINEAR.** The raster is a label map wearing
  colours; interpolating across a class edge produces a colour that belongs to no class,
  and a viewer cannot tell that invented colour from a real one. Supersampling is what
  keeps the edge smooth, and it does it without inventing anything.
* **The floor is quiet and the hazards are loud.** The drivable plate is desaturated
  toward the panel's own blue-grey with the terrain colour mixed in at `FLOOR_TINT`, so
  `floor_metal` and `floor_hard` stay distinguishable while neither competes with
  `caution`, which keeps its colour at full strength.
* **The boundary is one polygon per class run**, not one per ray. Per-ray quads with their
  own outlines drew a seam down every wall.
* **The far edge dissolves** rather than ending on a line. The mapped window has an edge;
  the floor it describes does not, and a crisp boundary there reads as "the room stops
  here".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .bev import IGNORE, BevGrid, ray_reach

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

# The drivable plate. Deliberately quiet and cool: the floor is the largest area on the
# panel, and a saturated floor leaves nothing for the hazards to contrast against.
DRIVABLE_RGB = (38, 52, 68)
# How much of the terrain colour survives on the floor. Enough to tell floor_metal from
# floor_hard, not enough to make the floor the loudest thing in frame.
FLOOR_TINT = 0.34
# Brightness of the far edge of the mapped window, against 1.0 at the ego. Low enough
# that the floor dissolves into the background instead of ending on a hard line -- the
# mapped window has an edge, but the floor it describes does not, and a crisp boundary
# there reads as "the room stops here".
FAR_FADE = 0.16

OBJECT_RGB = {
    "person": (255, 138, 196),
    "chair": (150, 196, 236),
    "_": (150, 196, 236),
}


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
    """Per ray: how far the floor reached, as a polyline rather than a raster.

    The same reduction the flat map's free-space filter runs on, and deliberately the
    same code: a ribbon drawn from a second implementation would drift away from the map
    it is supposed to be a picture of, and drift is the failure nobody looks for.
    """
    angles, reach, _, _ = ray_reach(trav_bev, grid, n_rays)
    return angles, reach


def _fit_cam(cam: VirtualCam, grid: BevGrid, size: tuple[int, int]) -> VirtualCam:
    """Size the virtual camera so the mapped window fills the panel.

    The window is set by ``--range`` at the call site, so a fixed focal length leaves the
    floor as a postage stamp the moment anyone changes it.
    """
    w, h = size
    probe = VirtualCam(cam.height_m, cam.setback_m, cam.pitch_deg, 1.0, 0.0, 0.0)
    top_m = max(CLASS_HEIGHT_M.values())
    us, vs = [], []
    for x in (grid.x_min, grid.x_max):
        for z in (grid.z_min, grid.z_max):
            for y in (0.0, top_m):
                pu, pv, _ = probe.project(x, y, z)
                us.append(float(pu))
                vs.append(float(pv))
    span_u = max(max(us) - min(us), 1e-6)
    span_v = max(max(vs) - min(vs), 1e-6)
    focal = min(w / span_u, h / span_v) * 0.94
    mid_u = (max(us) + min(us)) / 2 * focal
    mid_v = (max(vs) + min(vs)) / 2 * focal
    return VirtualCam(
        cam.height_m, cam.setback_m, cam.pitch_deg, focal, w / 2 - mid_u, h / 2 - mid_v
    )


def _font(px: int):
    """A real face if the system has one; PIL's bitmap default is a poor last resort but
    it keeps this importable on a bare Jetson."""
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _runs(values):
    """Contiguous spans of equal value: [(start, stop_exclusive, value), ...].

    The boundary used to be drawn as one quad per ray with its own outline, which showed
    up as vertical seams down every wall. A wall is one surface; it should be one polygon.
    """
    out, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((start, i, values[start]))
            start = i
    return out


def _floor_raster(trav_bev, terrain_bev, trav_colors, terrain_colors, bg):
    """The flat map as pixels, in the Tesla-ish value structure: the drivable surface is a
    quiet cool plate, and saturation is spent on the things that change behaviour."""
    src = np.zeros((*trav_bev.shape, 3), dtype=np.float32)
    src[:] = bg
    walk = (trav_bev == 2) | (trav_bev == 1)

    # Drivable floor: desaturated toward the panel's blue-grey rather than painted with
    # the terrain colour at full strength. Terrain still tints it, so `floor_metal` and
    # `floor_hard` remain distinguishable, but the floor stops competing with the hazards.
    base = np.array(DRIVABLE_RGB, dtype=np.float32)
    if terrain_bev is not None and terrain_colors is not None:
        ok = walk & (terrain_bev != IGNORE)
        tint = np.asarray(terrain_colors, dtype=np.float32)[terrain_bev[ok]]
        src[ok] = base * (1 - FLOOR_TINT) + tint * FLOOR_TINT
        rest = walk & ~ok
    else:
        rest = walk
    src[rest] = base

    # `caution` keeps its own colour at full strength: it is the one floor state that is
    # about the robot's behaviour rather than about what the floor is made of.
    care = walk & (trav_bev == 1)
    src[care] = np.asarray(trav_colors, dtype=np.float32)[1]

    # Range fade. Row 0 is the far field, so the ramp runs down the array.
    rows, cols = src.shape[:2]
    fade = np.linspace(FAR_FADE, 1.0, rows, dtype=np.float32)[:, None, None]
    # A little extra light straight ahead of the ego, falling off to the sides: the near
    # centre is where the depth return is densest, and the panel should look like it
    # knows most about the place it knows most about.
    lateral = 1.0 - 0.22 * np.abs(np.linspace(-1, 1, cols, dtype=np.float32))[None, :, None]
    fade = fade * lateral
    lit = src * fade + np.array(bg, dtype=np.float32) * (1 - fade)
    lit[~walk] = bg
    return np.clip(lit, 0, 255).astype(np.uint8)


def _vignette(size, strength=0.34):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - w / 2) / (w / 2)
    dy = (yy - h / 2) / (h / 2)
    r = np.clip(np.sqrt(dx * dx + dy * dy) / 1.35, 0, 1)
    a = (r**2.2 * 255 * strength).astype(np.uint8)
    return Image.fromarray(a, mode="L")


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
    bg=(7, 9, 13),
    supersample: int = 2,
) -> Image.Image:
    """Draw the panel. ``trav_bev``/``terrain_bev`` are the flat maps, far field at row 0.

    Geometry is drawn at ``supersample``x and resolved down, because every edge here is a
    polygon and PIL does not anti-alias one. Text is drawn afterwards at final size, where
    it is already sharp and does not pay the downsample.
    """
    w, h = size
    s = max(int(supersample), 1)
    cam = cam or VirtualCam()
    big = _draw_geometry(
        trav_bev,
        terrain_bev,
        grid,
        objects,
        (w * s, h * s),
        cam,
        trav_colors,
        terrain_colors,
        bg,
    )
    panel = big.resize((w, h), Image.LANCZOS) if s > 1 else big
    _draw_annotations(panel, terrain_bev, grid, objects, cam, terrain_colors, class_names)
    return panel


def _draw_geometry(
    trav_bev, terrain_bev, grid, objects, size, cam, trav_colors, terrain_colors, bg
) -> Image.Image:
    w, h = size
    cam = _fit_cam(cam, grid, size)
    panel = Image.new("RGB", (w, h), bg)

    # --- the floor, warped ---------------------------------------------------
    src = _floor_raster(trav_bev, terrain_bev, trav_colors, terrain_colors, bg)
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
    scr = [tuple(map(float, cam.project(x, 0.0, z)[:2])) for x, z in corners_m]
    coeffs = _perspective_coeffs(scr, corners_px)
    # NEAREST, not BILINEAR: this raster is a label map wearing colours, and interpolating
    # across a class edge invents a colour that belongs to no class. Supersampling is what
    # keeps the edge smooth, and it does it without inventing anything.
    ground = Image.fromarray(src).transform(
        (w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.NEAREST, fillcolor=bg
    )
    hit = (np.asarray(ground) != np.array(bg)).any(-1)
    mask = Image.fromarray((hit * 255).astype(np.uint8))
    panel.paste(ground, (0, 0), mask)

    d = ImageDraw.Draw(panel, "RGBA")
    px = max(w // 380, 1)  # line weight that survives the downsample

    # --- distance rules, faint: the scene is the subject, not the graph paper ---
    for z in range(int(np.ceil(grid.z_min)), int(grid.z_max) + 1):
        xs = np.linspace(grid.x_min, grid.x_max, 40)
        pts = [tuple(map(float, cam.project(x, 0.0, float(z))[:2])) for x in xs]
        d.line(pts, fill=(120, 150, 190, 34), width=px)

    # --- the boundary, raised as one surface per class run ---------------------
    angles, reach = boundary_rays(trav_bev, grid)
    classes = []
    for i in range(len(angles)):
        cls = None
        if terrain_bev is not None and reach[i] > 0:
            p = (reach[i] * np.sin(angles[i]), reach[i] * np.cos(angles[i]))
            cell = grid.to_cell(*p)
            if cell is not None:
                r, c = cell
                v = int(terrain_bev[max(r - 3, 0), c])
                cls = v if v != IGNORE else None
        classes.append(cls if reach[i] > 0 else "gap")

    strips = []
    for a, b, cls in _runs(classes):
        if cls == "gap" or b - a < 2:
            continue
        ht = CLASS_HEIGHT_M.get(cls, DEFAULT_HEIGHT_M) if cls is not None else DEFAULT_HEIGHT_M
        if ht <= 0.01:
            ht = 0.25  # something ended the floor even where the class list says it is flat
        colour = (
            tuple(int(v) for v in np.asarray(terrain_colors)[cls])
            if (cls is not None and terrain_colors is not None)
            else (150, 165, 190)
        )
        pts = [
            (reach[i] * np.sin(angles[i]), reach[i] * np.cos(angles[i])) for i in range(a, b)
        ]
        bpts = [tuple(map(float, cam.project(x, 0.0, z)[:2])) for x, z in pts]
        depth = float(np.mean([cam.project(x, 0.0, z)[2] for x, z in pts]))
        # Three stacked bands instead of one flat quad: a wall lit evenly reads as a
        # cut-out, and the gradient is what makes it read as a standing surface.
        bands = []
        for k in range(3):
            y0, y1 = ht * k / 3, ht * (k + 1) / 3
            lo = [tuple(map(float, cam.project(x, y0, z)[:2])) for x, z in pts]
            hi = [tuple(map(float, cam.project(x, y1, z)[:2])) for x, z in pts]
            bands.append((lo + hi[::-1], 0.34 + 0.26 * k))
        cap = [tuple(map(float, cam.project(x, ht, z)[:2])) for x, z in pts]
        strips.append((depth, bands, cap, colour, bpts))

    for _, bands, cap, colour, _bp in sorted(strips, key=lambda q: -q[0]):  # far first
        for poly, f in bands:
            d.polygon(poly, fill=(*_shade(colour, f), 232))
        # the bright top edge is the whole Tesla trick: it is what makes an extrusion read
        # as an edge you could bump into rather than as a coloured region
        d.line(cap, fill=(*_shade(colour, 1.6), 255), width=px * 2, joint="curve")

    # --- objects, solid and standing on the floor ------------------------------
    for obj in sorted(objects, key=lambda o: -(o.get("z_m") or 0.0)):
        x, z = obj.get("x_m"), obj.get("z_m")
        if x is None or z is None:
            continue
        if not (grid.x_min <= x <= grid.x_max and grid.z_min <= z <= grid.z_max):
            continue
        half = max(float(obj.get("width_m") or 0.5), 0.15) / 2
        ht = float(obj.get("height_m") or 1.7)
        col = OBJECT_RGB.get(str(obj.get("name", "")), OBJECT_RGB["_"])

        # contact shadow first, so the volume sits on the floor instead of hovering
        sh = [
            tuple(
                map(
                    float,
                    cam.project(x + half * 1.5 * np.cos(t), 0.0, z + half * 1.5 * np.sin(t))[
                        :2
                    ],
                )
            )
            for t in np.linspace(0, 2 * np.pi, 20, endpoint=False)
        ]
        d.polygon(sh, fill=(0, 0, 0, 120))

        base = [
            (x - half, z - half),
            (x + half, z - half),
            (x + half, z + half),
            (x - half, z + half),
        ]
        bp = [tuple(map(float, cam.project(px_, 0.0, pz)[:2])) for px_, pz in base]
        tp = [tuple(map(float, cam.project(px_, ht, pz)[:2])) for px_, pz in base]
        # Solid, but deliberately a plain extrusion of the measured footprint and height --
        # not a chair asset. It asserts exactly what the flat map and the box already did.
        for i, f in ((0, 0.52), (1, 0.40), (3, 0.62)):  # skip the hidden rear face
            j = (i + 1) % 4
            d.polygon([bp[i], bp[j], tp[j], tp[i]], fill=(*_shade(col, f), 250))
        d.polygon(tp, fill=(*_shade(col, 0.95), 255))
        d.line([*tp, tp[0]], fill=(*_shade(col, 1.45), 255), width=px * 2, joint="curve")

    # --- ego -------------------------------------------------------------------
    ex, ey = (float(v) for v in cam.project(0.0, 0.0, grid.z_min)[:2])
    r = max(w // 60, 6)
    d.polygon(
        [
            (ex, ey - r * 0.2),
            (ex - r * 0.72, ey + r),
            (ex, ey + r * 0.6),
            (ex + r * 0.72, ey + r),
        ],
        fill=(226, 240, 255, 255),
    )

    panel.paste(Image.new("RGB", (w, h), bg), (0, 0), _vignette((w, h)))
    return panel


def _draw_annotations(
    panel, terrain_bev, grid, objects, cam, terrain_colors, class_names
) -> None:
    """Text, at final resolution. Drawn after the downsample so it stays crisp."""
    w, h = panel.size
    cam = _fit_cam(cam, grid, (w, h))
    d = ImageDraw.Draw(panel, "RGBA")
    f_small = _font(max(h // 75, 10))
    f_label = _font(max(h // 64, 11))

    for z in range(int(np.ceil(grid.z_min)), int(grid.z_max) + 1):
        pu, pv = (float(v) for v in cam.project(grid.x_max, 0.0, float(z))[:2])
        lx = min(max(pu + 7, 2), w - 40)
        d.text(
            (lx, min(max(pv - 7, 2), h - 16)),
            f"{z} m",
            font=f_small,
            fill=(150, 172, 200, 210),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 190),
        )

    for obj in sorted(objects, key=lambda o: -(o.get("z_m") or 0.0)):
        x, z = obj.get("x_m"), obj.get("z_m")
        if x is None or z is None:
            continue
        if not (grid.x_min <= x <= grid.x_max and grid.z_min <= z <= grid.z_max):
            continue
        ht = float(obj.get("height_m") or 1.7)
        lu, lv = (float(v) for v in cam.project(x, ht, z)[:2])
        txt = f"{obj.get('name', '?')}  {obj.get('range_m', 0):.1f} m"
        d.text(
            (lu, lv - h // 52),
            txt,
            font=f_label,
            anchor="mb",
            fill=(232, 243, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0, 205),
        )

    if terrain_bev is not None and terrain_colors is not None and class_names:
        present = [
            int(c) for c in np.unique(terrain_bev) if c != IGNORE and int(c) < len(class_names)
        ]
        step = max(h // 46, 15)
        y = h - 12 - step * len(present)
        for cid in present:
            col = tuple(int(v) for v in np.asarray(terrain_colors)[cid])
            d.rounded_rectangle(
                [12, y, 12 + step // 2, y + step // 2], radius=max(step // 8, 2), fill=col
            )
            d.text(
                (12 + step, y - 1),
                class_names[cid],
                font=f_small,
                fill=(206, 220, 238, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 190),
            )
            y += step
