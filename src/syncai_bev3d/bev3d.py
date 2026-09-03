"""A perspective view of the floor map, with what is standing on it drawn standing up.

The flat top-down panel in ``bev.py`` is the honest primitive -- metres, class ids, no
colour -- and it stays that way. This renders it for a person: the floor in perspective,
the boundary raised into a wall, and detections as boxes on the ground rather than dots.

---------------------------------------------------------------------------
THIS IS NOT AN OLDER `scene_mesh`, AND THE CONFUSION IS THE REPOSITORY'S FAULT

`scene_mesh.py` arrived on 2026-08-25 and draws a solid-geometry room: extruded
footprints, real fixtures, contact shadows, a `scene.obj` per camera. It is what
`demo_video.py` and `heads_video.py` put in the right-hand panel, so it is what the
README's figures look like -- and this module's panel, which is what the README figures
looked like *before* that date, therefore reads as the previous generation of the same
thing. It is not, and nothing in the tree said so until 2026-08-30.

**The two take different inputs and answer different questions.**

    bev3d.render(trav_bev, terrain_bev, grid, objects, ...)   arrays, from a live forward pass
    scene_mesh.build_scene_regular(camera, root)              a camera name -> `runs/` artefacts

`scene_mesh` reads what commissioning measured, so it cannot draw a camera that has not
been commissioned -- **8 of the fleet's 48**. This module needs no `camera.json`, no
plate, and no masks: it draws what the network sees in the frame in front of it, which is
the only 3D panel available for the other 40 and for any footage from a camera nobody has
commissioned yet.

So: `scene_mesh` for "what is this camera's measured geometry", this for "what does the
network see here". Retiring either would remove an answer, not a duplicate. The commit
that introduced `scene_mesh` (`7a83e0e`) does not mention this module at all, which is
how a reader arrives at the wrong conclusion honestly.

---------------------------------------------------------------------------

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

from collections import Counter
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import meshes, shading
from .bev import IGNORE, BevGrid, ray_reach
from .shading import View

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
    """Where this panel is *drawn* from, in terms a caller can reason about.

    Unrelated to the camera that took the frame. The parameters are the ones that describe
    a view of a floor -- how high the eye is, how far back it starts, how far it tilts down
    and how far round it swings -- and `view()` turns them into the `shading.View` that
    every other surface in this package is projected through, so the panel and the mesh
    renderers cannot end up with two different cameras again -- which is what the panel
    and `scripts/mesh_preview.py` had done before `shading.py` settled it.

    ``pitch_deg`` and ``orbit_deg`` are what make this read as a room rather than a plot.
    Straight down a floor plan is a diagram; the pitch tilts it until surfaces have faces,
    and the orbit swings the eye off the centreline so the two visible sides of a fixture
    differ. At ``orbit_deg = 0`` the eye sits on the z axis and this reduces exactly to the
    head-on view it had before, which is what the projection tests pin.

    The swing is about the point on the floor the camera already looks at, not about the
    ego, so raising the orbit rotates the scene in place instead of sliding it out of frame.
    """

    height_m: float = 8.5
    setback_m: float = 2.6
    pitch_deg: float = 44.0
    focal_px: float = 620.0
    cx: float = 0.0
    cy: float = 0.0
    orbit_deg: float = 26.0

    def view(self) -> View:
        t = np.radians(float(self.pitch_deg))
        if not 1e-3 < t < np.pi / 2 - 1e-3:
            raise ValueError(
                f"pitch_deg must be within (0, 90) exclusive, got {self.pitch_deg}"
            )
        # The floor point the un-orbited camera aims at, and the horizontal distance back
        # from it. Both follow from the pitch: a camera `height_m` up, tilted `t` down,
        # meets y = 0 at `height_m / tan(t)` ahead of itself.
        reach = self.height_m / np.tan(t)
        pivot = np.array([0.0, 0.0, -self.setback_m + reach])
        psi = np.radians(float(self.orbit_deg))
        eye = pivot + np.array([reach * np.sin(psi), self.height_m, -reach * np.cos(psi)])
        return View(eye, pivot, self.focal_px, self.cx, self.cy)

    def project(self, x, y, z):
        """World (x right, y up from floor, z forward) -> panel pixels, plus depth."""
        return self.view().project(x, y, z)


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


def _fit_view(cam: VirtualCam, grid: BevGrid, size: tuple[int, int]) -> View:
    """The camera's view, sized so the mapped window fills the panel.

    The window is set by ``--range`` at the call site, so a fixed focal length leaves the
    floor as a postage stamp the moment anyone changes it.

    Fitting only ever changes the focal length and the principal point -- `with_intrinsics`
    is the whole reason that method exists. If it could move the eye, the fit would be free
    to answer "does it fit" by choosing a different view, and the picture would stop being
    of the camera the caller asked for.
    """
    w, h = size
    probe = cam.view().with_intrinsics(1.0, 0.0, 0.0)
    top_m = max(CLASS_HEIGHT_M.values())
    corners = np.array(
        [
            [x, y, z]
            for x in (grid.x_min, grid.x_max)
            for z in (grid.z_min, grid.z_max)
            for y in (0.0, top_m)
        ],
        dtype=float,
    )
    uv, _ = probe.project_points(corners)
    us, vs = uv[:, 0], uv[:, 1]
    span_u = max(float(us.max() - us.min()), 1e-6)
    span_v = max(float(vs.max() - vs.min()), 1e-6)
    focal = min(w / span_u, h / span_v) * 0.94
    mid_u = float(us.max() + us.min()) / 2 * focal
    mid_v = float(vs.max() + vs.min()) / 2 * focal
    return probe.with_intrinsics(focal, w / 2 - mid_u, h / 2 - mid_v)


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


def _smooth_reach(reach: np.ndarray, median_w: int = 5, mean_w: int = 5) -> np.ndarray:
    """Take the staircase out of the boundary polyline without moving where it is.

    ``reach`` is quantised twice before it gets here: once by the BEV cell, which is
    2.5 cm of floor, and once by the ray bin, which at 160 rays across the window is
    coarser than the cell in the near field and finer than it in the far field. Neither
    quantisation is information, and both of them land on the *cap line* -- the one bright
    edge the eye follows along a wall -- where a half-cell step reads as a notch in the
    wall rather than as a pixel grid.

    **This does move the drawn boundary, and that is the cost.** `boundary_rays` is
    deliberately the same reduction the flat map filters on so that the two cannot
    disagree, and smoothing here reintroduces a disagreement of up to about half the
    filter's width in range -- centimetres, and only in the drawing. The flat map keeps the
    raw reach, and nothing downstream of a *decision* reads this function.

    Median before mean, and both computed only over bearings that saw floor:

    * The median kills single-ray spikes, which come from one stray cell voting a bearing
      metres past its neighbours. A mean would smear such a spike across its whole window
      instead of removing it.
    * ``reach == 0`` means "no floor on this bearing at all", not "floor at zero metres".
      Averaging a gap in would drag the wall on either side of a doorway across the
      doorway, which is the one artefact here that a viewer would read as real geometry.
    """
    r = np.asarray(reach, dtype=float)
    valid = r > 0
    n = len(r)
    if int(valid.sum()) < 3:
        return r.copy()

    def _filter(src: np.ndarray, width: int, fn) -> np.ndarray:
        out = src.copy()
        half = max(width // 2, 0)
        for i in np.flatnonzero(valid):
            lo, hi = max(int(i) - half, 0), min(int(i) + half + 1, n)
            window = src[lo:hi][valid[lo:hi]]
            if len(window):
                out[i] = fn(window)
        return out

    # Not circular: the rays span the camera's angular window, not a full turn, so the
    # first and last bearings are edges of the view and not neighbours of each other.
    smoothed = _filter(_filter(r, median_w, np.median), mean_w, np.mean)
    smoothed[~valid] = 0.0
    return smoothed


def _smooth_classes(classes: list, width: int = 7) -> list:
    """A majority vote along the bearings, so one wall is one run.

    The boundary is drawn as one polygon per run of equal class, which is what stopped it
    drawing a seam down every wall. That only holds while the classes along a wall actually
    *are* equal, and they are not: the class is read three cells past where the floor
    ended, so a boundary that wanders by a couple of centimetres reads `wall`, `floor_hard`,
    `wall` along a single flat surface. Each flip splits the run, every run under two rays
    is dropped, and the wall comes out as a dashed line of panels with holes between them
    -- which looks like measured structure and is not.

    ``"gap"`` is held out of the vote and never voted onto a bearing. A gap is "no floor on
    this bearing", so there is nothing to raise there; filling one in would draw a wall
    across a doorway, and unlike a seam that would be a claim about the room.
    """
    out = list(classes)
    half = max(width // 2, 0)
    n = len(classes)
    for i in range(n):
        if classes[i] == "gap":
            continue
        window = [c for c in classes[max(i - half, 0) : min(i + half + 1, n)] if c != "gap"]
        if window:
            out[i] = Counter(window).most_common(1)[0][0]
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
    panel = big.resize((w, h), Image.Resampling.LANCZOS) if s > 1 else big
    _draw_annotations(panel, terrain_bev, grid, objects, cam, terrain_colors, class_names)
    return panel


def _draw_geometry(
    trav_bev, terrain_bev, grid, objects, size, cam, trav_colors, terrain_colors, bg
) -> Image.Image:
    w, h = size
    view = _fit_view(cam, grid, size)
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
    scr = [tuple(map(float, view.project(x, 0.0, z)[:2])) for x, z in corners_m]
    coeffs = _perspective_coeffs(scr, corners_px)
    # NEAREST, not BILINEAR -- the module docstring's first bullet carries the argument.
    ground = Image.fromarray(src).transform(
        (w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.NEAREST, fillcolor=bg
    )
    hit = (np.asarray(ground) != np.array(bg)).any(-1)
    px = max(w // 380, 1)  # line weight that survives the downsample
    # Feather the *outline* of the floor, and only the outline. The staircase along the
    # edge is the 2.5 cm BEV cell magnified by the perspective warp, so it is a sampling
    # artefact of the drawing and not a shape the map asserted -- while every edge *inside*
    # the plate is a real class boundary, which is why the colours above stay NEAREST. A
    # blur on the paste mask antialiases the silhouette without moving it and without
    # mixing one class's colour into another's.
    mask = Image.fromarray((hit * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(px * 1.2)
    )
    panel.paste(ground, (0, 0), mask)

    d = ImageDraw.Draw(panel, "RGBA")

    # --- distance rules, faint: the scene is the subject, not the graph paper ---
    for z in range(int(np.ceil(grid.z_min)), int(grid.z_max) + 1):
        xs = np.linspace(grid.x_min, grid.x_max, 40)
        pts = [tuple(map(float, view.project(x, 0.0, float(z))[:2])) for x in xs]
        d.line(pts, fill=(120, 150, 190, 34), width=px)

    # --- the boundary, raised as one surface per class run ---------------------
    angles, reach = boundary_rays(trav_bev, grid)
    reach = _smooth_reach(reach)
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
    classes = _smooth_classes(classes)

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
        bpts = [tuple(map(float, view.project(x, 0.0, z)[:2])) for x, z in pts]
        depth = float(np.mean([view.project(x, 0.0, z)[2] for x, z in pts]))
        # Three stacked bands instead of one flat quad: a wall lit evenly reads as a
        # cut-out, and the gradient is what makes it read as a standing surface.
        bands = []
        for k in range(3):
            y0, y1 = ht * k / 3, ht * (k + 1) / 3
            lo = [tuple(map(float, view.project(x, y0, z)[:2])) for x, z in pts]
            hi = [tuple(map(float, view.project(x, y1, z)[:2])) for x, z in pts]
            bands.append((lo + hi[::-1], 0.34 + 0.26 * k))
        cap = [tuple(map(float, view.project(x, ht, z)[:2])) for x, z in pts]
        strips.append((depth, bands, cap, colour, bpts))

    # --- what is standing on the floor, built before anything else is painted ---
    # Gathered here rather than inside the drawing loop because the contact shadows are one
    # blurred layer for all of them, and that layer has to go down *before* the boundary
    # does. A shadow belongs on the floor; a shadow composited after the walls falls across
    # a wall that stands in front of the thing casting it.
    standing = []
    for obj in sorted(objects, key=lambda o: -(o.get("z_m") or 0.0)):
        x, z = obj.get("x_m"), obj.get("z_m")
        if x is None or z is None:
            continue
        if not (grid.x_min <= x <= grid.x_max and grid.z_min <= z <= grid.z_max):
            continue
        # `meshes.for_object` owns the class-to-shape decision so this file does not grow a
        # second opinion about it, and it returns None only when the payload carries no
        # width or no height -- `bev.scene` leaves those None rather than filling them in.
        # The box below is where those get invented, at the same 0.5 m / 1.7 m this panel
        # has always used, and it asserts exactly what the flat map already did.
        side = max(float(obj.get("width_m") or 0.5), 0.15)
        mesh = meshes.for_object(obj) or meshes.box(
            side, float(obj.get("height_m") or 1.7), side
        )
        standing.append(
            (
                meshes.place(mesh, meshes.Placement(x_m=float(x), z_m=float(z))),
                # `detected_class` and not the raw name: a `--vocab retail` run labels
                # a box `fixture/oven`, and keying the colour off that string paints
                # every renamed object in the fallback grey.
                OBJECT_RGB.get(meshes.detected_class(obj.get("name", "")), OBJECT_RGB["_"]),
                obj,
            )
        )

    panel = Image.alpha_composite(
        panel.convert("RGBA"),
        shading.contact_shadows((w, h), view, [m for m, _, _ in standing], blur_px=px * 2.5),
    ).convert("RGB")
    d = ImageDraw.Draw(panel, "RGBA")

    for _, bands, cap, colour, _bp in sorted(strips, key=lambda q: -q[0]):  # far first
        for poly, f in bands:
            d.polygon(poly, fill=(*_shade(colour, f), 232))
        # the bright top edge is the whole Tesla trick: it is what makes an extrusion read
        # as an edge you could bump into rather than as a coloured region
        d.line(cap, fill=(*_shade(colour, 1.6), 255), width=px * 2, joint="curve")

    # --- objects, solid and standing on the floor ------------------------------
    # Every disc before any figure. The discs are flat on the floor, so a near object's
    # disc painted after a far object's body would lie on top of the body -- an uncertainty
    # radius drawn over the thing it does not describe.
    for _, _, obj in standing:
        sigma = obj.get("position_sigma_m")
        if sigma:
            # The uncertainty gets its own channel on the floor, in the same metres as
            # everything else on it. `analytics/dwell.py` measures this error in *metres*
            # when a shopper's feet are occluded behind a counter, and a confident figure
            # standing at a wrong position is the most convincing wrong number this
            # project can produce. Drawn only when the payload supplies it -- inventing a
            # radius would be the same failure wearing a warning's clothes.
            _draw_ground_disc(d, view, float(obj["x_m"]), float(obj["z_m"]), float(sigma))

    shading.draw_scene(d, view, [(m, col, 250) for m, col, _ in standing], bg=bg)

    # --- ego -------------------------------------------------------------------
    ex, ey = (float(v) for v in view.project(0.0, 0.0, grid.z_min)[:2])
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


def _draw_ground_disc(d, view: View, x: float, z: float, radius_m: float) -> None:
    """The position's error radius, on the floor, in metres."""
    ring = [
        tuple(
            map(
                float,
                view.project(x + radius_m * np.cos(t), 0.0, z + radius_m * np.sin(t))[:2],
            )
        )
        for t in np.linspace(0, 2 * np.pi, 28, endpoint=False)
    ]
    d.polygon(ring, fill=(238, 172, 64, 90))


def _draw_annotations(
    panel, terrain_bev, grid, objects, cam, terrain_colors, class_names
) -> None:
    """Text, at final resolution. Drawn after the downsample so it stays crisp."""
    w, h = panel.size
    view = _fit_view(cam, grid, (w, h))
    d = ImageDraw.Draw(panel, "RGBA")
    f_small = _font(max(h // 75, 10))
    f_label = _font(max(h // 64, 11))

    for z in range(int(np.ceil(grid.z_min)), int(grid.z_max) + 1):
        pu, pv = (float(v) for v in view.project(grid.x_max, 0.0, float(z))[:2])
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
        lu, lv = (float(v) for v in view.project(x, ht, z)[:2])
        # The score belongs here because the *shape* cannot carry it. `meshes.for_object`
        # draws a chair from the name alone, correctly -- the name is what the detector
        # committed to -- so a chair at 0.31 gets the same silhouette as a chair at 0.95,
        # and a viewer discounts `chair 0.31` in a way they cannot discount a mesh. Without
        # this number the panel is more confident than the run it came from, which is the
        # one failure mode here a customer sees before a metric does. Same format as the 2D
        # box labels in `cli/scene.py`, so the two panels read as one picture.
        txt = f"{obj.get('name', '?')}  {obj.get('range_m', 0):.1f} m"
        score = obj.get("score")
        if score is not None:
            txt += f"  {float(score):.2f}"
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
