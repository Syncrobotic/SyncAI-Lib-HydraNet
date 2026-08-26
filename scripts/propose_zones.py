#!/usr/bin/env python3
"""Candidate metre-space zone polygons from the fleet calibration, for a human to confirm.

    nice -n 10 .venv/bin/python scripts/propose_zones.py \\
        --cameras Taichung-cam01,Kaohsiung-cam04,Tao-Hsin-cam02 --out runs/zones01

Per the 2026-08-19 design ruling: *physical* zones (walkable floor, fixture footprints,
entrances) are automatable proposals; *policy* attributes (restricted, max_occupancy, ...)
are proposed-never-decided, so every proposal here carries them as explicit nulls and a
human fills them in. Zones are polygons in metres on the floor (docs/RETAIL.md section 5:
config, not weights), so everything below runs through the per-camera calibration from
runs/onboard01 rather than staying in pixels.

Per camera, from the same daytime plate the calibration used:

1. The current best checkpoint's terrain head labels the (undistorted) plate; `floor`
   pixels lie on the fitted ground plane by construction, so `pixel_to_ground` with the
   calib's pitch/roll/height puts them in metre space exactly. The walkable-floor
   proposal is the outer contour of that projected region on a BEV grid.
2. Fixture footprints are non-floor BEV regions adjacent to the fixture-floor contact
   trace. Only floor pixels are projected -- fixture pixels are elevated and would land
   past their footprint (the shelf-face failure RETAIL.md names) -- so a fixture shows
   up as a hole in the projected floor. A hole enclosed by floor is a closed footprint
   (it may include the occlusion shadow); a region open to the grid border gets the
   visible contact edge extruded a fixed depth, because its far side is unobservable.
   Both cases are stated on the proposal rather than hidden.
3. Entrance candidates come from track birth/death hotspots where a reviewed clip exists
   (runs/offline_tracks01: Kaohsiung-cam04, Taichung-cam11), with clip-boundary births and
   deaths censored. The terrain taxonomy has no door class, so on the other cameras the
   segmentation cannot support a door-edge candidate and none is proposed -- an honest
   "none" beats a guessed rectangle a human then anchors on.

Units discipline: cameras whose calib has `scale_source: unmeasured` have no metres
(DA-V2 raw height only). Their proposals are still emitted -- the *shape* is real -- but
`units` says `dav2_raw` and every note repeats it, because a number that looks like a
metre gets believed as one.

GPU sharing: batch is 1 and inference is a single forward per camera; start the process
under `nice -n 10` when a training run shares the GPU.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_bev3d.floorplan import (  # noqa: E402
    BevGrid,
    shoelace,
    simplify_ring,
)
from syncai_bev3d.plate_calibration import undistort_image  # noqa: E402
from syncai_hydranet.geometry.ground import (  # noqa: E402
    Camera,
    GroundPlane,
    ground_to_pixel,
    pixel_to_ground,
    undistort_points,
)
from syncai_hydranet.utils.visualize import (  # noqa: E402
    crop_box,
    overlay,
    preprocess,
    terrain_palette,
)

# Reviewed offline-tracking runs (see runs/offline_tracks01/NOTES.md). Track birth/death
# is the only evidence of "people appear here" this repo has; the mapping is written out
# because the run directories carry short names and cam11 there is *Taichung*, not
# Kaohsiung (checked against each provenance.json's clip path).
TRACK_RUNS = {
    "Kaohsiung-cam04": Path("runs/offline_tracks01/cam04"),
    "Taichung-cam11": Path("runs/offline_tracks01/cam11"),
}
NO_DOOR_CLASS_NOTE = (
    "no track data for this camera and the terrain taxonomy has no door class, so the "
    "segmentation cannot support a door-edge candidate; none proposed"
)
# Empty policy attributes, spelled out per proposal: the human confirm step fills these.
POLICY_NULL = {
    "restricted": None,
    "max_occupancy": None,
    "dwell_limit_s": None,
    "direction": None,
}

FLOOR_MAX_VERTICES = 30
FIXTURE_MAX_VERTICES = 20
MIN_CONTACT_CELLS = 6  # BEV cells of fixture-floor contact a footprint region needs
MIN_FIXTURE_AREA = 0.15  # units^2; smaller footprints are projection noise
OPEN_FIXTURE_DEPTH = 0.6  # units; assumed depth of a footprint whose far side is unseen
# A footprint whose simple outer contour holds much more area than its raster is a
# concave band wrapping the aisle; it gets bisected along its principal axis instead of
# shipping a floor-sized polygon (a simple polygon cannot carry the hole).
CONCAVE_SPLIT_RATIO = 1.6
CONCAVE_SPLIT_DEPTH = 4
ENTRANCE_CLUSTER_RADIUS = 1.5  # units (m where scaled); greedy event clustering
ENTRANCE_MARGIN = 0.5  # units; padding around a cluster's bounding box


# ---------------------------------------------------------------------------
# model


def load_model(config: str, checkpoint: str, weights: str):
    import torch  # noqa: F401  (device move below needs torch initialised)

    from syncai_hydranet.config import load_config
    from syncai_hydranet.models.hydranet import build_model
    from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
    from syncai_hydranet.utils.device import pick_device

    cfg = load_config(config, [])
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(checkpoint), weights))
    return model, cfg, device


def run_terrain(model, cfg: dict, device, img: Image.Image) -> np.ndarray:
    """Terrain argmax at the source image's resolution, via the shared preprocess."""
    import torch

    size = tuple(cfg["data"]["input_size"])  # (H, W)
    px, _canvas, region = preprocess(img, size)
    with torch.no_grad():
        logits = model.forward(px.to(device))["terrain"][0]
    lab = crop_box(logits.argmax(0).cpu().numpy().astype(np.uint8), region)
    lab_img = Image.fromarray(lab).resize(img.size, Image.Resampling.NEAREST)
    return np.asarray(lab_img, dtype=np.int64)


# ---------------------------------------------------------------------------
# polygons on a BEV grid


def outer_contour(grid: np.ndarray) -> np.ndarray | None:
    """Largest closed contour of a binary grid, in cell-index coordinates."""
    import contourpy

    padded = np.pad(grid.astype(float), 1)
    lines = contourpy.contour_generator(z=padded, name="serial").lines(0.5)
    best, best_area = None, 0.0
    for line in lines:
        if len(line) < 4:
            continue
        area = abs(shoelace(line[:-1]))
        if area > best_area:
            best, best_area = line, area
    if best is None:
        return None
    return best[:-1] - 1.0  # drop the closing repeat and the padding offset


def bisect_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a cell mask in two at the median along its principal axis."""
    rc = np.argwhere(mask).astype(float)
    d = rc - rc.mean(axis=0)
    _w, v = np.linalg.eigh(d.T @ d)
    proj = d @ v[:, -1]
    cut = np.median(proj)
    a = np.zeros_like(mask)
    b = np.zeros_like(mask)
    a[tuple(rc[proj <= cut].astype(int).T)] = True
    b[tuple(rc[proj > cut].astype(int).T)] = True
    return a, b


def close_and_trace(
    grid: np.ndarray, bev: BevGrid, max_vertices: int, close_iters: int = 2
) -> tuple[np.ndarray | None, np.ndarray]:
    """Close sampling gaps, keep the largest component, trace and simplify its outline."""
    g = ndimage.binary_closing(grid, iterations=close_iters)
    cc, n = ndimage.label(g)
    if not n:
        return None, g
    sizes = ndimage.sum_labels(np.ones_like(cc), cc, np.arange(1, n + 1))
    g = cc == (int(sizes.argmax()) + 1)
    contour = outer_contour(g)
    if contour is None:
        return None, g
    poly = simplify_ring(bev.to_metres(contour), tol=bev.cell, max_vertices=max_vertices)
    return poly, g


# ---------------------------------------------------------------------------
# entrances from track birth/death events


def track_events(run_dir: Path, plate_wh: tuple[int, int], k1: float):
    """Censored birth/death foot points from a reviewed tracking run, in plate pixels."""
    from syncai_hydranet.data.video import probe

    tracks_path = run_dir / "ground_truth.json"
    basis = "ground_truth.json (human-reviewed merges)"
    if not tracks_path.exists():
        tracks_path = run_dir / "tracks.json"
        basis = "tracks.json (offline proposals, unreviewed)"
    tracks = json.loads(tracks_path.read_text())
    clip = json.loads((run_dir / "provenance.json").read_text())["clip"]
    try:
        clip_w, clip_h, _ = probe(clip)
    except Exception:  # clip may be gitignored/absent; the boxes measured 1920x1080
        clip_w, clip_h = 1920, 1080
        basis += "; clip missing, frame size assumed 1920x1080"
    last_frame = max(int(f) for frames in tracks.values() for f in frames)
    pw, ph = plate_wh
    sx, sy = pw / clip_w, ph / clip_h
    births, deaths = [], []
    for frames in tracks.values():
        keys = sorted(frames, key=int)
        first, last = int(keys[0]), int(keys[-1])
        if first >= 2:  # born at the clip edge is censoring, not an entrance
            b = frames[keys[0]]
            births.append(((b[0] + b[2]) / 2 * sx, b[3] * sy))
        if last <= last_frame - 2:
            d = frames[keys[-1]]
            deaths.append(((d[0] + d[2]) / 2 * sx, d[3] * sy))
    pts = np.asarray(births + deaths, dtype=float).reshape(-1, 2)
    if len(pts) and abs(k1) > 1e-12:
        pts = undistort_points(pts, k1, (pw / 2.0, ph / 2.0), math.hypot(pw, ph) / 2.0)
    return pts, len(births), len(deaths), Path(clip).name, basis, last_frame + 1


def cluster_events(pts: np.ndarray, radius: float) -> list[np.ndarray]:
    clusters: list[list[np.ndarray]] = []
    for p in pts:
        for c in clusters:
            if math.hypot(*(p - np.mean(c, axis=0))) <= radius:
                c.append(p)
                break
        else:
            clusters.append([p])
    out = [np.asarray(c) for c in clusters]
    return sorted(out, key=len, reverse=True)


# ---------------------------------------------------------------------------
# rendering (PIL only, matching the other scripts in this directory)


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


FLOOR_OUTLINE = (40, 90, 220)
FIXTURE_COLOR = (235, 130, 30)
ENTRANCE_COLOR = (30, 160, 60)
CAMERA_COLOR = (200, 40, 40)


def render_bev(
    bev: BevGrid,
    floor_grid: np.ndarray,
    floor_poly: np.ndarray | None,
    fixtures: list[tuple[str, np.ndarray]],
    entrances: list[tuple[str, np.ndarray, int]],
    unit_label: str,
) -> Image.Image:
    ppu = max(min(540.0 / max(bev.nz * bev.cell, 1e-6), 60.0), 12.0)
    ml, mt, mr, mb = 64, 28, 20, 40
    w = int(bev.nx * bev.cell * ppu) + ml + mr
    h = int(bev.nz * bev.cell * ppu) + mt + mb

    def to_px(pt):
        return (
            ml + (pt[0] - bev.x0) * ppu,
            mt + (bev.z0 + bev.nz * bev.cell - pt[1]) * ppu,
        )

    img = Image.new("RGB", (w, h), (255, 255, 255))
    # floor raster, faint, so the polygon can be judged against the projected evidence
    base = np.where(floor_grid[..., None], (210, 224, 210), (255, 255, 255)).astype(np.uint8)
    raster = Image.fromarray(base[::-1]).resize(
        (int(bev.nx * bev.cell * ppu), int(bev.nz * bev.cell * ppu)),
        Image.Resampling.NEAREST,
    )
    img.paste(raster, (ml, mt))
    draw = ImageDraw.Draw(img, "RGBA")
    f_small, f_mid = _font(12), _font(14)

    # 1-unit grid with coordinates
    gx = int(np.floor(bev.x0))
    while gx <= bev.x0 + bev.nx * bev.cell:
        px = to_px((gx, bev.z0))[0]
        draw.line([(px, mt), (px, h - mb)], fill=(0, 0, 0, 28), width=1)
        draw.text((px - 6, h - mb + 4), f"{gx:g}", fill=(90, 90, 90), font=f_small)
        gx += 1
    gz = int(np.floor(bev.z0))
    while gz <= bev.z0 + bev.nz * bev.cell:
        py = to_px((bev.x0, gz))[1]
        draw.line([(ml, py), (w - mr, py)], fill=(0, 0, 0, 28), width=1)
        draw.text((6, py - 7), f"{gz:g}", fill=(90, 90, 90), font=f_small)
        gz += 1
    draw.text((6, mt - 22), f"z fwd / x lat, {unit_label}", fill=(60, 60, 60), font=f_small)

    def poly_px(poly):
        return [to_px(p) for p in poly]

    if floor_poly is not None:
        draw.polygon(poly_px(floor_poly), outline=FLOOR_OUTLINE, width=3)
    for name, poly in fixtures:
        draw.polygon(poly_px(poly), outline=FIXTURE_COLOR, fill=(*FIXTURE_COLOR, 60), width=2)
        cx, cy = np.mean([to_px(p) for p in poly], axis=0)
        draw.text((cx - 10, cy - 8), name.split("_")[-1], fill=(150, 70, 0), font=f_mid)
    for name, poly, n_ev in entrances:
        draw.polygon(poly_px(poly), outline=ENTRANCE_COLOR, fill=(*ENTRANCE_COLOR, 50), width=3)
        cx, cy = to_px(poly.mean(axis=0))
        draw.text((cx - 30, cy - 8), f"{name} n={n_ev}", fill=(10, 110, 40), font=f_mid)
    # the camera's ground foot is the origin of this frame
    cx, cy = to_px((0.0, 0.0))
    draw.polygon([(cx, cy - 8), (cx - 6, cy + 6), (cx + 6, cy + 6)], fill=CAMERA_COLOR)
    draw.text((cx + 8, cy - 8), "cam", fill=CAMERA_COLOR, font=f_mid)
    return img


def render_page(
    plate: Image.Image,
    labels: np.ndarray,
    palette: np.ndarray,
    label_points_px: list[tuple[str, float, float]],
    bev_img: Image.Image,
    header_lines: list[str],
) -> Image.Image:
    left = overlay(plate, labels, palette, alpha=0.4)
    draw = ImageDraw.Draw(left)
    f_mid = _font(16)
    for text, u, v in label_points_px:
        draw.text(
            (u - 6, v - 9),
            text,
            fill=(255, 255, 255),
            font=f_mid,
            stroke_width=2,
            stroke_fill=(150, 70, 0),
        )
    header_h = 20 + 18 * len(header_lines)
    height = max(left.height, bev_img.height)
    page = Image.new("RGB", (left.width + bev_img.width + 12, header_h + height), "white")
    d = ImageDraw.Draw(page)
    for i, line in enumerate(header_lines):
        d.text((10, 8 + 18 * i), line, fill=(20, 20, 20), font=_font(13))
    page.paste(left, (0, header_h))
    page.paste(bev_img, (left.width + 12, header_h))
    return page


# ---------------------------------------------------------------------------
# per-camera pipeline


def r2(poly: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 2), round(float(z), 2)] for x, z in poly]


def propose_for_camera(camera: str, calib_path: Path, model, cfg, device, args) -> dict:
    calib = json.loads(calib_path.read_text())
    if calib.get("pitch_deg") is None:
        return {"camera": camera, "skipped": "no orientation in calib (onboard failure)"}

    scaled = calib.get("scale") is not None
    units = "m" if scaled else "dav2_raw"
    height = calib["height_m"] if scaled else calib["height_dav2_raw_m"]
    plane = GroundPlane(
        height=float(height),
        pitch=math.radians(calib["pitch_deg"]),
        roll=math.radians(calib["roll_deg"]),
    )
    plate_path = Path(calib["plate_used"])
    rgb = np.asarray(Image.open(plate_path).convert("RGB"))
    ph, pw = rgb.shape[:2]
    k1 = float(calib.get("k1_division_model") or 0.0)
    rgb_u = undistort_image(rgb, k1)
    plate_u = Image.fromarray(rgb_u)
    cam = Camera.from_vfov(ph, pw, float(calib["vfov_assumed_deg"]))

    class_names = list(cfg["data"]["terrain_classes"])
    floor_idx = class_names.index("floor")
    fixture_idx = class_names.index("fixture")
    labels = run_terrain(model, cfg, device, plate_u)

    vfov_note = f"vfov {calib['vfov_assumed_deg']}° ({calib.get('vfov_source', 'assumed')})"
    unit_note = (
        "coordinates in metres"
        if scaled
        else "coordinates in DA-V2 raw shape units, NOT metres (scale unmeasured)"
    )

    # --- floor ---------------------------------------------------------------
    vv, uu = np.mgrid[0:ph, 0:pw].astype(float)
    floor_mask = labels == floor_idx
    fx_img, fz_img = pixel_to_ground(uu[floor_mask], vv[floor_mask], cam, plane)
    ok = (
        np.isfinite(fx_img)
        & np.isfinite(fz_img)
        & (fz_img > 0)
        & (fz_img <= args.max_range)
        & (np.abs(fx_img) <= args.max_range)
    )
    fx_img, fz_img = fx_img[ok], fz_img[ok]
    proposals: list[dict] = []
    if len(fx_img) < 100:
        return {
            "camera": camera,
            "skipped": f"only {len(fx_img)} floor pixels project inside range",
        }
    bev = BevGrid(fx_img, fz_img, cell=args.cell)
    floor_grid = bev.raster(fx_img, fz_img)
    floor_poly, floor_grid_closed = close_and_trace(floor_grid, bev, FLOOR_MAX_VERTICES)
    if floor_poly is not None:
        proposals.append(
            {
                "name_suggestion": "walkable_floor",
                "kind": "floor",
                "polygon_m": r2(floor_poly),
                "confidence_note": (
                    f"outer outline of the projected terrain floor mask "
                    f"({int(floor_mask.sum())} px, BEV area "
                    f"{abs(shoelace(floor_poly)):.1f} {units}^2); clipped to the camera "
                    f"frustum and {args.max_range:g} {units}; fixture holes not "
                    f"subtracted; {vfov_note}; {unit_note}"
                ),
                "policy": dict(POLICY_NULL),
            }
        )

    # --- fixtures ------------------------------------------------------------
    # Footprints are built in BEV space, not per image component: the fixture class
    # routinely connects distinct furniture across the frame, and filling one image
    # component's contact trace claims the whole aisle (measured on Taichung-cam01:
    # one "footprint" the size of the floor). Instead, only the fixture-floor contact
    # edge is projected -- those pixels are on the plane; fixture pixels are elevated
    # and would land past their footprint -- and the edge is extruded a fixed depth
    # into non-floor space, since the far side of a fixture is unobservable from one
    # camera. The extrusion alone still merges everything that surrounds the aisle
    # into one closed ring whose outer contour *is* the floor (also measured on
    # Taichung-cam01), so the extruded area is partitioned by nearest contact-edge
    # run: each connected run of fixture-floor edge owns its own footprint, and
    # adjacent furniture comes back as abutting polygons rather than one ring.
    contact = floor_mask & ndimage.binary_dilation(labels == fixture_idx, iterations=2)
    cy, cx = np.nonzero(contact)
    gx, gz = pixel_to_ground(cx.astype(float), cy.astype(float), cam, plane)
    keep = np.isfinite(gx) & np.isfinite(gz) & (gz > 0) & (gz <= args.max_range)
    trace = ndimage.binary_dilation(bev.raster(gx[keep], gz[keep]), iterations=1)
    free = ~floor_grid_closed
    depth_cells = max(round(OPEN_FIXTURE_DEPTH / args.cell), 1)
    grown = trace.copy()
    for _ in range(depth_cells):
        grown = ndimage.binary_dilation(grown) & (free | trace)
    filled = ndimage.binary_fill_holes(grown)
    assert filled is not None
    region_all = grown | (filled & free)
    trace_cc, n_trace = ndimage.label(trace)
    _dist, (iy, ix) = ndimage.distance_transform_edt(~trace, return_indices=True)
    owner = trace_cc[iy, ix]  # every BEV cell -> its nearest contact-edge run
    tsizes = ndimage.sum_labels(np.ones_like(trace_cc), trace_cc, np.arange(1, n_trace + 1))
    accepted: list[np.ndarray] = []
    skipped_fixtures = 0
    stack: list[tuple[np.ndarray, int]] = []
    for t_id in np.argsort(tsizes)[::-1] + 1:
        if tsizes[t_id - 1] < MIN_CONTACT_CELLS:
            skipped_fixtures += 1
            continue
        stack.append((region_all & (owner == t_id), 0))
    while stack:
        mask, depth = stack.pop()
        rcc, rn = ndimage.label(mask)
        if not rn:
            skipped_fixtures += 1
            continue
        for r_id in range(1, rn + 1):
            region = rcc == r_id
            poly, _g = close_and_trace(region, bev, FIXTURE_MAX_VERTICES, close_iters=1)
            area = abs(shoelace(poly)) if poly is not None else 0.0
            raster_area = float(region.sum()) * args.cell**2
            if poly is None or area < MIN_FIXTURE_AREA:
                skipped_fixtures += 1
                continue
            region_filled = ndimage.binary_fill_holes(region)
            assert region_filled is not None
            enclosed_floor = int((region_filled & floor_grid_closed).sum())
            ring = enclosed_floor > max(4, 0.05 * float(floor_grid_closed.sum()))
            concave = area > CONCAVE_SPLIT_RATIO * raster_area
            if (ring or concave) and depth < CONCAVE_SPLIT_DEPTH:
                stack += [(m, depth + 1) for m in bisect_mask(region)]
                continue
            accepted.append(poly)
    # largest first, so fixture_01 is the one worth checking first
    accepted.sort(key=lambda p: -abs(shoelace(p)))
    fixture_polys: list[tuple[str, np.ndarray]] = []
    for poly in accepted:
        area = abs(shoelace(poly))
        name = f"fixture_{len(fixture_polys) + 1:02d}"
        fixture_polys.append((name, poly))
        proposals.append(
            {
                "name_suggestion": name,
                "kind": "fixture",
                "polygon_m": r2(poly),
                "confidence_note": (
                    f"footprint from the visible fixture-floor contact edge, extruded "
                    f"{OPEN_FIXTURE_DEPTH:g} {units} into unobserved space (enclosed "
                    f"pockets filled); BEV area {area:.1f} {units}^2; far edges beyond "
                    f"the visible contact are assumed -- trim on confirm; {unit_note}"
                ),
                "policy": dict(POLICY_NULL),
            }
        )

    # --- entrances -----------------------------------------------------------
    entrance_polys: list[tuple[str, np.ndarray, int]] = []
    run_dir = TRACK_RUNS.get(camera)
    if run_dir is not None and run_dir.exists():
        pts, n_births, n_deaths, clip_name, basis, n_frames = track_events(
            run_dir, (pw, ph), k1
        )
        entrance_basis = (
            f"{basis}; {n_births} censored-safe births + {n_deaths} deaths over "
            f"{n_frames} frames of {clip_name}"
        )
        if len(pts):
            ex, ez = pixel_to_ground(pts[:, 0], pts[:, 1], cam, plane)
            keep = np.isfinite(ex) & np.isfinite(ez) & (ez > 0) & (ez <= args.max_range)
            events = np.stack([ex[keep], ez[keep]], axis=1)
            for c in cluster_events(events, ENTRANCE_CLUSTER_RADIUS):
                lo = c.min(axis=0) - ENTRANCE_MARGIN
                hi = c.max(axis=0) + ENTRANCE_MARGIN
                poly = np.array(
                    [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
                )
                name = f"entrance_{len(entrance_polys) + 1:02d}"
                entrance_polys.append((name, poly, len(c)))
                proposals.append(
                    {
                        "name_suggestion": name,
                        "kind": "entrance",
                        "polygon_m": r2(poly),
                        "confidence_note": (
                            f"track birth/death hotspot, {len(c)} event(s) from one "
                            f"300-frame clip ({entrance_basis}); mid-floor births can "
                            f"be occlusion fade, not a door -- weak evidence, confirm "
                            f"against the plate; {unit_note}"
                        ),
                        "policy": dict(POLICY_NULL),
                    }
                )
    else:
        entrance_basis = NO_DOOR_CLASS_NOTE

    # --- outputs -------------------------------------------------------------
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_hash = hashlib.sha256(calib_path.read_bytes()).hexdigest()[:12]
    dirty = [f for f in calib.get("flags", []) if f.startswith("dirty_plate")]
    result = {
        "schema": "hydranet-zone-proposals/v1",
        "camera": camera,
        "generated": dt.date.today().isoformat(),
        "calib": str(calib_path),
        "calib_hash": f"sha256:{calib_hash}",
        "calib_flags": calib.get("flags", []),
        "units": units,
        "scale": calib.get("scale"),
        "scale_source": calib.get("scale_source"),
        "checkpoint": args.checkpoint,
        "plate": str(plate_path),
        "entrance_basis": entrance_basis,
        "proposals": proposals,
        "policy_fields_empty": True,
    }
    (out_dir / f"{camera}.zones.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n"
    )

    unit_label = units if scaled else "RAW units (no metres!)"
    bev_img = render_bev(
        bev, floor_grid_closed, floor_poly, fixture_polys, entrance_polys, unit_label
    )
    header = [
        f"{camera}  --  zone proposals (a human confirms; policy fields are null)",
        (
            f"pitch {calib['pitch_deg']}°  roll {calib['roll_deg']}°  "
            f"H {height:g} {units}  scale {calib.get('scale') or 'UNMEASURED'}  "
            f"{vfov_note}  k1 {k1:+.3f}"
        ),
        (
            f"{len(fixture_polys)} fixture + "
            f"{1 if floor_poly is not None else 0} floor + "
            f"{len(entrance_polys)} entrance proposals; "
            f"{skipped_fixtures} candidate regions skipped (area gates)"
            + (f"; flags: {', '.join(dirty)}" if dirty else "")
        ),
    ]
    # label each footprint back on the plate panel, at its BEV centroid's image point
    label_points: list[tuple[str, float, float]] = []
    for name, poly in fixture_polys:
        cx_u, cz_u = poly.mean(axis=0)
        u, v, depth = ground_to_pixel(np.array([cx_u]), np.array([cz_u]), cam, plane)
        if depth[0] > 0 and 0 <= u[0] < pw and 0 <= v[0] < ph:
            label_points.append((name.split("_")[-1], float(u[0]), float(v[0])))
    page = render_page(
        plate_u,
        labels,
        terrain_palette(class_names),
        label_points,
        bev_img,
        header,
    )
    page.save(out_dir / f"{camera}_bev.png")
    return {
        "camera": camera,
        "floor": int(floor_poly is not None),
        "fixtures": len(fixture_polys),
        "fixtures_skipped": skipped_fixtures,
        "entrances": len(entrance_polys),
        "units": units,
    }


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--calib-dir", type=Path, default=Path("runs/onboard01"))
    ap.add_argument("--out", type=Path, default=Path("runs/zones01"))
    ap.add_argument("--config", default="runs/hydranet_retail_security_b03_gdino/config.yaml")
    ap.add_argument("--checkpoint", default="runs/hydranet_retail_security_b03_gdino/best.pt")
    ap.add_argument("--weights", default="ema")
    ap.add_argument(
        "--cameras", default=None, help="comma list; default every calibrated camera"
    )
    ap.add_argument("--cell", type=float, default=0.10, help="BEV cell, in the calib's units")
    ap.add_argument("--max-range", type=float, default=25.0)
    args = ap.parse_args(argv)

    calibs = {p.name[: -len(".calib.json")]: p for p in args.calib_dir.glob("*.calib.json")}
    cameras = (
        [c.strip() for c in args.cameras.split(",") if c.strip()]
        if args.cameras
        else sorted(calibs)
    )
    missing = [c for c in cameras if c not in calibs]
    if missing:
        raise SystemExit(f"no calib json under {args.calib_dir} for: {', '.join(missing)}")

    model, cfg, device = load_model(args.config, args.checkpoint, args.weights)
    rows = []
    for camera in cameras:
        row = propose_for_camera(camera, calibs[camera], model, cfg, device, args)
        rows.append(row)
        if "skipped" in row:
            print(f"{camera:22s}  SKIPPED: {row['skipped']}")
        else:
            print(
                f"{camera:22s}  floor {row['floor']}  fixtures {row['fixtures']:2d} "
                f"(+{row['fixtures_skipped']} skipped)  entrances {row['entrances']}  "
                f"[{row['units']}]"
            )
    (args.out / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")
    done = sum("skipped" not in r for r in rows)
    print(f"\n{done}/{len(rows)} cameras proposed -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
