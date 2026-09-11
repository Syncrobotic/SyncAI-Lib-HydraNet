"""Vertical openings located by floor contact or an already grounded wall plane.

Glass depth often measures the scene behind the pane. These fits use calibrated rays
and a vertical plane instead of lowering the pane's per-pixel depth to the floor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

from syncai_bev3d.footprints import _ground
from syncai_hydranet.geometry.ground import distort_points, undistort_points


@dataclass
class Surface:
    points: np.ndarray  # endpoints (x, z), in floor metres
    bottom: float
    height: float
    kind: str
    iou: float
    source: str


def _contact_plane(points):
    """A straight sill amid occlusion edges; those edges must not rotate the plane."""
    if len(points) < 12:
        return None
    rng = np.random.default_rng(0)
    best = None
    best_count = 0
    for _ in range(160):
        a, b = points[rng.choice(len(points), 2, replace=False)]
        direction = b - a
        length = np.linalg.norm(direction)
        if length < 0.15:
            continue
        along = direction / length
        normal = np.array([-along[1], along[0]])
        selected = np.abs((points - a) @ normal) < 0.10
        count = int(selected.sum())
        if count > best_count:
            best, best_count = selected, count
    if best is None or best_count < 12:
        return None
    supported = points[best]
    centre = np.mean(supported, axis=0)
    _, _, vt = np.linalg.svd(supported - centre, full_matrices=False)
    along, normal = vt
    if np.ptp(supported @ along) < 0.15:
        return None
    return along, normal, float(centre @ normal)


def _rays(px, cf, shape):
    fh, fw = shape
    pts = px * (np.asarray(cf.image_size_px) / [fw, fh])
    if cf.lens is not None:
        lens = cf.lens
        pts = undistort_points(pts, lens.k1, lens.centre_px, lens.radius_px)
    camera = np.c_[
        (pts[:, 0] - cf.camera.cx) / cf.camera.fx,
        (pts[:, 1] - cf.camera.cy) / cf.camera.fy,
        np.ones(len(pts)),
    ]
    rays = camera @ cf.plane.rotation
    rays[:, 1] *= -1
    return rays


def _project(points, cf, shape):
    level = np.asarray(points, float).copy()
    level[:, 1] = cf.plane.height - level[:, 1]
    camera = level @ cf.plane.rotation.T
    if (camera[:, 2] <= 0.05).any():
        return None
    px = camera[:, :2] / camera[:, 2, None]
    px = px * [cf.camera.fx, cf.camera.fy] + [cf.camera.cx, cf.camera.cy]
    if cf.lens is not None:
        lens = cf.lens
        px = distort_points(px, lens.k1, lens.centre_px, lens.radius_px)
    return px * (np.array(shape[::-1]) / cf.image_size_px)


def fit_surface(
    mask, ev, *, kind="door", walls=(), yaw=0.0, support: Surface | None = None
) -> Surface | None:
    """One connected mask -> a grounded door/pane, or a window on a known wall.

    Missing contact and missing wall support produce no surface. The top is a drawn
    convention when cropped by the image; it is not a depth measurement through glass.
    """
    fh, fw = mask.shape
    cols = np.flatnonzero(mask.any(axis=0))
    if len(cols) < 12:
        return None
    rows = np.array([np.flatnonzero(mask[:, c])[-1] for c in cols])
    floor_distance = ndimage.distance_transform_edt(~ev.walk)
    contact = (
        (rows < fh - 2)
        & (cols > 0)
        & (cols < fw - 1)
        & (floor_distance[rows, cols] <= max(3.0, 0.012 * fh))
    )
    g = _ground(np.c_[cols[contact] + 0.5, rows[contact] + 0.5], ev.cf, mask.shape)
    good = np.isfinite(g).all(axis=1) & (np.abs(g).max(axis=1) < 12)
    g = g[good]
    planes = []
    if support is not None:
        along = support.points[1] - support.points[0]
        along = along / np.linalg.norm(along)
        normal = np.array([-along[1], along[0]])
        extent = tuple(sorted(support.points @ along))
        planes.append(
            (along, normal, float(support.points[0] @ normal), "shared glazing plane", extent)
        )
    elif len(g) >= 12:
        plane = _contact_plane(g)
        if plane is not None:
            planes.append((*plane, "floor contact", None))
    if support is None and kind in {"window", "glass", "glass_door"}:
        cy, sy = np.cos(yaw), np.sin(yaw)
        for axis, perp, lo, hi, _thick in walls:
            along = np.array([cy, sy]) if axis == "u" else np.array([-sy, cy])
            normal = np.array([-sy, cy]) if axis == "u" else np.array([cy, sy])
            planes.append((along, normal, perp, "wall plane", (lo, hi)))
    boundary = mask & ~ndimage.binary_erosion(mask)
    r, c = np.nonzero(boundary)
    step = max(1, len(r) // 4000)
    rays = _rays(np.c_[c[::step] + 0.5, r[::step] + 0.5], ev.cf, mask.shape)
    proposals = []
    for along, normal, distance, source, extent in planes:
        denominator = rays[:, [0, 2]] @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = distance / denominator
        hits = rays * t[:, None] + [0, ev.cf.plane.height, 0]
        valid = np.isfinite(hits).all(1) & (t > 0) & (hits[:, 1] >= -0.2) & (hits[:, 1] <= 3.5)
        if valid.sum() < 12:
            continue
        hits = hits[valid]
        lo, hi = np.percentile(hits[:, [0, 2]] @ along, [1, 99])
        if extent is not None and min(hi, extent[1]) - max(lo, extent[0]) < 0.8 * (hi - lo):
            continue
        bottom = max(0.0, float(np.percentile(hits[:, 1], 1))) if kind == "window" else 0.0
        top = float(
            np.clip(
                np.percentile(hits[:, 1], 99), bottom + 0.4 if kind == "window" else 1.9, 3.2
            )
        )
        if mask[:2].any():
            top = max(top, 2.4)
        if not 0.4 <= hi - lo <= 10 or top - bottom < 0.4:
            continue
        points = np.array([along * lo + normal * distance, along * hi + normal * distance])
        corners = np.array([(x, h, z) for x, z in points for h in (bottom, top)])
        # Sample the rim so division-lens curvature is represented in the score too.
        rim = np.concatenate(
            [
                np.linspace(corners[a], corners[b], 24)
                for a, b in [(0, 1), (1, 3), (3, 2), (2, 0)]
            ]
        )
        px = _project(rim, ev.cf, mask.shape)
        if px is None:
            continue
        try:
            hull = px[ConvexHull(px).vertices]
        except Exception:
            continue
        im = Image.new("1", (fw, fh))
        ImageDraw.Draw(im).polygon(list(map(tuple, hull)), fill=1)
        sil = np.asarray(im, bool)
        score = float((sil & mask).sum() / max((sil | mask).sum(), 1))
        if score >= 0.35:
            proposals.append(Surface(points, bottom, top - bottom, kind, score, source))
    return max(proposals, key=lambda p: p.iou) if proposals else None


def scene_surfaces(camera, ev, root: Path, *, walls=(), yaw=0.0) -> list[Surface]:
    """Read explicit glazing masks first; door masks cannot overwrite them."""
    out = []
    covered = np.zeros(ev.walk.shape, bool)
    masks = {}
    filled_ev = ev
    fill_path = root / "runs/commission01" / camera / "masks/floor_fill.png"
    if fill_path.exists():
        with Image.open(fill_path) as image:
            filled = (
                np.asarray(image.resize(ev.walk.shape[::-1], Image.Resampling.NEAREST)) > 127
            )
        filled_ev = replace(ev, walk=ev.walk | filled)

    def fit(mask, *, kind, support=None):
        surface = fit_surface(mask, ev, kind=kind, walls=walls, yaw=yaw, support=support)
        if surface is None and filled_ev is not ev:
            surface = fit_surface(
                mask, filled_ev, kind=kind, walls=walls, yaw=yaw, support=support
            )
            if surface is not None:
                surface.source = "completed floor contact; " + surface.source
        return surface

    for kind in ("glass_door", "window", "glass", "door"):
        relative = ev.cf.mask_files.get(kind, f"{camera}/masks/{kind}.png")
        path = root / "runs/commission01" / relative
        if not path.exists():
            continue
        with Image.open(path) as image:
            mask = np.asarray(image.resize(ev.walk.shape[::-1], Image.Resampling.NEAREST)) > 127
        if kind == "door" and covered.any():
            components, n_components = ndimage.label(mask, np.ones((3, 3)))
            for component in range(1, n_components + 1):
                part = components == component
                if (part & covered).sum() > 0.5 * part.sum():
                    # The reviewed glazing already represents this opening. Ragged
                    # teacher-mask leftovers must not become extra opaque door slabs.
                    mask[part] = False
        mask &= ~covered
        covered |= mask
        masks[kind] = mask
    glazing = np.zeros_like(covered)
    for kind, mask in masks.items():
        if kind != "door":
            glazing |= mask
    groups, n_groups = ndimage.label(glazing, np.ones((3, 3)))
    supports = {}
    for group in range(1, n_groups + 1):
        support = fit(groups == group, kind="glass")
        if support is not None:
            supports[group] = support
    for kind, mask in masks.items():
        labels, count = ndimage.label(mask, np.ones((3, 3)))
        for label in range(1, count + 1):
            part = labels == label
            if part.sum() < 100:
                continue
            group = int(np.argmax(np.bincount(groups[part]), axis=0))
            surface = fit(part, kind=kind, support=supports.get(group))
            if surface is not None and group in supports:
                surface.source += "; " + supports[group].source
            if surface is not None:
                out.append(surface)
    return separate_glazing(out)


def separate_glazing(surfaces: list[Surface]) -> list[Surface]:
    """Reviewed door leaves own their interval; adjacent panes cannot double it."""
    doors = [s for s in surfaces if s.kind == "glass_door"]
    out = []
    for surface in surfaces:
        if surface.kind not in {"glass", "window"}:
            out.append(surface)
            continue
        origin = surface.points[0]
        delta = surface.points[1] - origin
        length = float(np.linalg.norm(delta))
        along = delta / length
        normal = np.array([-along[1], along[0]])
        intervals = [(0.0, length)]
        for door in doors:
            if np.max(np.abs((door.points - origin) @ normal)) > 0.05:
                continue
            if abs(door.bottom - surface.bottom) > 0.10:
                continue
            lo, hi = sorted((door.points - origin) @ along)
            remaining = []
            for start, end in intervals:
                if hi <= start or lo >= end:
                    remaining.append((start, end))
                else:
                    remaining.extend([(start, max(start, lo)), (min(end, hi), end)])
            intervals = remaining
        for start, end in intervals:
            if end - start >= 0.10:
                points = origin + np.array([start, end])[:, None] * along
                out.append(replace(surface, points=points))
    return out


def wall_sections(walls, surfaces, yaw: float, height: float = 2.4):
    """Carve supported door/window apertures out of coplanar wall runs.

    Returns (axis, perp, lo, hi, bottom, top). A raised window preserves its sill;
    an unrelated pane across the room cannot cut a wall just by overlapping in x/z.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    out = []
    for axis, perp, lo, hi, _thick in walls:
        cuts = []
        for surface in surfaces:
            points = surface.points
            uv = np.c_[
                points[:, 0] * cy + points[:, 1] * sy, -points[:, 0] * sy + points[:, 1] * cy
            ]
            along, across = (uv[:, 0], uv[:, 1]) if axis == "u" else (uv[:, 1], uv[:, 0])
            if np.max(np.abs(across - perp)) > 0.35:
                continue
            start, end = max(lo, float(along.min())), min(hi, float(along.max()))
            if end - start > 0.10:
                cuts.append((start, end, surface.bottom, surface.bottom + surface.height))
        boundaries = sorted({lo, hi, *(x for cut in cuts for x in cut[:2])})
        for a, b in pairwise(boundaries):
            if b - a < 0.10:
                continue
            middle = (a + b) / 2
            removed = sorted(
                (max(0.0, low), min(height, high))
                for left, right, low, high in cuts
                if left <= middle <= right
            )
            bottom = 0.0
            for low, high in removed:
                if high <= bottom:
                    continue
                if low - bottom >= 0.10:
                    out.append((axis, perp, a, b, bottom, low))
                bottom = max(bottom, high)
            if height - bottom >= 0.10:
                out.append((axis, perp, a, b, bottom, height))
    return out
