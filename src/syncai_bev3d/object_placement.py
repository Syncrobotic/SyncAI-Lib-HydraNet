"""Fit individual assets to raw-lens masks on reconstructed support planes.

The support fixes base height. Calibrated silhouette fitting estimates x/z, continuous
yaw and bounded dimensions independently for each detection. A model is emitted only
after silhouette and support containment checks; all rejected fits remain reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

from syncai_bev3d.meshes import Placement, place, shelf_levels
from syncai_bev3d.object_assets import FLOOR_OBJECTS, asset_mesh, plausible_aspect, templates
from syncai_bev3d.object_facing import (
    apply_observation,
    describe_orientation,
    load_observations,
)
from syncai_bev3d.object_instances import OBJECT_PROMPTS, load_instances
from syncai_bev3d.surfaces import _project, _rays


@dataclass
class Support:
    name: str
    polygon: np.ndarray  # actual convex footprint; includes rounded and rotated fixtures
    height: float

    def __post_init__(self):
        self.equations = ConvexHull(self.polygon).equations

    def contains(self, points, margin=0.0):
        equations = self.equations
        return np.all(
            np.asarray(points) @ equations[:, :2].T + equations[:, 2] <= margin + 1e-8
        )


def fixture_supports(items):
    supports = []
    for index, ((vertices, _faces), key, *_rest) in enumerate(items):
        if key not in {"display_table", "display_shelf"}:
            continue
        points = np.unique(vertices[:, [0, 2]], axis=0)
        polygon = points[ConvexHull(points).vertices]
        top = float(vertices[:, 1].max())
        levels = shelf_levels(top) if key == "display_shelf" else [top]
        supports.extend(
            Support(f"{key}_{index}@{level:.3f}", polygon, level) for level in levels
        )
    return supports


def transform(template, parameters, base):
    x, z, yaw, w, d, h = parameters
    vertices, faces = template
    vertices = vertices * [w, h, d] + [0, base, 0]
    return place((vertices, faces), Placement(x, z, yaw))


def footprint(parameters):
    x, z, yaw, w, d, _h = parameters
    corners = np.array([[-w, -d], [w, -d], [w, d], [-w, d]]) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return corners @ np.array([[c, -s], [s, c]]) + [x, z]


def overlap(a, b, *, tolerance=0.01):
    """Convex footprints intersect with positive area (separating axis test)."""
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        normals = np.c_[-edges[:, 1], edges[:, 0]]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        pa, pb = a @ normals.T, b @ normals.T
        penetration = np.minimum(pa.max(0), pb.max(0)) - np.maximum(pa.min(0), pb.min(0))
        if (penetration <= tolerance).any():
            return False
    return True


class Silhouette:
    """Rasterize actual triangles, including gaps between chair legs (not a box hull)."""

    def __init__(self, mask, cf):
        self.cf, self.shape = cf, mask.shape
        r, c = np.nonzero(mask)
        pad = max(8, int(max(np.ptp(r), np.ptp(c)) * 0.6))
        self.origin = np.array([max(0, c.min() - pad), max(0, r.min() - pad)])
        end = np.minimum([mask.shape[1], mask.shape[0]], [c.max() + pad + 1, r.max() + pad + 1])
        self.scale = min(1.0, 144 / max(end - self.origin))
        self.size = tuple(np.maximum(1, np.ceil((end - self.origin) * self.scale).astype(int)))
        crop = (
            Image.fromarray(mask)
            .crop((*self.origin, *end))
            .resize(self.size, Image.Resampling.NEAREST)
        )
        self.mask = np.asarray(crop, bool)

    def score(self, mesh):
        vertices, faces = mesh
        px = _project(vertices, self.cf, self.shape)
        if px is None or not np.isfinite(px).all():
            return 0.0
        px = (px - self.origin) * self.scale
        # Cropping must not reward a huge model extending beyond the comparison ROI.
        if (px < -2).any() or (px > np.array(self.size) + 2).any():
            return 0.0
        im = Image.new("1", self.size)
        draw = ImageDraw.Draw(im)
        for face in faces:
            draw.polygon([tuple(p) for p in px[face]], fill=1)
        pred = np.asarray(im, bool)
        return float((pred & self.mask).sum() / max(1, (pred | self.mask).sum()))


def _on_plane(pixel, height, cf, shape):
    ray = _rays(np.asarray(pixel, float).reshape(1, 2), cf, shape)[0]
    if ray[1] >= -1e-5 or height >= cf.plane.height - 0.1:
        return None
    point = ray * ((height - cf.plane.height) / ray[1])
    return point[[0, 2]] if np.isfinite(point).all() and np.abs(point).max() < 15 else None


def fit_instance(instance, cf, supports, *, floor_mask=None, maxiter=None):
    """Return mesh and audit record, or None and a rejection record."""
    category, mask = instance.category, instance.mask
    if maxiter is None:
        maxiter = 90 if category == "laptop" else 50
    record = {
        "category": category,
        "detector_score": instance.score,
        "prompt": instance.prompt,
        "status": "rejected",
        "dimension_source": "class priors + calibrated silhouette + support height",
    }
    if category not in OBJECT_PROMPTS:
        return None, dict(record, reason="category has no validated automatic prompt")
    if mask.sum() < 20:
        return None, dict(record, reason="too few visible pixels")
    r, c = np.nonzero(mask)
    if (
        r.max() == mask.shape[0] - 1
        or c.min() == 0
        or c.max() == mask.shape[1] - 1
        or r.min() == 0
    ):
        return None, dict(record, reason="object is cropped by the frame")
    floor_object = category in FLOOR_OBJECTS
    if floor_object:
        if floor_mask is None:
            return None, dict(record, reason="missing floor contact evidence")
        floor = np.asarray(
            Image.fromarray(floor_mask).resize(mask.shape[::-1], Image.Resampling.NEAREST), bool
        )
        distance = ndimage.distance_transform_edt(~floor)
        bottom = r >= np.percentile(r, 95)
        if np.percentile(distance[r[bottom], c[bottom]], 30) > max(4, mask.shape[0] * 0.015):
            return None, dict(record, reason="feet have no visible floor support")
    raster = Silhouette(mask, cf)
    pixel = [(c.min() + c.max()) / 2, (r.min() + r.max()) / 2]
    best = None
    for variant, nominal in templates(category):
        nominal = np.array(nominal)
        candidates = []
        choices = [None] if floor_object else supports
        for support in choices:
            base = 0.0 if support is None else support.height
            centre = _on_plane(pixel, base + nominal[2] * 0.5, cf, mask.shape)
            if centre is None or (
                support is not None and not support.contains([centre], margin=max(nominal[:2]))
            ):
                continue
            candidates.append((support, base, centre))
        # A cabinet's many levels can project similarly; compare the nearest candidates
        # using the mask's bottom contact rather than taking an arbitrary first shelf.
        candidates.sort(
            key=lambda entry: (
                np.linalg.norm(entry[2] - entry[0].polygon.mean(0)) if entry[0] else 0
            )
        )
        for support, base, centre in candidates[:6]:
            # Chair geometry has fixed absolute leg radii; normalize the nominal asset
            # instead so fitting and final rendering share the exact same mesh.
            template = asset_mesh(category, *nominal, variant=variant)
            template = (template[0] / nominal[[0, 2, 1]], template[1])
            radius = max(nominal[:2]) * 0.85
            bounds = [
                (centre[0] - radius, centre[0] + radius),
                (centre[1] - radius, centre[1] + radius),
                (-np.pi, np.pi),
                *[(v * 0.70, v * 1.40) for v in nominal],
            ]

            def objective(
                parameters,
                support=support,
                template=template,
                base=base,
                nominal=nominal,
                variant=variant,
            ):
                if not plausible_aspect(category, parameters[3:], nominal, variant):
                    return 2.0
                corners = footprint(parameters)
                if support is not None and not support.contains(corners):
                    return 2.0
                if floor_object and any(
                    s.height > 0.2 and overlap(corners, s.polygon) for s in supports
                ):
                    return 2.0
                mesh = transform(template, parameters, base)
                prior = np.mean(np.log(np.array(parameters[3:]) / nominal) ** 2)
                return 1 - raster.score(mesh) + 0.10 * prior

            result = differential_evolution(
                objective, bounds, seed=7, maxiter=maxiter, popsize=7, tol=0.003, polish=False
            )
            if result.fun >= 1:
                continue
            parameters = result.x
            mesh = transform(template, parameters, base)
            score = raster.score(mesh)
            quality = 1 - result.fun
            if best is None or quality > best[0]:
                flipped = parameters.copy()
                flipped[2] += np.pi
                flip_score = raster.score(transform(template, flipped, base))
                best = (quality, mesh, parameters, base, support, score, variant, flip_score)
    if best is None:
        return None, dict(record, reason="no supported pose fits the mask")
    _quality, mesh, parameters, base, support, score, variant, flip_score = best
    record.update(
        x_m=float(parameters[0]),
        z_m=float(parameters[1]),
        base_m=base,
        heading_deg=float(np.degrees(parameters[2])),
        dimensions_m=list(map(float, parameters[3:])),
        support="floor" if support is None else support.name,
        silhouette_iou=score,
        variant=variant,
        opposite_heading_iou=flip_score,
        heading_ambiguous=bool(abs(score - flip_score) < 0.05),
        heading_evidence="silhouette only; front/back appearance is not measured",
        calibration_scale="inherits commissioned camera scale; not a surveyed measurement",
        aspect_source="template proportions constrained to 0.80-1.25 of nominal ratios",
    )
    if score < 0.42:
        return None, dict(record, reason="silhouette IoU below 0.42")
    record["status"] = "placed"
    nominal_dimensions = np.array(dict(templates(category))[variant])
    fractions = parameters[3:] / nominal_dimensions
    record["dimension_bound_hit"] = bool(
        np.minimum(fractions - 0.7, 1.4 - fractions).min() < 0.02
    )
    describe_orientation(record)
    record["needs_review"] = bool(
        score < 0.60 or record["heading_requires_review"] or record["dimension_bound_hit"]
    )
    return mesh, record


def scene_objects(camera, ev, root: Path, items, *, report=None):
    path = root / f"runs/commission01/{camera}/masks/object_instances.npz"
    if not path.exists():
        return [], set()
    source = root / ev.cf.plate_file if ev.cf.plate_file else None
    instances, metadata = load_instances(path, source=source)
    observations, observation_status = (
        load_observations(path.with_name("object_facing.json"), source, instances, camera)
        if source is not None
        else ({}, "missing source plate")
    )
    supports = fixture_supports(items)
    built = []
    occupied = []
    floor = ev.walk.copy()
    fill = root / f"runs/commission01/{camera}/masks/floor_fill.png"
    if fill.exists():
        with Image.open(fill) as image:
            floor |= np.asarray(image.resize(floor.shape[::-1], Image.Resampling.NEAREST)) > 127
    for index, instance in enumerate(instances):
        mesh, record = fit_instance(instance, ev.cf, supports, floor_mask=floor)
        record["instance_id"] = index
        if mesh is not None:
            record["facing_observations_status"] = observation_status
            mesh = apply_observation(
                mesh,
                record,
                ev.cf,
                observations.get(index),
                Silhouette(instance.mask, ev.cf).score,
            )
            record["needs_review"] = bool(
                record["silhouette_iou"] < 0.60
                or record["heading_requires_review"]
                or record["dimension_bound_hit"]
            )
            polygon = footprint(
                [
                    record["x_m"],
                    record["z_m"],
                    np.radians(record["heading_deg"]),
                    *record["dimensions_m"],
                ]
            )
            bottom = record["base_m"]
            top = bottom + record["dimensions_m"][2]
            if any(
                min(top, hi) - max(bottom, lo) > 0.02 and overlap(polygon, other)
                for other, lo, hi in occupied
            ):
                mesh = None
                record.update(
                    status="rejected", reason="intersects a higher-confidence fitted object"
                )
            else:
                occupied.append((polygon, bottom, top))
        if report is not None:
            report.append(record)
        if mesh is not None:
            record["scene_mesh_index"] = len(items) + len(built)
            built.append((mesh, instance.category, 255, record["base_m"] < 0.05))
        print(
            f"  {camera}: object #{index} {instance.category}: {record['status']} "
            f"{record.get('reason', '')} IoU {record.get('silhouette_iou', 0):.2f}",
            flush=True,
        )
    return built, set(metadata["categories"])
