"""Joint tabletop/body refinement, guarded by the devices the surface already supports.

The table's own horizontal mask is a stronger witness than a merchandise-filled body.
Camera scale stays fixed. The result is an estimated rectangular support, not a survey.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull, QhullError

from syncai_bev3d.meshes import Placement, place
from syncai_bev3d.object_instances import load_instances
from syncai_bev3d.object_placement import Support, fit_instance, footprint, overlap
from syncai_bev3d.surfaces import _project


def _mask_at(mask, shape):
    return np.asarray(Image.fromarray(mask).resize(shape[::-1], Image.Resampling.NEAREST), bool)


def _hull(mask):
    r, c = np.nonzero(mask)
    if len(r) < 20:
        return mask.copy()
    points = np.c_[c, r]
    image = Image.new("1", mask.shape[::-1])
    try:
        boundary = points[ConvexHull(points).vertices]
    except QhullError:
        return mask.copy()
    ImageDraw.Draw(image).polygon(list(map(tuple, boundary)), fill=1)
    return np.asarray(image, bool)


def raster(mesh, cf, shape, *, top=False):
    vertices, faces = mesh
    if top:
        vertices = vertices[np.isclose(vertices[:, 1], vertices[:, 1].max())]
    px = _project(vertices, cf, shape)
    image = Image.new("1", shape[::-1])
    if px is None or not np.isfinite(px).all():
        return np.asarray(image, bool)
    draw = ImageDraw.Draw(image)
    if top:
        draw.polygon(list(map(tuple, px[ConvexHull(px).vertices])), fill=1)
    else:
        for face in faces:
            draw.polygon(list(map(tuple, px[face])), fill=1)
    return np.asarray(image, bool)


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def mesh_parameters(mesh, heading):
    vertices, _faces = mesh
    centre = (vertices[:, [0, 2]].min(0) + vertices[:, [0, 2]].max(0)) / 2
    c, s = np.cos(heading), np.sin(heading)
    local = (vertices[:, [0, 2]] - centre) @ np.array([[c, s], [-s, c]])
    w, d = np.ptp(local, axis=0)
    return np.array([*centre, heading, w, d, vertices[:, 1].max()])


def refine_support(mesh, heading, top_mask, body_mask, cf, factory, *, maxiter=100):
    """Estimate a table using both visible top boundary and body, in a bounded search."""
    shape = (180, 320)
    top = _hull(_mask_at(top_mask, shape))
    body = _mask_at(body_mask, shape)
    initial = mesh_parameters(mesh, heading)
    x, z, angle, w, d, h = initial
    cropped = bool(body_mask[-1].any() or body_mask[0].any())

    def build(parameters):
        return place(
            factory(*parameters[3:]), Placement(parameters[0], parameters[1], parameters[2])
        )

    before_top, before_body = (
        iou(raster(mesh, cf, shape, top=True), top),
        iou(raster(mesh, cf, shape), body),
    )
    report = {
        "accepted": False,
        "before_parameters": initial.tolist(),
        "before_top_iou": before_top,
        "before_body_iou": before_body,
        "dimension_source": "joint visible tabletop/body fit; inherited camera scale",
    }
    if before_top >= 0.82:
        return mesh, dict(report, reason="tabletop already aligns; preserve existing geometry")
    bounds = [
        (x - 0.5, x + 0.5),
        (z - 0.5, z + 0.5),
        (angle - 0.6, angle + 0.6),
        (max(0.3, w * 0.7), w * 1.6),
        (max(0.3, d * 0.7), d * 1.6),
        (h, h) if cropped else (max(0.55, h - 0.2), min(1.15, h + 0.2)),
    ]

    def objective(parameters):
        candidate = build(parameters)
        top_score = iou(raster(candidate, cf, shape, top=True), top)
        body_score = iou(raster(candidate, cf, shape), body)
        prior = np.mean(((parameters - initial) / [0.5, 0.5, 0.6, w, d, 0.2]) ** 2)
        return 1 - top_score + 0.7 * (1 - body_score) + 0.01 * prior

    result = differential_evolution(
        objective, bounds, seed=1, popsize=12, maxiter=maxiter, tol=0.001, polish=False
    )
    candidate = build(result.x)
    after_top = iou(raster(candidate, cf, shape, top=True), top)
    after_body = iou(raster(candidate, cf, shape), body)
    report.update(
        after_parameters=result.x.tolist(),
        after_top_iou=after_top,
        after_body_iou=after_body,
        frame_cropped=cropped,
        height_source="preserved: body cropped by frame"
        if cropped
        else "joint top/body estimate",
        height_change_m=float(result.x[5] - h),
    )
    # A measured top can correct an over-wide body. Cap that tradeoff explicitly, then
    # require independently detected devices to retain their support before installing.
    if (
        after_top < 0.75
        or after_top - before_top < 0.08
        or after_body < max(0.45, before_body - 0.15)
    ):
        return mesh, dict(report, reason="insufficient joint improvement")
    if (after_top + 0.7 * after_body) - (before_top + 0.7 * before_body) < 0.08:
        return mesh, dict(report, reason="joint score did not improve")
    report.update(accepted=True, reason="visible top and body jointly refined")
    return candidate, report


def refine_scene_supports(camera, ev, root: Path, items, heading, factory, *, report=None):
    path = root / f"runs/commission01/{camera}/masks/support_tops.npz"
    if not path.exists() or ev.objects is None:
        return items
    source = root / ev.cf.plate_file if ev.cf.plate_file else None
    tops, _metadata = load_instances(path, source=source)
    object_path = path.with_name("object_instances.npz")
    devices = load_instances(object_path, source=source)[0] if object_path.exists() else []
    output = list(items)
    used = set()
    for index, (mesh, key, alpha, shadow) in enumerate(items):
        if key != "display_table":
            continue
        points = np.unique(mesh[0][:, [0, 2]], axis=0)
        if len(ConvexHull(points).vertices) != 4:
            continue  # round tables retain their already fitted circular footprint
        initial_top = raster(mesh, ev.cf, ev.objects.shape, top=True)
        choices = [
            (iou(initial_top, _hull(_mask_at(t.mask, ev.objects.shape))), j, t)
            for j, t in enumerate(tops)
            if j not in used and t.score >= 0.65
        ]
        if not choices:
            continue
        match, top_index, observation = max(choices, key=lambda value: value[0])
        if match < 0.25:
            continue
        used.add(top_index)
        target = _mask_at(observation.mask, ev.objects.shape)
        ids, counts = np.unique(ev.objects[target & (ev.static == 4)], return_counts=True)
        valid = ids > 0
        if not valid.any():
            continue
        oid = int(ids[valid][np.argmax(counts[valid])])
        candidate, audit = refine_support(
            mesh, heading, observation.mask, ev.objects == oid, ev.cf, factory
        )
        audit.update(mesh_index=index, object_id=oid, detector_score=observation.score)
        if audit["accepted"]:
            old_polygon = footprint(audit["before_parameters"])
            new_polygon = footprint(audit["after_parameters"])
            for j, (other, other_key, *_rest) in enumerate(output):
                if j == index or other_key not in {"display_table", "display_shelf", "column"}:
                    continue
                q = np.unique(other[0][:, [0, 2]], axis=0)
                polygon = q[ConvexHull(q).vertices]
                if overlap(new_polygon, polygon) and not overlap(old_polygon, polygon):
                    audit.update(
                        accepted=False, reason="refinement introduces a fixture collision"
                    )
                    break
        anchors = []
        if audit["accepted"]:
            old_support = Support("before", old_polygon, audit["before_parameters"][5])
            new_support = Support("after", new_polygon, audit["after_parameters"][5])
            for device in sorted(devices, key=lambda obj: obj.score, reverse=True):
                if (
                    device.category not in {"laptop", "monitor", "tablet", "phone"}
                    or device.score < 0.70
                ):
                    continue
                pixels = _mask_at(device.mask, initial_top.shape)
                if (pixels & (initial_top | _hull(target))).sum() < 0.15 * pixels.sum():
                    continue
                old_mesh, old = fit_instance(device, ev.cf, [old_support])
                if old_mesh is None:
                    continue
                new_mesh, new = fit_instance(device, ev.cf, [new_support])
                anchors.append(
                    {
                        "category": device.category,
                        "before_iou": old["silhouette_iou"],
                        "after_iou": new.get("silhouette_iou", 0),
                    }
                )
                if new_mesh is None or new["silhouette_iou"] < old["silhouette_iou"] - 0.10:
                    audit.update(
                        accepted=False, reason="existing device loses support or alignment"
                    )
                    break
                if len(anchors) >= 4:
                    break
            audit["device_checks"] = anchors
        if audit["accepted"]:
            output[index] = (candidate, key, alpha, shadow)
        if report is not None:
            report.append(audit)
        print(
            f"  {camera}: tabletop #{oid}: {audit['reason']} "
            f"IoU {audit['before_top_iou']:.2f} -> "
            f"{audit.get('after_top_iou', audit['before_top_iou']):.2f}",
            flush=True,
        )
    return output
