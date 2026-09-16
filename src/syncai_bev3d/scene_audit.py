"""Audit exported scene geometry on the raw image, independently of fitting scores."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from syncai_bev3d.floor_review import floor_height_consistency, visible_floor_mask
from syncai_bev3d.material_review import material_overlap
from syncai_bev3d.render_provenance import sha256
from syncai_bev3d.surfaces import _project
from syncai_hydranet.geometry.camera_json import CameraFile

SCORING_VERSION = "raw-glb-silhouette-v1"
MATERIALS = ("glass_door", "window", "glass", "door", "wall")


def capture_inputs(root: Path, camera: str, *, opening_controls: Path | None = None) -> dict:
    """Include absent optional inputs, so adding a mask also invalidates a baseline."""
    camera_path = root / f"runs/commission01/{camera}.camera.json"
    cf = CameraFile.load(camera_path)
    folder = root / f"runs/commission01/{camera}/masks"
    paths = {
        camera_path,
        root / f"runs/onboard01/{camera}.calib.json",
        root / f"runs/site30k_qa/geometry_cache/{camera}.npz",
    }
    if opening_controls is not None:
        paths.add(opening_controls.resolve())
    paths.update(root / "runs/commission01" / p for p in cf.mask_files.values())
    paths.update(folder.glob("*"))
    paths.update(folder / f"{kind}.png" for kind in MATERIALS)
    paths.update(
        folder / name
        for name in (
            "object_instances.npz",
            "support_tops.npz",
            "object_facing.json",
            "floor_fill.png",
        )
    )
    if cf.plate_file:
        paths.add(root / cf.plate_file)
    code = set((root / "src/syncai_bev3d").rglob("*.py"))
    code.update((root / "src/syncai_hydranet/geometry").rglob("*.py"))
    code.add(root / "tools/commissioning/scene_mesh.py")
    code.add(root / "tools/commissioning/rebuild_geometry.py")
    code.update(root / name for name in ("pyproject.toml", "uv.lock"))

    def identities(files):
        return {
            str(p.relative_to(root)) if p.is_relative_to(root) else str(p): sha256(p)
            if p.is_file()
            else None
            for p in sorted(files)
            if not p.is_dir()
        }

    return {
        "inputs": identities(paths),
        "code": identities(code),
        "scoring_version": SCORING_VERSION,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pillow", "trimesh")
        },
    }


def raw_silhouette(vertices, faces, cf):
    """Union projected triangles; clip near plane and sample curved division-lens edges.

    This is full-frame geometric coverage with no occlusion discount, depth sorting,
    alpha blending or convex-hull completion. It is not the fitter's admission score.
    """
    w, h = cf.image_size_px
    im = Image.new("1", (w, h))
    draw = ImageDraw.Draw(im)
    vertices = np.asarray(vertices, float)
    if not np.isfinite(vertices).all():
        raise ValueError("non-finite exported mesh")
    level = vertices.copy()
    level[:, 1] = cf.plane.height - level[:, 1]
    depth = (level @ cf.plane.rotation.T)[:, 2]
    near = 0.051
    for face in faces:
        polygon = []
        for a, b in zip(face, np.roll(face, -1), strict=True):
            if depth[a] >= near:
                polygon.append(vertices[a])
            if (depth[a] >= near) != (depth[b] >= near):
                t = (near - depth[a]) / (depth[b] - depth[a])
                polygon.append(vertices[a] + t * (vertices[b] - vertices[a]))
        if len(polygon) < 3:
            continue
        rim = np.concatenate(
            [
                np.linspace(a, b, 25, endpoint=False)
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
            ]
        )
        px = _project(rim, cf, (h, w))
        if px is None or not np.isfinite(px).all():
            raise ValueError("exported mesh cannot be projected through the raw lens")
        draw.polygon([tuple(p) for p in px], fill=1)
    return np.asarray(im, bool)


def mask_metrics(prediction, reference):
    intersection = int((prediction & reference).sum())
    predicted, target = int(prediction.sum()), int(reference.sum())
    union = predicted + target - intersection
    return {
        "predicted_pixels": predicted,
        "reference_pixels": target,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": intersection / union if union else None,
        "precision": intersection / predicted if predicted else None,
        "recall": intersection / target if target else None,
    }


def audit_export(
    root, camera, glb, surface_rows, object_rows, support_rows, out, *, regular=True
):
    """Read the GLB back and bind report identifiers to the geometry actually exported."""
    import trimesh

    cf = CameraFile.load(root / f"runs/commission01/{camera}.camera.json")
    scene = trimesh.load(glb, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        raise ValueError("export did not contain a scene")
    nodes = {}
    for name in scene.graph.nodes_geometry:
        matrix, geometry = scene.graph[name]
        mesh = scene.geometry[geometry]
        nodes[name] = (trimesh.transform_points(mesh.vertices, matrix), np.asarray(mesh.faces))
    final = []
    assigned = set()
    for row in surface_rows:
        pieces = row.get("final_surfaces", [])
        if row["status"] == "final_wall_section":
            pieces = [dict(row, surface_id=f"wall:{len(final)}")]
        for piece in pieces:
            names = piece["mesh_nodes"]
            if not names or any(n not in nodes for n in names):
                raise ValueError(
                    f"surface {piece['surface_id']} has missing GLB nodes: {names}"
                )
            if assigned.intersection(names):
                raise ValueError("a GLB node belongs to multiple final surfaces")
            assigned.update(names)
            final.append({**piece, "kind": row["kind"], "source_id": row.get("source_id")})

    # Global fitting ids are distinct from the object-instance stage.
    cache_path = root / f"runs/site30k_qa/geometry_cache/{camera}.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        geometry_provenance: dict = (
            json.loads(str(cache["depth_source"]))
            if "depth_source" in cache.files
            else {
                "status": "legacy_unverified",
                "scope": "no reusable depth source binding",
            }
        )
        geometry_provenance["geometry_signature_present"] = "geometry_signature" in cache.files
        geometry_provenance["plate_sha256_present"] = "plate_sha256" in cache.files
        floor_check = None
        walk_path = (
            root
            / "runs/commission01"
            / cf.mask_files.get("walkable", f"{camera}/masks/walkable.png")
        )
        if {"height", "geom_ok"} <= set(cache.files) and walk_path.is_file():
            with Image.open(walk_path) as im:
                walkable = np.asarray(im.convert("L")) > 127
            review_path = root / f"runs/commission01/{camera}/masks/visible_floor.review.json"
            visible = (
                visible_floor_mask(root, cf, json.loads(review_path.read_text()))
                if review_path.is_file()
                else None
            )
            floor_check = floor_height_consistency(cache, walkable, visible=visible)
        by_object = (
            "objects" in cf.mask_files
            and (root / "runs/commission01" / cf.mask_files["objects"]).is_file()
            and "horiz" in cache.files
        )
    surface_stage = "completed" if regular and by_object else "not_run"
    if surface_stage == "completed":
        expected = {
            n for n in nodes if n.rsplit("_", 1)[0] in {"glass", "door", "door_frame", "wall"}
        }
        if expected != assigned:
            raise ValueError(f"unmapped final surface nodes: {sorted(expected ^ assigned)}")

    overlay = None
    if cf.plate_file and (root / cf.plate_file).is_file():
        with Image.open(root / cf.plate_file) as im:
            overlay = im.convert("RGB").resize(cf.image_size_px)
    coverage = []
    references = {}
    colours = ((0, 210, 240), (120, 180, 255), (50, 220, 130), (255, 160, 40), (230, 110, 230))
    for kind, colour in zip(MATERIALS, colours, strict=True):
        names = [n for piece in final if piece["kind"] == kind for n in piece["mesh_nodes"]]
        prediction = np.zeros(cf.image_size_px[::-1], bool)
        for name in names:
            prediction |= raw_silhouette(*nodes[name], cf)
        path = (
            root / "runs/commission01" / cf.mask_files.get(kind, f"{camera}/masks/{kind}.png")
        )
        row = {
            "kind": kind,
            "mesh_nodes": names,
            "stage": surface_stage,
            "mask_source": str(path.relative_to(root))
            if path.is_relative_to(root)
            else str(path),
        }
        if not path.is_file():
            row["status"] = "missing_reference"
        elif surface_stage != "completed":
            row["status"] = "not_run"
        else:
            with Image.open(path) as im:
                reference = (
                    np.asarray(
                        im.convert("L").resize(cf.image_size_px, Image.Resampling.NEAREST)
                    )
                    > 127
                )
            row.update(status="scored", **mask_metrics(prediction, reference))
            references[kind] = reference
            if not reference.any():
                row["status"] = "empty_reference"
        coverage.append(row)
        Image.fromarray(prediction.astype(np.uint8) * 255).save(
            out / f"scene.{kind}.projection.png"
        )
        if overlay is not None:
            edge = prediction & ~ndimage.binary_erosion(prediction, iterations=2)
            overlay.paste(colour, mask=Image.fromarray(edge.astype(np.uint8) * 255))
    controlled_openings = []
    for row in surface_rows:
        if "control_id" not in row:
            continue
        result = {
            "control_id": row["control_id"],
            "status": row["status"],
            "review_status": row.get("review_status"),
            "reason": row.get("reason"),
            "top_source": row.get("top_source"),
            "boundary_fit_residual_px": row.get("boundary_fit_residual_px"),
            "scope": "source observations used for fitting; not independent validation",
        }
        names = [
            n
            for piece in final
            if piece["source_id"] == row["source_id"]
            for n in piece["mesh_nodes"]
        ]
        result["mesh_nodes"] = names
        if names and row.get("observed_edges_px"):
            prediction = np.zeros(cf.image_size_px[::-1], bool)
            for name in names:
                prediction |= raw_silhouette(*nodes[name], cf)
            boundary = prediction & ~ndimage.binary_erosion(prediction)
            if overlay is not None:
                overlay.paste(
                    (0, 210, 240), mask=Image.fromarray(boundary.astype(np.uint8) * 255)
                )
            points = np.concatenate(row["observed_edges_px"])
            if boundary.any():
                residuals = ndimage.map_coordinates(
                    ndimage.distance_transform_edt(~boundary), points[:, ::-1].T, order=1
                )
                result["final_glb_boundary_residual_px"] = {
                    "mean": float(residuals.mean()),
                    "max": float(residuals.max()),
                }
            else:
                result["review_status"] = "no_visible_exported_boundary"
        controlled_openings.append(result)
    if overlay is not None:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, overlay.width, 22), fill="black")
        draw.text(
            (5, 4),
            f"{camera} | final GLB raw projection | metric scale unverified",
            fill="white",
        )
        overlay.save(out / "scene.surfaces.overlay.png")
    folder = root / f"runs/commission01/{camera}/masks"
    return {
        "schema": 1,
        "camera": camera,
        "scoring_version": SCORING_VERSION,
        "scoring_space": (
            "full raw camera frame; masks resized nearest; triangle edges sampled 25 times; "
            "no occlusion discount"
        ),
        "reference_scope": (
            "all original material-mask pixels, including rejected components; "
            "no threshold tuning or independent ground truth"
        ),
        "calibration_scale": (
            "inherits commissioned calibration; not independently surveyed by this audit"
        ),
        "thresholds": {
            "opening_fit_admission": 0.35,
            "device_fit_admission": 0.42,
            "gate_d_fixture_target_iou": 0.6,
            "gate_d_fixture_target_share": 0.9,
        },
        "surface_stage": surface_stage,
        "surface_fits": surface_rows,
        "final_surfaces": final,
        "controlled_openings": controlled_openings,
        "material_coverage": coverage,
        "geometry_provenance": geometry_provenance,
        "floor_height_consistency": floor_check,
        "material_reference_consistency": {
            "status": "checked",
            **material_overlap(references),
            "interpretation": (
                "overlapping glass and glass-door labels are conflicting references"
            ),
        }
        if {"glass", "glass_door"} <= references.keys()
        else {"status": "unavailable; both material references are required"},
        "devices": {
            "stage": "completed"
            if regular and (folder / "object_instances.npz").is_file()
            else "not_run",
            "candidates": len(object_rows)
            if regular and (folder / "object_instances.npz").is_file()
            else None,
            "placed": sum(r["status"] == "placed" for r in object_rows)
            if regular and (folder / "object_instances.npz").is_file()
            else None,
        },
        "supports": {
            "stage": "completed"
            if regular and (folder / "support_tops.npz").is_file()
            else "not_run",
            "reports": len(support_rows),
        },
        "fixtures": {
            "gate_d_status": "not_evaluated",
            "reason": (
                "material coverage and device fitting scores do not measure per-fixture "
                "Gate D IoU"
            ),
        },
        "glb_sha256": sha256(glb),
        "mesh_nodes": sorted(nodes),
    }
