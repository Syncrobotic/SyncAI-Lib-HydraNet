"""Source-bound image observations for upright entrance panels.

Two visible floor-contact points fix a vertical plane under the existing calibration.
Frame-edge rays then fix panel intervals. Missing lintels remain explicit drawing
conventions; fitting these observations does not independently validate calibration.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from syncai_bev3d.render_provenance import sha256
from syncai_hydranet.geometry.camera_json import CameraFile

KINDS = ("glass_door", "glass", "window")
BOUNDARY_REVIEW_PX = 8.0  # review trigger at the raw image resolution, not site accuracy


def source_identity(root: Path, camera: str) -> dict:
    camera_path = root / f"runs/commission01/{camera}.camera.json"
    cf = CameraFile.load(camera_path)
    if not cf.plate_file:
        raise ValueError("opening controls require a source plate")
    folder = root / f"runs/commission01/{camera}/masks"
    paths = {camera_path, root / cf.plate_file}
    paths.update(root / "runs/commission01" / p for p in cf.mask_files.values())
    paths.update(folder / f"{kind}.png" for kind in (*KINDS, "door", "floor_fill"))
    return {str(p.relative_to(root)): sha256(p) if p.is_file() else None for p in sorted(paths)}


def load_controls(path: Path, root: Path, camera: str) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema") != 1 or data.get("camera") != camera:
        raise ValueError("opening controls have an unsupported schema or camera")
    if data.get("source_identity") != source_identity(root, camera):
        raise ValueError("stale opening controls: plate, calibration or masks changed")
    if not data.get("reviewer") or not data.get("evidence"):
        raise ValueError("opening controls need reviewer and visible evidence")
    cf = CameraFile.load(root / f"runs/commission01/{camera}.camera.json")
    if data.get("image_size_px") != list(cf.image_size_px):
        raise ValueError("opening controls must use raw camera pixels")
    if not data.get("panels"):
        raise ValueError("opening controls need panels")
    ids = [p["id"] for p in data["panels"]]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("opening panel ids must be unique nonempty strings")
    if any(p["kind"] not in KINDS for p in data["panels"]):
        raise ValueError("unsupported opening material")
    if set(data.get("replaces_kinds", [])) != {p["kind"] for p in data["panels"]}:
        raise ValueError("replacement scope must match the observed panel materials")
    return data


def _pixels(value, cf, *, exact=None):
    points = np.asarray(value, float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or (exact is not None and len(points) != exact)
        or not np.isfinite(points).all()
        or (points < 0).any()
        or (points > np.asarray(cf.image_size_px) - 1).any()
    ):
        raise ValueError("opening observations require finite in-frame raw pixel points")
    return points


def _nearest_distance(points, curve):
    return np.sqrt(((points[:, None] - curve[None]) ** 2).sum(axis=2).min(axis=1))


def fit_controls(data: dict, cf):
    """Return candidate surfaces and diagnostics; never infer hinge or swing state."""
    from syncai_bev3d.footprints import _ground
    from syncai_bev3d.surfaces import Surface, _project, _rays

    shape = cf.image_size_px[::-1]
    contact = _pixels(data["floor_contact_px"], cf, exact=2)
    ground = _ground(contact, cf, shape)
    if not np.isfinite(ground).all() or np.max(np.abs(ground)) > 20:
        return [], [{"status": "abstained", "reason": "floor contact misses usable ground"}]
    delta = ground[1] - ground[0]
    length = float(np.linalg.norm(delta))
    if length < 0.15:
        return [], [{"status": "abstained", "reason": "floor-contact baseline too short"}]
    along = delta / length
    normal = np.array([-along[1], along[0]])
    distance = float(ground[0] @ normal)

    def intersect(px):
        rays = _rays(px, cf, shape)
        denom = rays[:, [0, 2]] @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = distance / denom
        hits = rays * t[:, None] + [0, cf.plane.height, 0]
        if not np.isfinite(hits).all() or (t <= 0).any():
            return None
        return hits

    surfaces, rows = [], []
    for panel in data["panels"]:
        row = {
            "kind": panel["kind"],
            "source_id": f"control:{panel['id']}",
            "control_id": panel["id"],
            "status": "abstained",
            "geometry_stage": "source-bound vertical plane fit",
            "evidence": panel.get("evidence", ""),
            "occlusion": panel.get("occlusion", "unspecified"),
            "hinge_and_swing": "unknown; not inferred",
            "metric_scale": "inherits calibration; not surveyed",
            "floor_contact_px": contact.tolist(),
        }
        edges = [_pixels(panel[k], cf) for k in ("left_edge_px", "right_edge_px")]
        hits = [intersect(edge) for edge in edges]
        if any(hit is None for hit in hits):
            row["reason"] = "frame rays do not intersect the supported plane in front"
            rows.append(row)
            continue
        positions = [float(np.median(hit[:, [0, 2]] @ along)) for hit in hits]
        endpoints = np.array([along * v + normal * distance for v in positions])
        width = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        top_px = panel.get("top_edge_px")
        if top_px is not None:
            top_px = _pixels(top_px, cf)
            top_hits = intersect(top_px)
            top = float(np.median(top_hits[:, 1])) if top_hits is not None else -1
            top_source = "observed lintel rays on supported plane"
        else:
            if panel.get("top_visibility") not in {"cropped", "occluded"}:
                raise ValueError("missing lintel must explicitly be cropped or occluded")
            top = 2.4
            top_source = "2.4 m drawing convention; lintel not observed"
        # Floor-to-ceiling panes only in this version. Raised windows need sill controls.
        if panel["kind"] == "window":
            row["reason"] = "raised windows require independent sill controls"
            rows.append(row)
            continue
        if not 0.15 <= width <= 10 or not 0.4 <= top <= 3.5:
            row["reason"] = "frame outside finite drawing bounds"
            rows.append(row)
            continue
        # Use a dense projection of each UPRIGHT edge, not a free image quadrilateral.
        # Residuals reveal incompatible calibration instead of bending the frame to fit.
        residuals = []
        for endpoint, observed in zip(endpoints, edges, strict=True):
            rim = np.array([[endpoint[0], y, endpoint[1]] for y in np.linspace(0, top, 1200)])
            projected = _project(rim, cf, shape)
            if projected is None:
                residuals = []
                break
            residuals.extend(_nearest_distance(observed, projected).tolist())
        if not residuals:
            row["reason"] = "upright frame crosses camera near plane"
            rows.append(row)
            continue
        if top_px is not None:
            rim = np.array(
                [[x, top, z] for x, z in np.linspace(endpoints[0], endpoints[1], 1200)]
            )
            projected = _project(rim, cf, shape)
            if projected is None:
                row["reason"] = "lintel crosses camera near plane"
                rows.append(row)
                continue
            residuals.extend(_nearest_distance(top_px, projected).tolist())
        maximum = float(max(residuals))
        surface = Surface(
            endpoints,
            0.0,
            top,
            panel["kind"],
            None,
            "source-bound floor contact and visible frame edges",
            row["source_id"],
        )
        surfaces.append(surface)
        row.update(
            status="controlled_candidate",
            reason=None,
            review_status="calibration_or_observation_review"
            if maximum > BOUNDARY_REVIEW_PX
            else "candidate_requires_visual_review",
            points_m=endpoints.tolist(),
            bottom_m=0.0,
            height_m=top,
            top_source=top_source,
            iou=None,
            boundary_fit_residual_px={
                "mean": float(np.mean(residuals)),
                "max": maximum,
                "review_trigger": BOUNDARY_REVIEW_PX,
                "scope": "fitting observations; not independent accuracy",
            },
            observed_edges_px=[edge.tolist() for edge in edges]
            + ([top_px.tolist()] if top_px is not None else []),
        )
        rows.append(row)
    # Partial replacement could remove an unrepresented leaf. Preserve all automatic
    # surfaces unless every panel in the explicitly replaced material scope is supported.
    if len(surfaces) != len(data["panels"]):
        for row in rows:
            row.update(status="abstained", reason=row.get("reason") or "incomplete panel group")
        return [], rows
    # One plane, disjoint material intervals: overlapping leaves/panes are contradictory.
    intervals = sorted(tuple(sorted(s.points @ along)) for s in surfaces)
    if any(b[0] < a[1] - 0.02 for a, b in pairwise(intervals)):
        return [], [
            dict(r, status="abstained", reason="overlapping panel intervals") for r in rows
        ]
    return surfaces, rows


def controls_overlay(data, source: Path, out: Path):
    with Image.open(source) as im:
        canvas = im.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for panel in data["panels"]:
        colour = "cyan" if panel["kind"] == "glass_door" else "lime"
        for key in ("left_edge_px", "right_edge_px", "top_edge_px"):
            if panel.get(key) is not None:
                points = [tuple(p) for p in panel[key]]
                draw.line(points, fill=colour, width=2)
                for x, y in points:
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colour)
    draw.line([tuple(p) for p in data["floor_contact_px"]], fill="yellow", width=3)
    canvas.save(out)
