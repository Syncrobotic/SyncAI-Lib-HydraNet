"""Reviewed appearance cues for facing, separate from a template's geometric symmetry.

A camera can see a screen or a rear casing even when both headings have nearly the
same silhouette. Observations are bound to the source plate and the individual mask.
No front/back is inferred for an asset whose template does not represent it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from syncai_bev3d.object_instances import image_digest


def mask_identity(instance) -> str:
    h = hashlib.sha256()
    h.update(instance.category.encode())
    h.update(str(instance.mask.shape).encode())
    h.update(np.packbits(instance.mask).tobytes())
    return h.hexdigest()


def axis_only(category: str, variant: str) -> bool:
    return (
        (category in {"phone", "tablet"} and variant == "flat")
        or (category == "stool" and variant == "standard")
        or category == "computer_tower"
    )


def load_observations(path: Path, source: Path, instances, camera: str):
    """Stale observations are reported and ignored; they never turn a new detection."""
    if not path.exists():
        return {}, "absent"
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("unsupported facing observation schema")
    if data.get("camera") != camera or data.get("source_sha256") != image_digest(source):
        return {}, "stale camera or source plate; review facing again"
    cues = {}
    seen = set()
    stale = 0
    for row in data["observations"]:
        index = row["instance_id"]
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("facing instance ids must be unique nonnegative integers")
        seen.add(index)
        if row.get("visible_side") not in {"front", "back"}:
            raise ValueError("visible_side must be front or back")
        if not row.get("reviewer", "").strip() or not row.get("evidence", "").strip():
            raise ValueError("facing observations need a reviewer and visible evidence")
        if index >= len(instances) or row.get("mask_sha256") != mask_identity(instances[index]):
            stale += 1
            continue
        cues[index] = row
    return cues, f"{len(cues)} valid; {stale} stale observations ignored"


def describe_orientation(record):
    """Preserve legacy silhouette ambiguity while distinguishing an undirected axis."""
    symmetric = axis_only(record["category"], record["variant"])
    record.update(
        axis_heading_deg=float((record["heading_deg"] + 90) % 180 - 90),
        orientation_kind="axis_only" if symmetric else "directed",
        silhouette_heading_ambiguous=record["heading_ambiguous"],
        heading_requires_review=bool(record["heading_ambiguous"] and not symmetric),
        front_back_status="not_represented_by_template" if symmetric else "unverified",
    )
    if symmetric:
        record["heading_evidence"] = (
            "silhouette determines an axis modulo 180 degrees; "
            "this template contains no observable front/back"
        )


def _front_cosine(record, cf):
    yaw = np.radians(record["heading_deg"])
    front = np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])
    centre = np.array(
        [record["x_m"], record["base_m"] + record["dimensions_m"][2] / 2, record["z_m"]]
    )
    toward_camera = np.array([0, cf.plane.height, 0]) - centre
    return float(front @ toward_camera / max(np.linalg.norm(toward_camera), 1e-8))


def apply_observation(mesh, record, cf, cue, score_mesh):
    """Choose between 180-degree alternatives without moving or resizing the asset.

    A grazing view is not a front/back witness. Any flip must retain silhouette IoU
    >= .42 and lose no more than .05 against the unconstrained fit.
    """
    if cue is None:
        return mesh
    record["facing_observation"] = cue
    if record["orientation_kind"] == "axis_only":
        record["facing_result"] = "ignored: front/back is not represented by this template"
        return mesh
    cosine = _front_cosine(record, cf)
    record["front_camera_cosine_before"] = cosine
    if abs(cosine) < 0.15:
        record["facing_result"] = "unresolved: view is too close to edge-on"
        record["heading_requires_review"] = True
        return mesh
    wanted = 1 if cue["visible_side"] == "front" else -1
    needs_flip = cosine * wanted < 0
    old_heading, old_iou = record["heading_deg"], record["silhouette_iou"]
    if needs_flip:
        vertices = mesh[0].copy()
        vertices[:, 0] = 2 * record["x_m"] - vertices[:, 0]
        vertices[:, 2] = 2 * record["z_m"] - vertices[:, 2]
        candidate = vertices, mesh[1]
        candidate_iou = float(score_mesh(candidate))
        record["facing_candidate_iou"] = candidate_iou
        if (
            not np.isfinite(candidate_iou)
            or candidate_iou < 0.42
            or old_iou - candidate_iou > 0.05
        ):
            record["facing_result"] = "unresolved: appearance conflicts with silhouette"
            record["heading_requires_review"] = True
            return mesh
        mesh = candidate
        record["heading_deg"] = float((old_heading + 360) % 360 - 180)
        record["silhouette_iou"] = candidate_iou
        record["opposite_heading_iou"] = old_iou
    record.update(
        heading_ambiguous=False,
        heading_requires_review=False,
        front_back_status="reviewed_appearance",
        heading_evidence="reviewed visible face and silhouette; not automatic semantics",
        facing_result="flipped 180 degrees" if needs_flip else "confirmed existing direction",
        facing_before_heading_deg=old_heading,
        facing_before_iou=old_iou,
    )
    return mesh
