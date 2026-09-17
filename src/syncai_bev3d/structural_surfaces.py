"""Source-bound, fixed-teacher surface hints for autonomous structural observations."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from syncai_bev3d.render_provenance import sha256
from syncai_bev3d.structural_observations import (
    ObservationConfig,
    _read,
    _similar,
    extract_edges,
    group_tracks,
)

PROMPTS = ("floor", "person", "mirror", "glass", "poster")
MIN_SCORE = 0.5


class SurfaceTeacher:
    """Reuse the repository's pinned SAM 3 inference, with no target-specific prompts."""

    def __init__(self, device="cpu"):
        import torch

        from syncai_bev3d.teachers import sam3

        if device == "cpu":
            torch.set_num_threads(4)
        self.device = device
        self.proc, self.model = sam3.load_sam3(sam3.MODEL_ID, device)
        self.identity = {
            "model": sam3.MODEL_ID,
            "revision": sam3.MODEL_REVISION,
            "prompts": list(PROMPTS),
            "min_score": MIN_SCORE,
            "mask_threshold": 0.5,
            "device": device,
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "teacher_code_sha256": sha256(Path(sam3.__file__)),
        }

    def __call__(self, rgb):
        from syncai_bev3d.teachers.sam3 import segment, vision_features

        image = Image.fromarray(rgb)
        features = vision_features(self.proc, self.model, image, self.device)
        masks, scores = {}, {}
        for prompt in PROMPTS:
            results = segment(
                self.proc,
                self.model,
                image,
                prompt,
                MIN_SCORE,
                self.device,
                vision_embeds=features,
            )
            union = np.zeros(rgb.shape[:2], dtype=bool)
            for mask, _score in results:
                union |= mask
            masks[prompt] = union
            scores[prompt] = [float(score) for _, score in results]
        return masks, scores


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def enrich_surfaces(proposals: dict, teacher, evidence_dir: Path):
    """Predict each selected raw frame, recover weak floor lines, reject occlusions.

    The callback receives RGB pixels only, so neither camera ID nor commissioned
    geometry can enter teacher inference. Output masks are retained, not treated
    as ground truth, and are bound to the exact source frame and teacher revision.
    """
    if proposals.get("schema") != "structural-edge-proposals-v1":
        raise ValueError("expected structural proposals")
    evidence_dir.mkdir(parents=True, exist_ok=False)
    report = copy.deepcopy(proposals)
    config = ObservationConfig(**report["config"])
    floor_config = replace(config, min_gradient=0.003, edge_quantile=0.8)
    selected = [row for row in report["frames"] if row["selected"]]
    identity = copy.deepcopy(teacher.identity)
    progress = {"status": "running", "frames_completed": [], "teacher": identity}
    _write(evidence_dir / "progress.json", progress)
    rows = []
    rejected = []
    records = []
    try:
        for frame in selected:
            rgb, digest = _read(Path(frame["path"]))
            if digest != frame["sha256"]:
                raise ValueError("source frame changed before surface inference")
            masks, scores = teacher(rgb)
            if set(masks) != set(PROMPTS) or any(
                mask.shape != rgb.shape[:2] or mask.dtype != bool for mask in masks.values()
            ):
                raise ValueError("teacher masks must be boolean at source resolution")
            # Native margins scale with the image; erosion excludes mask-boundary
            # artefacts from the weak-edge pass, while dilation excludes people.
            margin = max(2, round(np.hypot(*rgb.shape[:2]) / 400))
            blocked = ndimage.binary_dilation(masks["person"], iterations=margin)
            blocked |= masks["mirror"] | masks["glass"] | masks["poster"]
            floor = ndimage.binary_erosion(masks["floor"], iterations=margin) & ~blocked
            candidates, diagnostic = extract_edges(rgb, floor_config, support_mask=floor)
            original = [row for row in report["observations"] if row["frame_id"] == frame["id"]]
            for candidate in candidates:
                if not any(_similar(candidate, old, margin) for old in original):
                    candidate.update(
                        id=f"floor-{frame['id']}-{len(original):04d}",
                        frame_id=frame["id"],
                        extraction="weak_floor",
                    )
                    original.append(candidate)
            kept_count = 0
            for row in original:
                points = np.asarray(row["points_px"], dtype=int)
                x, y = points.T
                row["floor_fraction"] = float(floor[y, x].mean())
                row["blocked_fraction"] = float(blocked[y, x].mean())
                row["surface_evidence_frame"] = frame["id"]
                row.pop("track_id", None)
                if row["blocked_fraction"] > 0.1:
                    row["rejection_reason"] = "predicted_person_mirror_glass_or_poster"
                    rejected.append(row)
                else:
                    rows.append(row)
                    kept_count += 1
            path = evidence_dir / f"{frame['id']}.npz"
            np.savez_compressed(path, **masks, usable_floor=floor, blocked=blocked)
            records.append(
                {
                    "frame_id": frame["id"],
                    "source_sha256": digest,
                    "mask_file": str(path.resolve()),
                    "mask_sha256": sha256(path),
                    "source_size_px": [rgb.shape[1], rgb.shape[0]],
                    "scores": scores,
                    "mask_pixels": {name: int(mask.sum()) for name, mask in masks.items()},
                    "usable_floor_pixels": int(floor.sum()),
                    "floor_extraction": diagnostic,
                    "weak_floor_candidates": len(candidates),
                    "retained_observations": kept_count,
                }
            )
            progress["frames_completed"].append(frame["id"])
            _write(evidence_dir / "progress.json", progress)
    except Exception as exc:
        progress.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write(evidence_dir / "progress.json", progress)
        raise
    report["observations"] = rows
    report["rejected_observations"] = rejected
    report["tracks"] = group_tracks(
        rows, len(selected), float(np.hypot(*report["image_size_px"])), config.min_persistence
    )
    report["surface_evidence"] = {
        "teacher": identity,
        "frames": records,
        "floor_config": asdict(floor_config),
        "implementation_sha256": sha256(Path(__file__)),
        "status": "predictions_only",
        "remaining_risks": ["missed_occlusion", "shadows", "false_surface_labels"],
    }
    report["summary"].update(
        observations=len(rows),
        rejected_observations=len(rejected),
        persistent_tracks=sum(t["state"] == "persistent" for t in report["tracks"]),
        ambiguous_tracks=sum(t["state"] == "ambiguous_match" for t in report["tracks"]),
        transient_tracks=sum(t["state"] == "transient" for t in report["tracks"]),
    )
    report["status"] = (
        "partial" if report["summary"]["persistent_tracks"] else "insufficient_evidence"
    )
    report["reasons"] = [
        "direction_hypotheses_pending",
        "semantic_predictions_not_geometric_truth",
        "no_metric_scale_estimated",
    ]
    progress["status"] = "completed"
    progress["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode()
    ).hexdigest()
    _write(evidence_dir / "progress.json", progress)
    return report
