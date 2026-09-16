"""AI scene annotations with instance masks, explicit conflicts and no human claims."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_utils

from .studioa_contract import ENTITY_IDS, VIEWS, contract_sha256, entity_views

SCHEMA = "studioa.ai.v1"
# Each prompt is a candidate instrument, not evidence of calibrated accuracy.
PROMPTS = {
    "floor": ("floor", "floor tiles"),
    "wall": ("wall",),
    "ceiling": ("ceiling",),
    "column": ("structural column", "pillar"),
    "door": ("door", "glass door"),
    "glass_panel": ("glass wall", "glass window"),
    "display_cabinet": ("display cabinet", "glass display cabinet"),
    "display_table": ("display table", "retail display table"),
    "counter": ("counter", "checkout counter", "cash register counter"),
    "laptop": ("open laptop", "laptop computer"),
    "phone": ("smartphone", "mobile phone"),
    "tablet": ("tablet computer", "ipad"),
    "boxed_stock": ("retail product box", "packaged product"),
    "cardboard_box": ("cardboard shipping box", "cardboard carton"),
    "speaker": ("loudspeaker", "portable speaker"),
    "poster": ("advertising poster", "wall poster"),
    "fire_equipment": ("fire extinguisher", "fire hose cabinet"),
    "chair": ("chair", "stool"),
    "person": ("person",),
    "other_monitor": ("computer monitor",),
    "other_shelf": ("open shelving unit",),
}
GROUPS = (
    {"floor", "wall", "ceiling", "column", "door", "glass_panel"},
    {"display_cabinet", "display_table", "counter", "other_shelf"},
    {"laptop", "phone", "tablet", "boxed_stock", "cardboard_box", "speaker", "other_monitor"},
)
SEMANTIC_REGIONS = {"floor", "wall", "ceiling"}
MIN_PIXELS = 40
MIN_SCORE = 0.40
DUPLICATE_IOU = 0.55
CONFLICT_IOU = 0.50
CONFLICT_CONTAINMENT = 0.85


@dataclass
class Candidate:
    entity: str
    mask: np.ndarray
    score: float
    prompts: list[str] = field(default_factory=list)
    area: int = field(init=False)
    bounds: tuple[int, int, int, int] = field(init=False)

    def __post_init__(self):
        if self.mask.ndim != 2 or self.mask.dtype != bool:
            raise ValueError("candidate requires a two-dimensional boolean mask")
        ys, xs = np.where(self.mask)
        self.area = len(xs)
        self.bounds = (
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            if self.area
            else (0, 0, 0, 0)
        )


def policy() -> dict:
    return {
        "schema": SCHEMA,
        "contract_sha256": contract_sha256(),
        "prompts": {name: list(prompts) for name, prompts in PROMPTS.items()},
        "semantic_region_union": sorted(SEMANTIC_REGIONS),
        "semantic_conflicts": "retain nonconflicting pixels; preserve overlap as unresolved",
        "min_score": MIN_SCORE,
        "min_pixels": MIN_PIXELS,
        "duplicate_iou": DUPLICATE_IOU,
        "conflict_iou": CONFLICT_IOU,
        "conflict_containment": CONFLICT_CONTAINMENT,
        "absence_policy": "not_detected never means absent",
        "quality": "AI pseudo-labels; not independently verified accuracy",
    }


def encode(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def decode(rle: dict) -> np.ndarray:
    return mask_utils.decode(
        {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    ).astype(bool)


def overlap(a: Candidate, b: Candidate) -> tuple[float, float]:
    x0, y0 = max(a.bounds[0], b.bounds[0]), max(a.bounds[1], b.bounds[1])
    x1, y1 = min(a.bounds[2], b.bounds[2]), min(a.bounds[3], b.bounds[3])
    intersection = (
        int((a.mask[y0:y1, x0:x1] & b.mask[y0:y1, x0:x1]).sum()) if x1 > x0 and y1 > y0 else 0
    )
    return intersection / max(1, a.area + b.area - intersection), intersection / max(
        1, min(a.area, b.area)
    )


def _record(candidate: Candidate, identity: str) -> dict:
    mask = candidate.mask
    ys, xs = np.where(mask)
    result = {
        "id": identity,
        "entity": candidate.entity,
        "geometry_kind": "semantic_region"
        if candidate.entity in SEMANTIC_REGIONS
        else "instance_candidate",
        "score": candidate.score,
        "prompts": candidate.prompts,
        "segmentation": encode(mask),
        "area": int(mask.sum()),
        "bbox_xywh": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ],
        "role": "unknown",
        "glazed": None,
        "materials": [],
        "basis": "AI text-prompt segmentation; attributes are AI-inferred",
    }
    if candidate.entity == "door" and "glass door" in candidate.prompts:
        result.update(glazed=True, materials=["glass"])
    if candidate.entity == "glass_panel" or "glass display cabinet" in candidate.prompts:
        result["materials"] = ["glass"]
    if candidate.entity == "counter" and any(
        "register" in p or "checkout" in p for p in candidate.prompts
    ):
        result.update(
            role="checkout", role_evidence="AI prompt: " + ", ".join(candidate.prompts)
        )
    return result


def coverage(entities: list[dict], unresolved: list[dict]) -> dict[str, str]:
    counts = Counter(view for row in entities for view in entity_views(row))
    uncertain = {
        view
        for row in unresolved
        for view, rule in VIEWS.items()
        if rule["entity"] in row["possible_entities"]
    }
    if any(row["entity"] == "door" and row.get("glazed") is None for row in entities):
        uncertain.add("glass_door")
    if any(row["entity"] == "counter" and row.get("role") == "unknown" for row in entities):
        uncertain.add("checkout_counter")
    return {
        view: "candidate_present"
        if counts[view]
        else "uncertain"
        if view in uncertain
        else "not_detected"
        for view in VIEWS
    }


def annotate(candidates: list[Candidate], shape: tuple[int, int]) -> dict:
    """Deduplicate same-class masks, retain cross-class ambiguities instead of argmax."""
    kept: list[Candidate] = []
    dropped = Counter()
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if (
            candidate.entity not in PROMPTS
            or candidate.mask.shape != shape
            or candidate.mask.dtype != bool
        ):
            raise ValueError("invalid candidate class/mask")
        if not np.isfinite(candidate.score) or not 0 <= candidate.score <= 1:
            raise ValueError("invalid candidate score")
        if candidate.score < MIN_SCORE or candidate.area < MIN_PIXELS:
            dropped["small_or_low_score"] += 1
            continue
        duplicate = next(
            (
                other
                for other in kept
                if other.entity == candidate.entity
                and overlap(other, candidate)[0] > DUPLICATE_IOU
            ),
            None,
        )
        if duplicate is not None:
            duplicate.prompts = sorted(set(duplicate.prompts + candidate.prompts))
            dropped["same_entity_duplicate"] += 1
        else:
            kept.append(
                Candidate(
                    candidate.entity, candidate.mask, candidate.score, list(candidate.prompts)
                )
            )
    for entity in sorted(SEMANTIC_REGIONS):
        regions = [item for item in kept if item.entity == entity]
        if len(regions) > 1:
            merged = np.zeros(shape, dtype=bool)
            for region in regions:
                merged |= region.mask
            combined = Candidate(
                entity,
                merged,
                min(item.score for item in regions),
                sorted({prompt for item in regions for prompt in item.prompts}),
            )
            kept = [item for item in kept if item.entity != entity] + [combined]
            dropped["semantic_regions_merged"] += len(regions) - 1
    reasons: dict[int, set[str]] = {}
    conflict_pixels: dict[int, np.ndarray] = {}
    alternatives: dict[int, set[str]] = {}
    for i, a in enumerate(kept):
        if a.entity.startswith("other_"):
            reasons.setdefault(i, set()).add("outside_requested_taxonomy")
        for j in range(i):
            b = kept[j]
            iou, contained = overlap(a, b)
            if (
                a.entity != b.entity
                and any({a.entity, b.entity} <= group for group in GROUPS)
                and (iou > CONFLICT_IOU or contained > CONFLICT_CONTAINMENT)
            ):
                for index, other in ((i, b), (j, a)):
                    if kept[index].entity in SEMANTIC_REGIONS:
                        conflict_pixels.setdefault(index, np.zeros(shape, bool))[:] |= (
                            a.mask & b.mask
                        )
                    else:
                        reasons.setdefault(index, set()).add("conflicting_entity_predictions")
                    alternatives.setdefault(index, set()).add(other.entity)
            # A device/person wholly on a detected poster may be printed, not physical.
            for index, obj, background in ((i, a, b), (j, b, a)):
                fraction = contained * min(a.area, b.area) / max(1, obj.area)
                if (
                    background.entity == "poster"
                    and obj.entity in GROUPS[2] | {"person"}
                    and fraction > 0.85
                ):
                    reasons.setdefault(index, set()).add("possible_poster_depiction")
                if (
                    obj.entity == "glass_panel"
                    and background.entity in {"door", "display_cabinet"}
                    and fraction > 0.85
                ):
                    reasons.setdefault(index, set()).add(
                        "possible_glass_part_not_separate_fixed_panel"
                    )
    entities, unresolved = [], []
    for index, candidate in enumerate(kept):
        if index in conflict_pixels:
            conflicted = Candidate(
                candidate.entity, conflict_pixels[index], candidate.score, candidate.prompts
            )
            region = _record(conflicted, f"ai-{index:04d}-conflict")
            region.update(
                reasons=["conflicting_semantic_pixels"],
                possible_entities=sorted({candidate.entity} | alternatives[index]),
            )
            unresolved.append(region)
            candidate = Candidate(
                candidate.entity,
                candidate.mask & ~conflict_pixels[index],
                candidate.score,
                candidate.prompts,
            )
            if candidate.area < MIN_PIXELS:
                dropped["small_semantic_remainder"] += 1
                continue
        row = _record(candidate, f"ai-{index:04d}")
        if index in reasons:
            row.update(
                reasons=sorted(reasons[index]),
                possible_entities=sorted({candidate.entity} | alternatives.get(index, set())),
            )
            unresolved.append(row)
        else:
            entities.append(row)
    counts = Counter(view for entity in entities for view in entity_views(entity))
    return {
        "schema": SCHEMA,
        "annotation_kind": "ai_generated",
        "status": "ai_labeled",
        "contract_sha256": contract_sha256(),
        "image_size_px": list(shape[::-1]),
        "pixel_space": "source_image",
        "entities": entities,
        "unresolved_candidates": unresolved,
        "filtered_counts": dict(dropped),
        "coverage": coverage(entities, unresolved),
        "counts_by_view": dict(counts),
        "human_reviewed": False,
    }


def validate_annotation(data: dict, image_sha256: str, shape: tuple[int, int]) -> None:
    if (
        data.get("schema") != SCHEMA
        or data.get("annotation_kind") != "ai_generated"
        or data.get("contract_sha256") != contract_sha256()
        or data.get("image_sha256") != image_sha256
        or data.get("image_size_px") != list(shape[::-1])
        or data.get("human_reviewed") is not False
    ):
        raise ValueError("AI annotation identity mismatch")
    if data.get("status") != "ai_labeled" or data.get("pixel_space") != "source_image":
        raise ValueError("incomplete AI annotation")
    ids = set()
    for row in data["entities"] + data["unresolved_candidates"]:
        if row["id"] in ids:
            raise ValueError("duplicate AI instance ID")
        ids.add(row["id"])
        mask = decode(row["segmentation"])
        if mask.shape != shape or int(mask.sum()) != row["area"] or not mask.any():
            raise ValueError("AI mask dimensions/area disagree")
        ys, xs = np.where(mask)
        expected = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
        if row["bbox_xywh"] != expected:
            raise ValueError("AI box does not bound its mask")
    counts = Counter(view for row in data["entities"] for view in entity_views(row))
    expected_coverage = coverage(data["entities"], data["unresolved_candidates"])
    if data.get("counts_by_view") != dict(counts) or data.get("coverage") != expected_coverage:
        raise ValueError("AI coverage/counts disagree")


def coco_annotations(data: dict, image_id: int, first_id: int) -> list[dict]:
    """Positive instance pseudo-labels; unresolved masks stay in the companion JSON."""
    return [
        {
            "id": first_id + index,
            "image_id": image_id,
            "category_id": ENTITY_IDS[row["entity"]],
            "segmentation": row["segmentation"],
            "bbox": row["bbox_xywh"],
            "area": row["area"],
            "iscrowd": 0,
            "score": row["score"],
            "annotation_kind": "ai_generated",
            "attributes": {key: row[key] for key in ("role", "glazed", "materials", "prompts")},
        }
        for index, row in enumerate(data["entities"])
    ]
